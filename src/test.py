import json
import os
import pickle

import fsspec
import hydra
import lightning.pytorch as L
import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf

from src.data.ray_data import get_items_map
from src.models import SpecialTokens
from src.utils.tools import patch_fsspec

FILENAMES = {
    "metrics": "metrics.pkl",
    "config": "config.yaml",
    "checkpoint": "checkpoint.ckpt",
    "progress": "checkpoint_manager_snapshot.json",
}

HEAD_FRACTION = 0.01  # head : the 1 % most popular items
# Block sizes that start the bands of PASID_BANDS after head
BAND_EDGES = [2, 10, 50, 200]


def pasid_bands(codes, popularity, n_head):
    """(N, Q + 1) PASID codes, (N,) popularity, number of head items -> (N,) index in
    PASID_BANDS : cold, head (a value of p of its own), or the size of the block of p."""
    p = codes[:, -1]  # (N,)
    values, inverse, block_size = np.unique(p, return_inverse=True, return_counts=True)
    bands = 1 + np.searchsorted(BAND_EDGES, block_size[inverse], side="right")  # (N,)
    bands[p < n_head] = 1  # the head values are 0 .. n_head - 1
    bands[popularity == 0] = 0
    return bands


def target_bands(fs, cfg):
    """(V,) band of every item id (src.lightning_module.PASID_BANDS), -1 for the special
    tokens, from the PASID code of the run's embeddings. Popularity : occurrences in the
    train timelines, as for the PASID code."""
    category = cfg.data.ray_datasets.category
    items = get_items_map(fs, cfg.paths.unique_items_tplt.format(category=category))
    train = pd.read_parquet(
        cfg.paths.timelines_tplt.format(category=category, split="train"),
        filesystem=fs,
        columns=["timeline"],
    )
    counts = train["timeline"].explode().value_counts()
    # item_to_id numbers the items in order, after the special tokens
    products = list(items["item_to_id"])
    popularity = counts.reindex(products).fillna(0).to_numpy()  # (N,)

    path = cfg.paths.semantic_ids_tplt.format(
        emb_method=cfg.dataset.emb_method,
        category=category,
        quant_method="rqkmeans_256-256-256-256_pasid",
    )
    sids = pd.read_parquet(path, filesystem=fs).set_index("product_id").loc[products]
    n_head = int(np.ceil(HEAD_FRACTION * len(popularity)))
    bands = torch.full((len(SpecialTokens) + len(popularity),), -1)
    bands[len(SpecialTokens) :] = torch.from_numpy(
        pasid_bands(sids.to_numpy(), popularity, n_head)
    )
    return bands


def list_runs(fs, folder, prefix):
    """Folders of the runs under folder whose name starts with prefix."""
    return [
        d.rstrip("/")
        for d in fs.ls(folder)
        if os.path.basename(d.rstrip("/")).startswith(prefix)
    ]


def last_checkpoint(fs, run_dir):
    """Run folder -> config, folder and validation metrics of its checkpoint : Ray keeps
    only the last one, the model at the end of the training budget."""
    with fs.open(os.path.join(run_dir, FILENAMES["progress"])) as f:
        result = json.load(f)["checkpoint_results"][-1]
    checkpoint = os.path.join(run_dir, result["checkpoint_dir_name"])
    with fs.open(os.path.join(checkpoint, FILENAMES["config"])) as f:
        cfg = OmegaConf.load(f)
    return cfg, checkpoint, result["metrics"]


@ray.remote(num_gpus=0.5, max_retries=0)
def get_metrics(cfg, ckpt_path, split, ray_datasets, bands):
    patch_fsspec()

    # Load datamodule
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")  # Load valid and test

    # Load lightning module
    model = hydra.utils.instantiate(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    model.full_hydra_config = cfg
    model.target_bands = bands  # metrics per band of the target too

    # Trainer
    trainer = L.Trainer(accelerator="gpu", precision="bf16-mixed")

    fn = trainer.validate if split == "valid" else trainer.test

    # Our own checkpoints : their hyperparameters hold functools.partial (the optimizer),
    # which torch >= 2.6 refuses to unpickle by default
    metrics = fn(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.paths.protocol + "://" + ckpt_path,
        weights_only=False,
    )

    metrics[0]["n_params"] = n_params

    return metrics


def to_pickle(fs, path, data):
    with fs.open(path, "wb") as f:
        pickle.dump(data, f)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(test_config):
    """Evaluate the checkpoint of every run whose name starts with prefix, all at once."""
    ray.init()

    patch_fsspec()
    fs = fsspec.filesystem(test_config.paths.protocol)

    bands = {}  # (category, embeddings) -> its target bands, in the object store
    jobs = {}  # checkpoint -> model config, valid and test metrics (refs)
    runs = list_runs(fs, test_config.paths.model_folder_tplt, test_config.prefix)
    for run_dir in runs:
        try:
            cfg, checkpoint, _ = last_checkpoint(fs, run_dir)
        # Failed before its first validation, or running
        except (FileNotFoundError, IndexError):
            print(f"Skipped, no checkpoint : {os.path.basename(run_dir)}")
            continue

        cfg.data.ray_datasets.which = ["valid", "test"]  # Not train.
        ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)

        # The strata use the SIDs of the run's embeddings (content ablations differ)
        key = cfg.data.ray_datasets.category, cfg.dataset.emb_method
        if key not in bands:
            bands[key] = ray.put(target_bands(fs, cfg))
        model_path = os.path.join(checkpoint, FILENAMES["checkpoint"])
        jobs[checkpoint] = (
            cfg.model,
            [
                get_metrics.remote(cfg, model_path, split, ray_datasets, bands[key])
                for split in ["valid", "test"]
            ],
        )

    for checkpoint, (model_cfg, (valid_metrics, test_metrics)) in jobs.items():
        to_pickle(
            fs,
            os.path.join(checkpoint, FILENAMES["metrics"]),
            {
                "model_cfg": model_cfg,
                "valid_metrics": ray.get(valid_metrics),
                "test_metrics": ray.get(test_metrics),
            },
        )
        print(f"Tested : {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
