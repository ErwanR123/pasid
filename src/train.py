from datetime import datetime
from pprint import pprint

import fsspec
import hydra
import lightning.pytorch as L
import pyarrow.fs
import pyarrow.parquet
import ray
import ray.train
import ray.train.torch
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf
from ray.train.lightning import prepare_trainer

# Imported by name : torch >= 2.12 makes torch.backends a GenericModule, which Ray cannot
# serialize when it ships train() to the workers
from torch.backends import mha

from src.utils.tools import patch_fsspec

OmegaConf.register_new_resolver("eval", eval)


def log_hyperparameters(cfg, model, trainer):
    """Send the config and the number of parameters of the model to the loggers (wandb)."""
    config = OmegaConf.to_container(cfg)
    hparams = {k: config[k] for k in ["model", "data", "trainer", "task_name", "seed"]}
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def train(cfg):
    # Set precision to use tensor cores - highest | high | medium
    torch.set_float32_matmul_precision("high")
    # Disable MHA Fast path (in this version it can NaN because of left-padding)
    mha.set_fastpath_enabled(False)
    # Model init, dropout and sampled negatives. The Ray Data actors (crops, shuffles) stay random
    L.seed_everything(cfg.seed)

    rank_zero_info("Instantiating datamodule")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    rank_zero_info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    # Storing it here so that we can access it in the callback
    model.full_hydra_config = cfg

    rank_zero_info("Instantiating trainer")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer = prepare_trainer(trainer)

    rank_zero_info("Logging hyperparameters!")
    log_hyperparameters(cfg, model, trainer)

    rank_zero_info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Main entry point for training : runs train() on a Ray GPU worker.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: the Ray Result of the run.
    """
    # Before any HDFS access : the HDFS client starts a JVM, whose own SIGSEGVs Ray's
    # signal handler would take for crashes if Ray installed it after the JVM
    ray.init(runtime_env=OmegaConf.to_container(cfg.ray.runtime_env))

    # Budget in epochs over the users -> steps : it follows the size of the category
    patch_fsspec()
    fs = fsspec.filesystem(cfg.paths.protocol)
    split = cfg.paths.timelines_tplt.format(
        category=cfg.dataset.category, split="train"
    )
    with fs.open(split) as f:
        users = pyarrow.parquet.ParquetFile(f).metadata.num_rows
    steps_per_epoch = users // cfg.data.datamodule.train_batch_size
    cfg.trainer.max_steps = cfg.epochs * steps_per_epoch
    cfg.trainer.val_check_interval = cfg.trainer.max_steps  # one validation, at the end

    # Every interpolation, on the driver : the workers get plain values
    OmegaConf.resolve(cfg)

    print(ray.cluster_resources())
    pprint(OmegaConf.to_container(cfg))

    assert cfg.paths.protocol in [
        "viewfs",
        "file",
    ], "This code has only been tested for viewfs:// and file://."
    filesystem = (
        pyarrow.fs.FileSystem.from_uri("viewfs://root")[0]
        if cfg.paths.protocol == "viewfs"
        else None
    )

    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    # Item encoder in the run name : the SID candidate, or "id" for the baseline
    encoder = cfg.model.net.item_encoder
    items = encoder.quant_method if "quant_method" in encoder else "id"
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{cfg.task_name}_{items}_seed{cfg.seed}_{date}"

    cfg.trainer.logger.name = name

    trainable = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train,
        train_loop_config=cfg,
        scaling_config=hydra.utils.instantiate(cfg.ray.scaling_config),
        run_config=hydra.utils.instantiate(
            cfg.ray.run_config,
            name=name,
            storage_filesystem=filesystem,
            callbacks=[],
        ),
        datasets=ray_datasets,
    )

    return trainable.fit()


if __name__ == "__main__":
    main()
