"""Results of the training runs : one row per run at its checkpoint, then the
table of the paper, mean ± std over the seeds of each category and item encoder.

Usage, from the repo root :
    python submit.py src/collect_results.py prefix=tune
    python submit.py src/collect_results.py prefix=final split=test
Reads every run under paths.model_folder_tplt whose name starts with prefix, and the
config saved with its checkpoints. split=valid : the validation metrics logged during
training. split=test : the metrics written by src/test.py next to the checkpoint
(run it first), with the metrics per PASID band of the target.
"""

import os
import pickle

import fsspec
import hydra
import pandas as pd

from src.lightning_module import PASID_BANDS
from src.test import FILENAMES, last_checkpoint, list_runs
from src.utils.tools import patch_fsspec

# Columns of the paper table
PAPER_METRICS = [
    "HR@10",
    "NDCG@10",
    *[f"band/{b}/HR@10" for b in PASID_BANDS],
]


def run_row(fs, run_dir, split):
    """Run folder -> metrics of its checkpoint (the last one) and its hyperparameters."""
    cfg, checkpoint, metrics = last_checkpoint(fs, run_dir)
    step = metrics["step"]
    if split == "test":
        with fs.open(os.path.join(checkpoint, FILENAMES["metrics"]), "rb") as f:
            metrics = pickle.load(f)["test_metrics"][0]
    category = cfg.data.ray_datasets.category
    prefix = f"{split}/{category}/"
    encoder = cfg.model.net.item_encoder
    return {
        "run": os.path.basename(run_dir),
        "category": category,
        "d_model": cfg.model.net.d_model,
        "items": encoder.get("quant_method", "id"),
        "seed": cfg.seed,
        "lr": cfg.model.optimizer.lr,
        "weight_decay": cfg.model.optimizer.weight_decay,
        "dropout": cfg.model.net.dropout,
        "step": step,
        **{k[len(prefix) :]: v for k, v in metrics.items() if k.startswith(prefix)},
    }


def paper_table(table):
    """Runs -> one row per category, dimension and item encoder : mean ± std over the
    seeds, in %."""
    columns = [m for m in PAPER_METRICS if m in table]
    runs = table.groupby(["category", "d_model", "items"])
    mean, std = runs[columns].mean() * 100, runs[columns].std() * 100
    paper = mean.map("{:.2f}".format) + " ± " + std.map("{:.2f}".format)
    paper.insert(0, "seeds", runs.size())
    return paper


@hydra.main(config_path="../configs", config_name="collect_results", version_base="1.3")
def main(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)
    rows = []
    for run_dir in list_runs(fs, config.paths.model_folder_tplt, config.prefix):
        try:
            rows.append(run_row(fs, run_dir, config.split))
        # Failed before its first validation, or running
        except (FileNotFoundError, IndexError):
            print(f"Skipped, no checkpoint : {os.path.basename(run_dir)}")
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["category", "d_model", "items", "HR@10"], ascending=[True, True, True, False]
    )
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(table.to_string(index=False))
        print()
        print(paper_table(table).to_string())


if __name__ == "__main__":
    main()
