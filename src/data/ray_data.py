import pickle

import fsspec
import numpy as np
import pandas as pd
import ray
import ray.data

from src.data.sasrec import SASRecPrePro
from src.models import SpecialTokens
from src.utils.tools import patch_fsspec


def get_quantized(fs, path):
    df = pd.read_parquet(path, filesystem=fs)
    df = df.set_index("product_id")
    # Reorder : make sure to have L0, L1...
    sorted_cols = sorted([col for col in df.columns if col.startswith("L")])
    df = df[sorted_cols]

    return df


def get_items_map(fs, path):
    with fs.open(path, "rb") as f:
        items = pickle.load(f)

    # Ids after the special tokens, in the order of the catalog
    return {
        "item_to_id": {item: i + len(SpecialTokens) for i, item in enumerate(items)}
    }


def _drop_len_1(row):
    return len(row["timeline"]) > 1


def make_pipeline(fs, split, path, items_ref, prepro_cfg, num_cpus):
    if split == "train":
        # The timelines as item ids, once : src.data.datamodule.TimelineBatches draws
        # the training examples from them on the GPU
        df = pd.read_parquet(path, filesystem=fs, columns=["timeline", "timestamp"])
        df = df[df.timeline.apply(len) > 1]
        item_to_id = ray.get(items_ref)["item_to_id"]
        df["timeline"] = [np.array([item_to_id[i] for i in tl]) for tl in df.timeline]
        ds = ray.data.from_pandas(df)

    else:
        # Valid and test : one example per user, prepared once
        ds = (
            ray.data.read_parquet(path, filesystem=fs, columns=["timeline"])
            .filter(fn=_drop_len_1)
            .map(
                SASRecPrePro,
                fn_constructor_kwargs={**prepro_cfg, "items_ref": items_ref},
                concurrency=num_cpus,
            )
            .materialize()  # Compute once and store
        )

    return ds


def make_ray_dataset(
    category,
    prepro_cfg,
    num_cpus,
    paths,
    which=["train", "valid"],
):
    patch_fsspec()
    fs = fsspec.filesystem(paths.protocol)

    items_path = paths.unique_items_tplt.format(category=category)
    items_ref = ray.put(get_items_map(fs, items_path))

    datasets = {}
    for split in which:
        datasets[split] = make_pipeline(
            fs=fs,
            split=split,
            path=paths.timelines_tplt.format(category=category, split=split),
            items_ref=items_ref,
            prepro_cfg=prepro_cfg,
            num_cpus=num_cpus,
        )
    return datasets
