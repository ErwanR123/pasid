import os
import pickle
from functools import partial

import fsspec
import hydra
import numpy as np
import pandas as pd
import ray
from sentence_transformers import SentenceTransformer

from src.utils.tools import patch_fsspec


def preprocess(row, eos, max_chars=None):
    """Metadata row -> text of the item, one field per line, cut to its first max_chars
    characters if set (content ablation)."""
    lines = []

    if row["title"]:
        lines.append(f"Title: {row['title']}")

    if row["store"]:
        lines.append(f"Store: {row['store']}")

    if row["price"] and not np.isnan(row["price"]):
        lines.append(f"Price: {row['price']}$")

    if len(row["categories"]):
        # Amazon 2023 : a flat list, from the top category down to the leaf
        lines.append(f"Category: {' > '.join(row['categories'])}")

    if "features" in row and len(row["features"]):
        lines.append(f"Features: {' '.join(row['features'])}")

    return "\n".join(lines)[:max_chars] + eos


@hydra.main(
    config_path="../configs", config_name="data/1_make_embeddings", version_base="1.2"
)
def main(config):
    scaled_embed = ray.remote(embed).options(num_gpus=config.num_gpus)
    ray.get(scaled_embed.remote(config))


def embed(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    # Writing directly to embeddings as it's already filtered
    output_file = config.paths.embeddings_tplt.format(
        emb_method=config.dataset.emb_method, category=config.dataset.category
    )
    print("Outputs will be written to : ", output_file)
    fs.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Try removing the embeddings, we don't want to have multiple versions in the same folder
    if config.force:
        try:
            fs.rm(output_file, recursive=True)
        except (FileNotFoundError, OSError):
            pass
    else:
        if fs.exists(output_file):
            print(f"File already exists for {config.dataset.category}, skipping...")
            return

    print(f"Processing {config.dataset.category}...")

    print("Loading model.")

    model = SentenceTransformer(config.model_folder, trust_remote_code=True).eval()

    print("Model loaded.")

    with fs.open(
        config.paths.unique_items_tplt.format(category=config.dataset.category), "rb"
    ) as f:
        items = pickle.load(f)

    pq_path = config.paths.meta_tplt.format(category=config.dataset.category)
    print("Input parquet path", pq_path)
    input_df = pd.read_parquet(pq_path, filesystem=fs)
    input_df = input_df[input_df["parent_asin"].isin(set(items))]

    print(
        f"Found {len(input_df)} items in the parquet file ({len(items)} in the pickle)."
    )

    sentences = input_df.apply(
        partial(preprocess, eos=model.tokenizer.eos_token, max_chars=config.max_chars),
        axis=1,
    ).values.tolist()
    if config.empty_fraction:
        # Content ablation : these items lose their whole text (missing content)
        empty = np.random.default_rng(0).random(len(sentences)) < config.empty_fraction
        sentences = [
            model.tokenizer.eos_token if e else s for s, e in zip(sentences, empty)
        ]
        print(f"{empty.mean():.1%} items with an empty text")
    print("OK. Start processing...")

    embeddings = model.encode(
        sentences,
        device=config.device,
        normalize_embeddings=True,
        batch_size=config.batch_size,
        show_progress_bar=True,
    )

    out_df = pd.DataFrame(
        {"product_id": input_df["parent_asin"], "embedding": list(embeddings)}
    )

    out_df.to_parquet(output_file, filesystem=fs, index=False)


if __name__ == "__main__":
    main()
