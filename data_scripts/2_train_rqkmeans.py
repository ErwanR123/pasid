"""RQ-KMeans tokenizer : residual k-means on the item content embeddings.

N items, D embedding dim.
"""

import fsspec
import hydra
import numpy as np
import pandas as pd
import ray
from sklearn.cluster import KMeans

from src.utils.tools import patch_fsspec


@hydra.main(
    config_path="../configs", config_name="data/2_train_rqkmeans", version_base="1.2"
)
def main(config):
    # sklearn parallelizes with OpenMP threads, Ray sets them to num_cpus
    scaled_run = ray.remote(run).options(num_cpus=config.num_cpus)
    ray.get(scaled_run.remote(config))


def run(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    embeddings_path = config.paths.embeddings_tplt.format(
        emb_method=config.dataset.emb_method, category=config.dataset.category
    )
    embs_df = pd.read_parquet(embeddings_path, filesystem=fs)
    residual = np.stack(embs_df["embedding"].values)  # (N, D)

    # Format read by src.data.ray_data.get_quantized : product_id, L0, L1...
    sids = embs_df[["product_id"]].copy()

    for level, k in enumerate(config.codebook_sizes):
        kmeans = KMeans(n_clusters=k, random_state=config.seed).fit(residual)
        residual = residual - kmeans.cluster_centers_[kmeans.labels_]  # (N, D)
        sids[f"L{level}"] = kmeans.labels_

    # Fraction of items sharing their full SID with another item
    collided = sids.drop(columns="product_id").duplicated(keep=False).mean()
    print(f"{collided:.2%} collided items")

    quant_method = "rqkmeans_" + "-".join(str(k) for k in config.codebook_sizes)
    output_path = config.paths.semantic_ids_tplt.format(
        emb_method=config.dataset.emb_method,
        category=config.dataset.category,
        quant_method=quant_method,
    )
    sids.to_parquet(output_path, filesystem=fs, index=False)
    print("Semantic IDs saved to", output_path)


if __name__ == "__main__":
    main()
