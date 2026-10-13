"""PASID : append a popularity-aware code p to the Semantic IDs of each item.

N items, Q content levels.
Items are ranked by train popularity n. The head_fraction most popular items get a
value of p of their own ; tau is the popularity of the last head item. Below, items are
taken in popularity order and grouped : a group closes as soon as its items add up to tau
train events, and all the items of a group share a value of p. Cold items join the last
group, so that no value is carried by cold items only : it would never be a training
target. Groups are cut separately for each rank of TIGER's tie-break (the rank of the
item among the items with the same codes), so two items with the same codes never share
p : every SID is unique.
"""

import fsspec
import hydra
import numpy as np
import pandas as pd

from src.utils.tools import patch_fsspec


def popularity_blocks(popularity, tau):
    """(n,) popularity, most popular first -> (n,) group index. A group closes as soon as
    its train events reach tau."""
    block = np.zeros(len(popularity), dtype=int)  # (n,)
    current, events = 0, 0
    for i, n in enumerate(popularity):
        block[i] = current
        events += n
        if n > 0 and events >= tau:
            current, events = current + 1, 0
    # The last warm item can close a group : the cold items then join it
    cold = popularity == 0
    if cold.any() and not cold.all():
        block[cold] = block[~cold][-1]
    return block


def pasid_token(codes, popularity, head_fraction):
    """(N, Q) codes DataFrame, (N,) popularity -> (N,) code p."""
    # TIGER's tie-break : rank of the item among the items sharing its codes
    ranks = codes.groupby(list(codes.columns)).cumcount().to_numpy()  # (N,)
    order = np.argsort(-popularity, kind="stable")  # (N,) most popular first
    n_head = int(np.ceil(head_fraction * len(order)))
    tau = popularity[order[n_head - 1]]  # popularity of the last head item

    p = np.empty(len(order), dtype=int)  # (N,)
    p[order[:n_head]] = np.arange(n_head)  # a value of their own
    next_value = n_head  # each rank uses its own range of values
    rest = order[n_head:]  # (N - n_head,) most popular first
    for rank in np.unique(ranks[rest]):
        items = rest[ranks[rest] == rank]
        p[items] = next_value + popularity_blocks(popularity[items], tau)
        next_value = p[items].max() + 1
    return p


@hydra.main(
    config_path="../configs", config_name="data/3_add_pasid_token", version_base="1.2"
)
def main(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    def sids_path(quant_method):
        return config.paths.semantic_ids_tplt.format(
            emb_method=config.dataset.emb_method,
            category=config.dataset.category,
            quant_method=quant_method,
        )

    sids = pd.read_parquet(sids_path(config.source), filesystem=fs)
    codes = sids.drop(columns="product_id")  # (N, Q)

    # Popularity = number of occurrences of the item in the train timelines
    train_path = config.paths.timelines_tplt.format(
        category=config.dataset.category, split="train"
    )
    train = pd.read_parquet(train_path, filesystem=fs, columns=["timeline"])
    counts = train["timeline"].explode().value_counts()
    popularity = sids["product_id"].map(counts).fillna(0).to_numpy()  # (N,)

    p = pasid_token(codes, popularity, config.head_fraction)
    sids[f"L{codes.shape[1]}"] = p
    print(f"{len(np.unique(p))} values of p")

    output_path = sids_path(f"{config.source}_{config.name}")
    sids.to_parquet(output_path, filesystem=fs, index=False)
    print("Semantic IDs saved to", output_path)


if __name__ == "__main__":
    main()
