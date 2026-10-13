"""Item encoders of src/models/item_encoders.py."""

import pickle

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.models.item_encoders import SIDMLPItemEncoder, id_encoder, sid_encoder

# Row 0 = padding item, then 3 items. Same code values at different levels
CODES = torch.tensor([[0, 0], [1, 1], [2, 1], [1, 2]])


def test_padding_item_is_a_zero_vector():
    encoder = SIDMLPItemEncoder(CODES, d_model=8, mlp_hidden_dims=[16])
    assert encoder(torch.tensor([0])).abs().max() == 0


def test_levels_use_distinct_rows():
    encoder = SIDMLPItemEncoder(CODES, d_model=8, mlp_hidden_dims=[16])
    # code 1 of level 0 and code 1 of level 1 must not share an embedding
    assert encoder.offset_codes[1, 0] != encoder.offset_codes[1, 1]


def _write_dataset(root):
    """3 products, their SIDs stored in another order than the items pickle."""
    paths = OmegaConf.create(
        {
            "protocol": "file",
            "unique_items_tplt": f"{root}/{{category}}_items.pkl",
            "semantic_ids_tplt": f"{root}/{{emb_method}}_{{category}}_{{quant_method}}.parquet",
        }
    )
    with open(f"{root}/toy_items.pkl", "wb") as f:
        pickle.dump(["a", "b", "c"], f)
    sids = pd.DataFrame(
        {"product_id": ["c", "a", "b"], "L0": [2, 0, 1], "L1": [5, 3, 4]}
    )
    sids.to_parquet(f"{root}/emb_toy_sids.parquet", index=False)
    return paths


def test_sid_encoder_gives_each_item_its_own_codes(tmp_path):
    paths = _write_dataset(tmp_path)
    encoder = sid_encoder(paths, "toy", "emb", "sids", d_model=8, mlp_hidden_widths=[2])
    # item ids 1, 2, 3 = a, b, c ; codes + 1, level 1 after the 3 codes of level 0
    assert encoder.offset_codes.tolist() == [[0, 0], [1, 7], [2, 8], [3, 9]]


def test_id_encoder_has_one_row_per_item_plus_padding(tmp_path):
    paths = _write_dataset(tmp_path)
    assert id_encoder(paths, "toy", d_model=8).num_embeddings == 4
