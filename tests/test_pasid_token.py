"""The PASID code p of data_scripts/3_add_pasid_token.py."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def items():
    """2 000 items with 3 levels of 8 codes (many twins) and a long-tailed popularity."""
    rng = np.random.default_rng(0)
    codes = pd.DataFrame(rng.integers(0, 8, (2000, 3)), columns=["L0", "L1", "L2"])
    popularity = rng.zipf(1.5, 2000).astype(float)
    popularity[rng.random(2000) < 0.1] = 0  # cold items
    return codes, popularity


def test_every_sid_is_unique(load_script, items):
    codes, popularity = items
    p = load_script("3_add_pasid_token").pasid_token(codes, popularity, 0.01)
    assert not codes.assign(p=p).duplicated().any()


def test_head_items_have_their_own_p(load_script, items):
    codes, popularity = items
    p = load_script("3_add_pasid_token").pasid_token(codes, popularity, 0.01)
    head = np.argsort(-popularity, kind="stable")[:20]  # 1 % of 2 000
    assert np.isin(p, p[head]).sum() == len(head)  # no other item shares their values


def test_groups_close_at_tau_events(load_script):
    blocks = load_script("3_add_pasid_token").popularity_blocks(
        np.array([6.0, 5, 4, 3, 2, 2, 1, 1, 1, 1, 0, 0]), tau=10
    )
    assert blocks.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


def test_cold_items_join_a_warm_block(load_script):
    blocks = load_script("3_add_pasid_token").popularity_blocks(
        np.array([5.0, 5, 0, 0]), tau=10
    )
    assert blocks.tolist() == [0, 0, 0, 0]  # not a group of cold items only
