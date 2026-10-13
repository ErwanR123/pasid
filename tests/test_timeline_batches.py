"""Training batches drawn on the GPU (src.data.datamodule.TimelineBatches) : the same
examples as src.data.sasrec.SASRecPrePro on random users."""

import numpy as np
import torch

from src.data.datamodule import TimelineBatches
from src.models import SpecialTokens

PAD = SpecialTokens.PAD.value
# Item ids from 10 on. User 1 has ties : items 20, 21, 22 share a timestamp
TIMELINES = [np.array([10, 11, 12]), np.arange(20, 27), np.arange(30, 42)]
TIMESTAMPS = [np.array([1, 2, 3]), np.array([1, 2, 5, 5, 5, 6, 7]), np.arange(12)]


def _draws(n_batches=200, crop_length=5, augment=True):
    torch.manual_seed(0)
    batches = TimelineBatches(TIMELINES, TIMESTAMPS, 64, crop_length, "cpu", augment)
    return [next(batches) for _ in range(n_batches)]


def _windows(batch):
    """Batch -> the window of each example : its query items, then the last target."""
    windows = []
    for query, target in zip(batch["query"].tolist(), batch["target"].tolist()):
        windows.append([i for i in query if i != PAD] + [target[-1]])
    return windows


def test_query_and_target_are_the_window_shifted_by_one():
    for batch in _draws(10):
        query, target = batch["query"], batch["target"]
        assert query.shape == target.shape == (64, 5)
        # The target of a position is the query of the next one, where it is an item
        real = target[:, :-1] != PAD
        assert (target[:, :-1] == query[:, 1:])[real].all()
        # As many targets as queries : the window minus one item
        assert ((query != PAD).sum(1) == (target != PAD).sum(1)).all()
        # Left padding : the first column of the query is always padding
        assert (query[:, 0] == PAD).all()


def test_windows_are_crops_of_one_timeline():
    for batch in _draws(20):
        for window in _windows(batch):
            user = [u for u, tl in enumerate(TIMELINES) if window[0] in tl][0]
            timeline, times = TIMELINES[user], TIMESTAMPS[user]
            assert len(window) == min(len(timeline), 5)
            # Same items as a contiguous slice, up to the order of equal timestamps
            positions = sorted(np.flatnonzero(np.isin(timeline, window)))
            assert positions == list(range(positions[0], positions[0] + len(window)))
            window_times = [times[list(timeline).index(i)] for i in window]
            assert window_times == sorted(window_times)


def test_crops_cover_every_start_and_ties_get_shuffled():
    windows = [tuple(w) for b in _draws() for w in _windows(b)]
    # User 2 (12 items, crop 5) : the 8 possible starts all appear
    assert {w[0] for w in windows if w[0] >= 30} == set(range(30, 38))
    # User 1 : the 3 items of timestamp 5 appear in several orders
    orders = {tuple(i for i in w if i in (22, 23, 24)) for w in windows if 20 in w}
    assert len({o for o in orders if len(o) == 3}) > 1
    # Users are drawn uniformly : user 0 (3 items, never cropped) about a third
    share = np.mean([w[0] < 20 for w in windows])
    assert abs(share - 1 / 3) < 0.03


def test_without_augment_windows_are_the_last_items_in_order():
    for batch in _draws(10, augment=False):
        for window in _windows(batch):
            user = [u for u, tl in enumerate(TIMELINES) if window[0] in tl][0]
            assert window == TIMELINES[user][-5:].tolist()
