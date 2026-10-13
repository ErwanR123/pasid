from typing import Any, Optional

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from ray.train import get_dataset_shard
from ray.train.torch import get_device
from torch.utils.data import DataLoader

from src.models import SpecialTokens


class TimelineBatches:
    """Endless training batches drawn on the GPU, as src.data.sasrec.SASRecPrePro does
    on CPU for one random user : a random window of at most crop_length items, the items
    of the same timestamp shuffled, left padding. Without augment, as the original
    SASRec : the last crop_length items, in their order.

    U users, T train events in total, B batch size, L crop length.
    """

    def __init__(self, timelines, timestamps, batch_size, crop_length, device, augment):
        """timelines, timestamps : U arrays of item ids and of their timestamps, in
        chronological order."""
        lengths = torch.tensor([len(t) for t in timelines], device=device)  # (U,)
        self.lengths = lengths
        self.starts = torch.cumsum(lengths, 0) - lengths  # (U,) first event of a user
        self.items = torch.from_numpy(np.concatenate(timelines)).to(device)  # (T,)
        self.times = torch.from_numpy(np.concatenate(timestamps)).to(device)  # (T,)
        self.batch_size, self.crop_length, self.device = batch_size, crop_length, device
        self.augment = augment

    def __iter__(self):
        return self

    def __next__(self):
        B, L, device = self.batch_size, self.crop_length, self.device
        users = torch.randint(
            len(self.lengths), (B,), device=device
        )  # with replacement
        length = self.lengths[users]  # (B,)
        window = length.clamp(max=L)  # (B,) items kept
        # Window start uniform in [0, length - window], as random.randint(0, excess),
        # or the last window
        if self.augment:
            shift = (
                torch.rand(B, device=device) * (length - window + 1)
            ).long()  # (B,)
        else:
            shift = length - window  # (B,)
        # Right-aligned window : column j holds the item j - (L - window) of the window
        rank = torch.arange(L, device=device)[None, :] - (L - window)[:, None]  # (B, L)
        padding = rank < 0
        events = (self.starts[users] + shift)[:, None] + rank.clamp(min=0)  # (B, L)
        items = self.items[events].masked_fill(padding, SpecialTokens.PAD.value)
        times = self.times[events].masked_fill(padding, torch.iinfo(torch.int64).min)

        # Shuffle the items of the same timestamp : sort by a random key, then by time
        # with a stable sort, which keeps the random order inside each timestamp
        if self.augment:
            order = torch.rand(B, L, device=device).argsort(dim=1)  # (B, L)
            order = order.gather(1, times.gather(1, order).argsort(dim=1, stable=True))
            items = items.gather(1, order)

        # Query : the window but its last item, target : but its first, both left padded
        pad = torch.full((B, 1), SpecialTokens.PAD.value, device=device)
        query = torch.cat([pad, items[:, :-1]], dim=1)  # (B, L)
        target = items.masked_fill(rank <= 0, SpecialTokens.PAD.value)  # (B, L)
        return {"query": query, "target": target}


class RayDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        crop_length: int,
        augment: bool = True,  # random windows and shuffled ties (TimelineBatches)
        ray_datasets=None,  # Used to load the datasets without a TorchTrainer
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.ray_datasets = ray_datasets

    def get_dataset(self, name: str):
        if self.ray_datasets is not None:
            return self.ray_datasets[name]
        else:
            return get_dataset_shard(name)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # The train timelines, read once, live on the GPU
            timelines, timestamps = [], []
            for batch in self.get_dataset("train").iter_batches(
                batch_size=100_000, batch_format="numpy"
            ):
                timelines.extend(batch["timeline"])
                timestamps.extend(batch["timestamp"])
            self.data_train = TimelineBatches(
                timelines,
                timestamps,
                self.hparams.train_batch_size,
                self.hparams.crop_length,
                get_device(),
                self.hparams.augment,
            )
        if stage == "test" or stage is None:
            self.data_test = self.get_dataset("test")
        self.data_valid = self.get_dataset("valid")

    def train_dataloader(self) -> Any:
        return self.data_train

    def val_dataloader(self) -> DataLoader[Any]:
        return self.data_valid.iter_torch_batches(
            batch_size=self.hparams.valid_batch_size,
            drop_last=False,
            prefetch_batches=3,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return self.data_test.iter_torch_batches(
            batch_size=self.hparams.valid_batch_size,
            drop_last=False,
            prefetch_batches=3,
        )
