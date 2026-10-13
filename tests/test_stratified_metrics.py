"""Metrics per PASID band of the target : the bands, and their logged means."""

import lightning.pytorch as L
import numpy as np
import torch
from omegaconf import OmegaConf

from src.lightning_module import PASID_BANDS, LitModule
from src.models.sasrec import SampledSoftmax, SASRec
from src.test import pasid_bands


def test_group_metrics_are_means_over_the_cases_of_the_group():
    torch.manual_seed(0)
    net = SASRec(
        n_layers=1,
        d_head=8,
        d_model=16,
        item_encoder=torch.nn.Embedding(30, 16, padding_idx=0),
        dropout=0.0,
        seq_len=5,
        criterion=SampledSoftmax(n_items=20),
    )
    model = LitModule(net, optimizer=None)
    model.full_hydra_config = OmegaConf.create(
        {"data": {"ray_datasets": {"category": "c"}}}
    )
    model.target_bands = torch.tensor([-1] + [b % 6 for b in range(29)])
    # Batches of different sizes and group compositions, every group among the targets
    batches = []
    for start, n in [(1, 7), (8, 3), (11, 12)]:
        target = torch.randint(1, 30, (n, 5))
        target[:, -1] = torch.arange(start, start + n) % 29 + 1  # evaluated target
        batches.append({"query": torch.randint(1, 30, (n, 5)), "target": target})
    loader = torch.utils.data.DataLoader(batches, batch_size=None)
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    logged = trainer.validate(model, loader, verbose=False)[0]

    model.eval()
    with torch.no_grad():
        hits = [
            (model.net.scores(b).topk(10).indices == b["target"][:, -1:]).any(dim=1)
            for b in batches
        ]
    hits, targets = torch.cat(hits), torch.cat([b["target"][:, -1] for b in batches])
    assert np.isclose(logged["valid/c/HR@10"], hits.float().mean().item())
    for g, name in enumerate(PASID_BANDS):
        in_group = model.target_bands[targets] == g
        expected = hits[in_group].float().mean().item()
        assert np.isclose(logged[f"valid/c/band/{name}/HR@10"], expected)
        assert logged[f"valid/c/band/{name}/count"] == in_group.sum().item()


def test_pasid_bands_follow_the_size_of_the_block_of_p():
    # p : 2 head values, a block of 3, a block of 12, cold items in the last block
    p = np.array([0, 1] + [2] * 3 + [3] * 12)
    codes = np.stack([np.zeros_like(p), p], axis=1)  # (N, 2)
    popularity = np.array([50, 40] + [5] * 3 + [1] * 10 + [0, 0])
    bands = pasid_bands(codes, popularity, n_head=2)
    assert bands[:2].tolist() == [1, 1]  # head
    assert (bands[2:5] == 2).all()  # block of 3 : 2-9
    assert (bands[5:15] == 3).all()  # block of 12 : 10-49
    assert (bands[15:] == 0).all()  # cold
