import torch
from lightning.pytorch import LightningModule

# Place of the target in PASID (src.test.pasid_bands) : cold, its own value of p (head), or
# the size of the block sharing its value
PASID_BANDS = ["cold", "head", "2-9", "10-49", "50-199", "200+"]


def rank_probabilities(scores, target, groups, n_ranks):
    """(B, V) scores, -inf for the filtered items, (B,) target ids, (V,) SID group of
    every item id -> (B, n_ranks) probability of the target at each rank.

    The target is the item itself. Its twins (same SID) have the same vector, hence the
    same score : the target is ranked uniformly at random among its m twins left in the
    catalogue, and gets 1/m of each of their ranks. A twin ranked first is not a hit.
    Other items with exactly the target's score (e.g. a collapsed model) tie with it too.
    """
    twin = groups[None, :] == groups[target][:, None]  # (B, V) target included
    target_score = scores.gather(1, target[:, None])  # (B, 1)
    above = ((scores > target_score) & ~twin).sum(dim=1)  # (B,) ranked before the twins
    left = scores > float("-inf")  # (B, V) not filtered
    equal = scores == target_score  # (B, V) target included
    tied = ((twin | equal) & left).sum(dim=1)  # (B,) twins and ties, target included
    ranks = torch.arange(n_ranks, device=scores.device)[None, :]  # (1, n_ranks)
    inside = (ranks >= above[:, None]) & (ranks < (above + tied)[:, None])
    probability = inside / tied.clamp(min=1)[:, None]  # (B, n_ranks)
    # A target in the history is filtered out : never retrieved
    return probability * (target_score > float("-inf"))


class LitModule(LightningModule):
    def __init__(self, net, optimizer, scheduler=None):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.Ks = [1, 5, 10, 20]

        self.tplts = {
            "loss": "{split}/{category}/loss",
            "HR": "{split}/{category}/{group}HR@{K}",
            "NDCG": "{split}/{category}/{group}NDCG@{K}",
        }

        self.dcg_denom = torch.log2(torch.arange(1, max(self.Ks) + 1) + 1).view(1, -1)
        # (V,) index in PASID_BANDS of every item id, -1 for none : set by src.test to also
        # log the metrics per band of the target, with the number of cases of each band
        self.target_bands = None

    @property
    def category(self):
        return self.full_hydra_config.data.ray_datasets.category

    def training_step(self, batch, batch_idx):
        loss, _ = self.net.get_loss(batch)

        self.log(
            self.tplts["loss"].format(
                split="train",
                category=self.category,
            ),
            loss,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def _shared_eval_step(self, batch, split):
        # Forward + Loss
        loss, _ = self.net.get_loss(batch)

        # Log the loss - Dataset-wise
        self.log(
            self.tplts["loss"].format(split=split, category=self.category),
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            add_dataloader_idx=False,
        )

        target = batch["target"][:, -1]  # (B,) the next item after the history

        if self.dcg_denom.device != target.device:
            self.dcg_denom = self.dcg_denom.to(target.device)

        # (B, max K) probability of the target at each rank, over the ties of its twins
        # A diverged model scores NaN : ranked last, never a hit (NaN compares as False,
        # so its items would never be counted above the target)
        scores = self.net.scores(batch).nan_to_num(nan=float("-inf"))
        at_rank = rank_probabilities(scores, target, self.net.item_groups, max(self.Ks))

        # (B, max K) per case : expected hit in the top k, expected DCG of the top k
        hit_at = at_rank.cumsum(dim=1)
        dcg_at = (at_rank / self.dcg_denom).cumsum(dim=1)
        # Note : we have a single positive, so IDCG is (2**1 - 1) / log2(2) = 1, we ignore it.

        self._log_ranking(split, "", hit_at, dcg_at)
        if self.target_bands is None:
            return
        bands = self.target_bands.to(target.device)[target]  # (B,)
        for b, name in enumerate(PASID_BANDS):
            mask = bands == b
            if mask.any():
                group = f"band/{name}/"
                self._log_ranking(split, group, hit_at[mask], dcg_at[mask])
                self.log(
                    f"{split}/{self.category}/{group}count",
                    mask.sum().float(),
                    on_step=False,
                    on_epoch=True,
                    reduce_fx="sum",
                    add_dataloader_idx=False,
                )

    def _log_ranking(self, split, group, hit_at, dcg_at):
        """Log HR@K and NDCG@K of (n, Beams) cases. Their mean over the epoch is weighted
        by n : the mean over all the cases, even when a group has few cases in a batch."""
        for K in self.Ks:
            for metric, values in [("HR", hit_at), ("NDCG", dcg_at)]:
                self.log(
                    self.tplts[metric].format(
                        split=split, category=self.category, group=group, K=K
                    ),
                    values[:, K - 1].mean(),
                    batch_size=len(values),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    reduce_fx="mean",
                    add_dataloader_idx=False,
                )

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.net.get_param_groups())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return {"optimizer": optimizer}
