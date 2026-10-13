"""Items sharing their SID (twins) : SID groups, sampled negatives and evaluation hits."""

import torch

from src.lightning_module import rank_probabilities
from src.models.item_encoders import SIDMLPItemEncoder
from src.models.sasrec import BinaryCrossEntropy, SampledSoftmax, SASRec

# Items 1, 2, 3 are twins, 4 and 5 are twins, 6 is alone (row 0 = padding)
CODES = torch.tensor([[0, 0], [1, 1], [1, 1], [1, 1], [2, 3], [2, 3], [3, 1]])


def _sasrec(item_encoder):
    return SASRec(
        n_layers=1,
        d_head=8,
        d_model=16,
        item_encoder=item_encoder,
        dropout=0.0,
        seq_len=5,
        criterion=SampledSoftmax(n_items=50),
    )


def test_twins_share_a_group_ids_do_not():
    sid = _sasrec(SIDMLPItemEncoder(CODES, 16, [16]))
    ids = _sasrec(torch.nn.Embedding(7, 16, padding_idx=0))
    assert sid.item_groups.tolist() == [0, 1, 1, 1, 2, 2, 3]
    assert ids.item_groups.tolist() == list(range(7))


def test_sampled_negatives_exclude_twins_of_the_target():
    sid = _sasrec(SIDMLPItemEncoder(CODES, 16, [16]))
    torch.manual_seed(0)
    query, target = torch.tensor([[0, 0, 4, 6, 1]]), torch.tensor([[0, 0, 6, 1, 2]])
    _, logits = sid.criterion(sid(query), target, sid)
    torch.manual_seed(0)
    samples = torch.randint(1, 7, (50,))  # the real items, as the criterion
    twin = sid.item_groups[samples] == sid.item_groups[2]  # last target = item 2
    noise_logits = logits[-1, 1:]
    assert (noise_logits[twin] < -1e3).all() and (noise_logits[~twin] > -1e3).all()


# Scores of items 0..6 : item 6 first, then the twins 1, 2, 3 (same score), then 4, 5
SCORES = torch.tensor([[-1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.9]])


def test_the_target_shares_the_ranks_of_its_twins():
    groups = _sasrec(SIDMLPItemEncoder(CODES, 16, [16])).item_groups
    at_rank = rank_probabilities(SCORES, torch.tensor([2]), groups, 5)
    # Item 6 is first, then the target is uniform over the ranks of its 3 twins
    assert torch.allclose(at_rank, torch.tensor([[0, 1 / 3, 1 / 3, 1 / 3, 0]]))


def test_ids_rank_the_exact_target():
    at_rank = rank_probabilities(SCORES, torch.tensor([2]), torch.arange(7), 5)
    # Unique groups, but items 1 and 3 have exactly the target's score : a tie at random
    assert torch.allclose(at_rank, torch.tensor([[0, 1 / 3, 1 / 3, 1 / 3, 0]]))
    scores = torch.tensor([[-1.0, 0.1, 0.5, 0.3, 0.2, 0.2, 0.9]])
    assert rank_probabilities(
        scores, torch.tensor([2]), torch.arange(7), 5
    ).tolist() == [[0, 1, 0, 0, 0]]


def test_a_collapsed_model_is_not_a_hit():
    # Every item gets the same score : the target is uniform over the 6 real items
    scores = torch.tensor([[-1.0] + [0.7] * 6])
    at_rank = rank_probabilities(scores, torch.tensor([2]), torch.arange(7), 10)
    assert torch.allclose(at_rank.sum(), torch.tensor(1.0)) and at_rank[0, 0] == 1 / 6


def test_filtered_twins_and_targets():
    groups = _sasrec(SIDMLPItemEncoder(CODES, 16, [16])).item_groups
    scores = SCORES.clone()
    scores[0, 3] = float("-inf")  # a twin in the history : 2 twins left
    at_rank = rank_probabilities(scores, torch.tensor([2]), groups, 5)
    assert torch.allclose(at_rank, torch.tensor([[0, 0.5, 0.5, 0, 0]]))
    scores[0, 2] = float("-inf")  # the target in the history : never retrieved
    assert rank_probabilities(scores, torch.tensor([2]), groups, 5).sum() == 0


def test_padding_positions_stay_out_of_the_loss():
    sid = _sasrec(SIDMLPItemEncoder(CODES, 16, [16]))
    query, target = torch.tensor([[0, 0, 4, 6, 1]]), torch.tensor([[0, 0, 6, 1, 2]])
    torch.manual_seed(0)
    loss, logits = sid.criterion(sid(query), target, sid)
    assert len(logits) == 3  # the 3 real targets, not the 5 positions
    # Same loss as with the real positions only
    torch.manual_seed(0)
    real_only, _ = sid.criterion(sid(query)[:, 2:], target[:, 2:], sid)
    assert torch.isclose(loss, real_only)


def test_vanilla_scores_are_dot_products_and_bce_keeps_real_positions():
    ids = SASRec(
        n_layers=1,
        d_head=8,
        d_model=16,
        item_encoder=torch.nn.Embedding(7, 16, padding_idx=0),
        dropout=0.0,
        seq_len=5,
        criterion=BinaryCrossEntropy(),
        normalize=False,
    )
    query, target = torch.tensor([[0, 0, 4, 6, 1]]), torch.tensor([[0, 0, 6, 1, 2]])
    loss, logits = ids.criterion(ids(query), target, ids)
    assert logits.shape == (3, 2) and loss > 0  # the 3 real targets, target and noise
    # Item vectors are not normalized : the scores are plain dot products
    ids.eval()
    assert torch.allclose(ids.get_embs(), ids.item_encoder.weight)
    expected = ids(query)[:, -1] @ ids.item_encoder.weight.T  # (1, 7)
    scores = ids.scores({"query": query})
    real = scores > float("-inf")
    assert torch.allclose(scores[real], expected[real], atol=1e-5)


def test_nan_scores_are_misses():
    ids = _sasrec(torch.nn.Embedding(7, 16, padding_idx=0))
    with torch.no_grad():
        ids.item_encoder.weight.fill_(float("nan"))
    from omegaconf import OmegaConf

    from src.lightning_module import LitModule

    model = LitModule(ids, optimizer=None)
    model.full_hydra_config = OmegaConf.create(
        {"data": {"ray_datasets": {"category": "c"}}}
    )
    batch = {
        "query": torch.tensor([[0, 0, 4, 6, 1]]),
        "target": torch.tensor([[0, 0, 6, 1, 2]]),
    }
    logged = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))
    model.eval()
    model._shared_eval_step(batch, "valid")
    assert logged["valid/c/HR@10"] == 0
