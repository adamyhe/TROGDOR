"""Tests for chiaroscuro.metrics (binary AUPRC / AUROC)."""

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from chiaroscuro.metrics import binary_auprc, binary_auroc


def _random_case(seed, n=5000, n_levels=None):
    rng = np.random.default_rng(seed)
    target = (rng.random(n) < 0.1).astype(np.int64)
    scores = rng.random(n) + 0.3 * target
    if n_levels is not None:
        # Quantise to force many tied scores
        scores = np.round(scores * n_levels) / n_levels
    return torch.tensor(scores, dtype=torch.float32), torch.tensor(target)


@pytest.mark.parametrize("n_levels", [None, 20, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn(seed, n_levels):
    scores, target = _random_case(seed, n_levels=n_levels)
    expected_ap = average_precision_score(target.numpy(), scores.numpy())
    expected_auc = roc_auc_score(target.numpy(), scores.numpy())
    assert binary_auprc(scores, target).item() == pytest.approx(expected_ap, rel=1e-9)
    assert binary_auroc(scores, target).item() == pytest.approx(expected_auc, rel=1e-9)


def test_perfect_and_inverted_ranking():
    target = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert binary_auprc(scores, target).item() == pytest.approx(1.0)
    assert binary_auroc(scores, target).item() == pytest.approx(1.0)
    assert binary_auroc(-scores, target).item() == pytest.approx(0.0)


def test_all_tied_scores():
    target = torch.tensor([0, 1, 0, 1, 0])
    scores = torch.full((5,), 0.5)
    assert binary_auprc(scores, target).item() == pytest.approx(0.4)
    assert binary_auroc(scores, target).item() == pytest.approx(0.5)


def test_degenerate_targets():
    scores = torch.tensor([0.1, 0.5, 0.9])
    assert math.isnan(binary_auprc(scores, torch.zeros(3)).item())
    assert binary_auprc(scores, torch.ones(3)).item() == pytest.approx(1.0)
    assert binary_auroc(scores, torch.zeros(3)).item() == 0.5
    assert binary_auroc(scores, torch.ones(3)).item() == 0.5


def test_accepts_multidimensional_and_float_targets():
    scores, target = _random_case(3, n=600)
    flat = binary_auprc(scores, target).item()
    shaped = binary_auprc(scores.reshape(2, 1, 300), target.float().reshape(2, 1, 300))
    assert shaped.item() == pytest.approx(flat)


def test_mismatched_sizes_raise():
    with pytest.raises(ValueError):
        binary_auprc(torch.rand(4), torch.tensor([0, 1, 1]))
