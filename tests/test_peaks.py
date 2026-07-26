"""Tests for TROGDOR peak-calling helpers."""

import sys
import types

import pytest

sys.modules.setdefault("pybigtools", types.ModuleType("pybigtools"))
torcheval = types.ModuleType("torcheval")
torcheval_metrics = types.ModuleType("torcheval.metrics")
torcheval_functional = types.ModuleType("torcheval.metrics.functional")
torcheval_functional.binary_auprc = lambda *args, **kwargs: None
torcheval_functional.binary_auroc = lambda *args, **kwargs: None
sys.modules.setdefault("torcheval", torcheval)
sys.modules.setdefault("torcheval.metrics", torcheval_metrics)
sys.modules.setdefault("torcheval.metrics.functional", torcheval_functional)

import chiaroscuro.peaks as peaks_module
from chiaroscuro.peaks import (
    FEATURE_NAMES,
    _pairwise_features,
    _predict_split,
    call_peaks,
    call_profile_peaks,
    resolve_seed_score,
)

FEATURE_NAMES_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def test_adjacent_bins_merge():
    peaks = call_peaks([(0, 16, 0.9), (16, 32, 0.8)], min_score=0.7)
    assert len(peaks) == 1
    assert peaks[0]["start"] == 0
    assert peaks[0]["end"] == 32


def test_short_gap_merges_only_when_permitted():
    intervals = [(0, 16, 0.9), (32, 48, 0.8)]
    assert len(call_peaks(intervals, min_score=0.7, max_gap=0)) == 2
    merged = call_peaks(intervals, min_score=0.7, max_gap=16)
    assert len(merged) == 1
    assert merged[0]["start"] == 0
    assert merged[0]["end"] == 48


def test_min_width_filters_short_peaks():
    peaks = call_peaks(
        [(0, 16, 0.9), (32, 64, 0.95)],
        min_score=0.7,
        min_width=32,
    )
    assert len(peaks) == 1
    assert peaks[0]["start"] == 32
    assert peaks[0]["end"] == 64


def test_peak_score_is_max_score_after_merging():
    peaks = call_peaks(
        [(0, 16, 0.8), (16, 32, 0.97), (32, 48, 0.9)],
        min_score=0.7,
    )
    assert peaks[0]["score"] == pytest.approx(0.97)


def test_summit_is_max_score_bin():
    peaks = call_peaks(
        [(0, 16, 0.8), (16, 32, 0.97), (32, 48, 0.9)],
        min_score=0.7,
    )
    assert peaks[0]["summit_start"] == 16
    assert peaks[0]["summit_end"] == 32
    assert peaks[0]["summit_score"] == pytest.approx(0.97)


def test_invalid_refined_parameters_raise():
    with pytest.raises(ValueError, match="max_gap"):
        call_peaks([], max_gap=-1)
    with pytest.raises(ValueError, match="min_width"):
        call_peaks([], min_width=-1)


def test_profile_single_summit_keeps_seeded_shoulders():
    intervals = [
        (0, 10, 0.4),
        (10, 20, 0.7),
        (20, 30, 0.96),
        (30, 40, 0.7),
        (40, 50, 0.4),
    ]

    peaks = call_profile_peaks(intervals, min_score=0.95, seed_score=0.5)

    assert len(peaks) == 1
    assert peaks[0]["start"] == 10
    assert peaks[0]["end"] == 40
    assert peaks[0]["summit_start"] == 20
    assert peaks[0]["summit_score"] == pytest.approx(0.96)


def test_profile_splits_two_summits_with_deep_valley():
    intervals = [
        (0, 10, 0.95),
        (10, 20, 0.55),
        (20, 30, 0.2),
        (30, 40, 0.6),
        (40, 50, 0.97),
    ]

    peaks = call_profile_peaks(
        intervals,
        min_score=0.9,
        seed_score=0.1,
        valley_fraction=0.5,
    )

    assert len(peaks) == 2
    assert (peaks[0]["start"], peaks[0]["end"]) == (0, 20)
    assert (peaks[1]["start"], peaks[1]["end"]) == (30, 50)


def test_profile_keeps_shallow_valley_merged():
    intervals = [
        (0, 10, 0.95),
        (10, 20, 0.8),
        (20, 30, 0.97),
    ]

    peaks = call_profile_peaks(
        intervals,
        min_score=0.9,
        seed_score=0.5,
        valley_fraction=0.5,
    )

    assert len(peaks) == 1
    assert peaks[0]["start"] == 0
    assert peaks[0]["end"] == 30


def test_profile_boundary_fraction_trims_low_shoulders():
    intervals = [
        (0, 10, 0.5),
        (10, 20, 0.8),
        (20, 30, 1.0),
        (30, 40, 0.7),
        (40, 50, 0.4),
    ]

    peaks = call_profile_peaks(
        intervals,
        min_score=0.95,
        seed_score=0.4,
        boundary_fraction=0.5,
    )

    assert len(peaks) == 1
    assert peaks[0]["start"] == 10
    assert peaks[0]["end"] == 40


def test_profile_boundary_fraction_zero_is_true_noop():
    # Regression test: the trim threshold used to be summit_score *
    # boundary_fraction, anchored at 0 instead of seed_score. Every bin in a
    # segment already clears seed_score during seeding, so that made
    # boundary_fraction values well above 0 a silent no-op too (e.g. 0.5 with
    # seed_score=0.4 and summit_score=1.0 computed threshold=0.5, keeping
    # everything anyway). Anchoring at seed_score makes boundary_fraction=0
    # the *only* no-op, and the rest of the range does real trimming.
    intervals = [
        (0, 10, 0.5),
        (10, 20, 0.8),
        (20, 30, 1.0),
        (30, 40, 0.7),
        (40, 50, 0.4),
    ]

    peaks = call_profile_peaks(
        intervals,
        min_score=0.95,
        seed_score=0.4,
        boundary_fraction=0.0,
    )

    assert len(peaks) == 1
    assert peaks[0]["start"] == 0
    assert peaks[0]["end"] == 50


def test_profile_boundary_fraction_one_keeps_only_summit():
    intervals = [
        (0, 10, 0.5),
        (10, 20, 0.8),
        (20, 30, 1.0),
        (30, 40, 0.7),
        (40, 50, 0.4),
    ]

    peaks = call_profile_peaks(
        intervals,
        min_score=0.95,
        seed_score=0.4,
        boundary_fraction=1.0,
    )

    assert len(peaks) == 1
    assert peaks[0]["start"] == 20
    assert peaks[0]["end"] == 30


def test_profile_width_filters():
    intervals = [
        (0, 10, 0.95),
        (10, 20, 0.96),
        (40, 50, 0.97),
    ]

    min_filtered = call_profile_peaks(intervals, min_score=0.9, min_width=15)
    max_filtered = call_profile_peaks(
        intervals, min_score=0.9, seed_score=0.9, max_width=15
    )

    assert len(min_filtered) == 1
    assert (min_filtered[0]["start"], min_filtered[0]["end"]) == (0, 20)
    assert len(max_filtered) == 1
    assert (max_filtered[0]["start"], max_filtered[0]["end"]) == (40, 50)


def test_resolve_seed_score_defaults_to_half_min_score():
    assert resolve_seed_score(0.95, None) == pytest.approx(0.5)
    assert resolve_seed_score(0.3, None) == pytest.approx(0.3)
    assert resolve_seed_score(0.95, 0.8) == pytest.approx(0.8)


def test_profile_default_seed_score_splits_without_explicit_seed():
    # No seed_score passed: previously this fell back to seeding at
    # min_score (0.9), which would have excluded the 0.55/0.6 shoulder bins
    # entirely and prevented the valley split below from ever engaging.
    intervals = [
        (0, 10, 0.95),
        (10, 20, 0.55),
        (20, 30, 0.2),
        (30, 40, 0.6),
        (40, 50, 0.97),
    ]

    peaks = call_profile_peaks(intervals, min_score=0.9, valley_fraction=0.5)

    assert len(peaks) == 2
    assert (peaks[0]["start"], peaks[0]["end"]) == (0, 20)
    assert (peaks[1]["start"], peaks[1]["end"]) == (30, 50)


def test_invalid_profile_parameters_raise():
    with pytest.raises(ValueError, match="seed_score"):
        call_profile_peaks([], min_score=0.5, seed_score=0.6)
    with pytest.raises(ValueError, match="valley_fraction"):
        call_profile_peaks([], valley_fraction=1.1)
    with pytest.raises(ValueError, match="boundary_fraction"):
        call_profile_peaks([], boundary_fraction=-0.1)
    with pytest.raises(ValueError, match="smooth_bins"):
        call_profile_peaks([], smooth_bins=0)
    with pytest.raises(ValueError, match="split_merge_rule"):
        call_profile_peaks([], split_merge_rule="bogus")
    with pytest.raises(ValueError, match="split_merge_cutoff"):
        call_profile_peaks([], split_merge_cutoff=1.1)


class TestPairwiseFeatures:
    """Tests for _pairwise_features()'s dREG-equivalent feature extraction."""

    def test_basic_asymmetric_pair(self):
        block = [(0, 10, 0.5), (10, 20, 0.9), (20, 30, 0.3), (30, 40, 0.2), (40, 50, 0.8)]
        scores = [0.5, 0.9, 0.3, 0.2, 0.8]

        features = _pairwise_features(block, scores, left=1, valley_i=3, right=4)

        dist, r1, r2, y1, y2, maxy, d1, d2, d3, dr = features
        assert dist == pytest.approx(30.0)
        assert r1 == pytest.approx(20.0)
        assert r2 == pytest.approx(10.0)
        assert y1 == pytest.approx(0.9)
        assert y2 == pytest.approx(0.8)
        assert maxy == pytest.approx(0.9)
        assert d1 == pytest.approx(0.1)
        assert d2 == pytest.approx(0.6)
        assert d3 == pytest.approx(0.2)
        assert dr == pytest.approx(2.0)

    def test_dr_divide_by_zero_guard(self):
        # y1 == y2 (d1=0) and valley_score == 0 (d3=0) -> denom == 0.
        block = [(0, 10, 0.0), (10, 20, 0.0), (20, 30, 0.0)]
        scores = [0.5, 0.0, 0.5]

        features = _pairwise_features(block, scores, left=0, valley_i=1, right=2)

        dr = features[FEATURE_NAMES_INDEX["dr"]]
        assert dr == pytest.approx(0.0)

    def test_d2_clipped_when_valley_exceeds_weaker_summit(self):
        # Valley score (0.5) higher than the weaker summit (0.3) -- a
        # degenerate/adversarial case a smoothing artifact could produce.
        # Without clipping, d2 would be negative (0.3 - 0.5 = -0.2).
        block = [(0, 10, 0.0), (10, 20, 0.0), (20, 30, 0.0)]
        scores = [0.3, 0.5, 0.9]

        features = _pairwise_features(block, scores, left=0, valley_i=1, right=2)

        d2 = features[FEATURE_NAMES_INDEX["d2"]]
        assert d2 == pytest.approx(0.0)
        assert d2 >= 0.0


class TestPredictSplit:
    """Tests for _predict_split()'s sigmoid(w . x + b) inference, using
    hand-picked weights rather than the real shipped fit (which is expected
    to change as the fitting script/data evolve)."""

    def test_zero_weights_and_bias_gives_half(self):
        weights = (0.0,) * 10
        features = tuple(range(10))
        assert _predict_split(features, weights=weights, bias=0.0) == pytest.approx(0.5)

    def test_single_feature_dominates(self):
        # Only the d2 (valley-depth) feature has nonzero weight.
        weights = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0)
        shallow = (0,) * 7 + (0.001,) + (0, 0)
        deep = (0,) * 7 + (0.5,) + (0, 0)
        assert _predict_split(shallow, weights=weights, bias=-1.0) < 0.5
        assert _predict_split(deep, weights=weights, bias=-1.0) > 0.99


class TestProfileLearnedSplitMergeRule:
    """Tests for call_profile_peaks(split_merge_rule="learned"). Mocks
    _predict_split itself (rather than the shipped weights, which are bound
    into _predict_split's default arguments at import time and would not be
    affected by monkeypatching the module-level constants) so these tests
    don't churn whenever the model is refit."""

    def test_learned_rule_splits_when_predictor_says_split(self, monkeypatch):
        monkeypatch.setattr(peaks_module, "_predict_split", lambda features: 1.0)
        intervals = [(0, 10, 0.95), (10, 20, 0.8), (20, 30, 0.97)]

        peaks = call_profile_peaks(
            intervals, min_score=0.9, seed_score=0.5, split_merge_rule="learned"
        )

        assert len(peaks) == 2

    def test_learned_rule_merges_when_predictor_says_merge(self, monkeypatch):
        monkeypatch.setattr(peaks_module, "_predict_split", lambda features: 0.0)
        intervals = [(0, 10, 0.95), (10, 20, 0.2), (20, 30, 0.97)]

        peaks = call_profile_peaks(
            intervals, min_score=0.9, seed_score=0.1, split_merge_rule="learned"
        )

        assert len(peaks) == 1

    def test_learned_rule_respects_cutoff(self, monkeypatch):
        monkeypatch.setattr(peaks_module, "_predict_split", lambda features: 0.4)
        intervals = [(0, 10, 0.95), (10, 20, 0.8), (20, 30, 0.97)]

        merged = call_profile_peaks(
            intervals,
            min_score=0.9,
            seed_score=0.5,
            split_merge_rule="learned",
            split_merge_cutoff=0.5,
        )
        split = call_profile_peaks(
            intervals,
            min_score=0.9,
            seed_score=0.5,
            split_merge_rule="learned",
            split_merge_cutoff=0.3,
        )

        assert len(merged) == 1
        assert len(split) == 2

    def test_threshold_rule_default_unaffected_by_learned_mock(self, monkeypatch):
        # The default ("threshold") path must never call _predict_split, so
        # mocking it to always predict "split" must not change its output.
        monkeypatch.setattr(peaks_module, "_predict_split", lambda features: 1.0)
        intervals = [(0, 10, 0.95), (10, 20, 0.8), (20, 30, 0.97)]

        peaks = call_profile_peaks(
            intervals, min_score=0.9, seed_score=0.5, valley_fraction=0.5
        )

        assert len(peaks) == 1
