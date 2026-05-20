"""Tests for TROGDOR peak-calling helpers."""

import pytest

from chiaroscuro.peaks import call_peaks


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
