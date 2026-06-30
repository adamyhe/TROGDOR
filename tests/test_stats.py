"""Tests for empirical calibration helpers."""

import sys
import types
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

sys.modules.setdefault("pybigtools", types.ModuleType("pybigtools"))
hf = types.ModuleType("huggingface_hub")
hf.hf_hub_download = lambda *args, **kwargs: None
hf.try_to_load_from_cache = lambda *args, **kwargs: None
sys.modules.setdefault("huggingface_hub", hf)
torcheval = types.ModuleType("torcheval")
torcheval_metrics = types.ModuleType("torcheval.metrics")
torcheval_functional = types.ModuleType("torcheval.metrics.functional")
torcheval_functional.binary_auprc = lambda *args, **kwargs: None
torcheval_functional.binary_auroc = lambda *args, **kwargs: None
sys.modules.setdefault("torcheval", torcheval)
sys.modules.setdefault("torcheval.metrics", torcheval_metrics)
sys.modules.setdefault("torcheval.metrics.functional", torcheval_functional)

from chiaroscuro.stats import (
    score_peaks_from_array,
    select_fdr_threshold,
    shuffle_peaks_within_intervals,
)


def _load_commands_module():
    path = Path(__file__).resolve().parents[1] / "cli" / "commands.py"
    spec = importlib.util.spec_from_file_location("trogdor_cli_commands", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_commands = _load_commands_module()
_call_chrom_peaks = _commands._call_chrom_peaks
_default_raw_peak_output = _commands._default_raw_peak_output
_score_peak_records_from_array = _commands._score_peak_records_from_array


def test_score_peaks_from_array_uses_output_stride_bins():
    scores = np.array([0.1, 0.8, 0.4, 0.2], dtype=np.float32)
    peaks = pd.DataFrame(
        [("chr1", 16, 48), ("chr1", 0, 16)],
        columns=["chrom", "start", "end"],
    )

    out = score_peaks_from_array(scores, peaks, "chr1", output_stride=16, stat="max")

    assert out[0] == pytest.approx(0.8)
    assert out[1] == pytest.approx(0.1)


def test_shuffle_peaks_within_intervals_keeps_width_and_scope():
    peaks = pd.DataFrame(
        [("chr1", 0, 10), ("chr1", 10, 20)],
        columns=["chrom", "start", "end"],
    )
    allowed = pd.DataFrame(
        [("chr1", 100, 110), ("chr1", 200, 220)],
        columns=["chrom", "start", "end"],
    )
    rng = np.random.default_rng(0)

    shuffled = shuffle_peaks_within_intervals(peaks, allowed, ["chr1"], rng)

    assert len(shuffled) == 2
    assert set(shuffled["end"] - shuffled["start"]) == {10}
    for _, row in shuffled.iterrows():
        in_first = 100 <= row["start"] and row["end"] <= 110
        in_second = 200 <= row["start"] and row["end"] <= 220
        assert in_first or in_second


def test_select_fdr_threshold_returns_lowest_passing_threshold():
    thresholds = np.array([0.1, 0.2, 0.3, 0.4])
    n_real = np.array([10, 8, 6, 3])
    fdr = np.array([0.3, 0.12, 0.04, 0.01])

    threshold, n_peaks = select_fdr_threshold(thresholds, n_real, fdr, 0.05)

    assert threshold == pytest.approx(0.3)
    assert n_peaks == 6


def test_default_raw_peak_output_uses_bed_sibling():
    assert (
        _default_raw_peak_output("sample.profile.fdr05.bed.gz")
        == "sample.profile.fdr05.raw.bed.gz"
    )
    assert _default_raw_peak_output("sample.bed") == "sample.raw.bed"


def test_simple_peak_records_include_summit_for_calibration():
    params = {
        "threshold": 0.5,
        "mode": "simple",
        "max_gap": 0,
        "min_width": 0,
        "max_width": None,
        "seed_score": None,
        "smooth_bins": 1,
        "valley_fraction": 0.5,
        "boundary_fraction": 0.0,
        "min_support_signal": 0.0,
    }

    peaks = _call_chrom_peaks(
        "chr1",
        [(0, 16, 0.6), (16, 32, 0.9), (48, 64, 0.8)],
        params,
    )

    assert peaks[0]["start"] == 0
    assert peaks[0]["end"] == 32
    assert peaks[0]["summit_start"] == 16
    assert peaks[0]["summit_score"] == pytest.approx(0.9)


def test_summit_calibration_score_uses_recorded_summit_score():
    records = [
        {
            "chrom": "chr1",
            "start": 0,
            "end": 48,
            "score": 0.9,
            "summit_start": 16,
            "summit_end": 32,
            "summit_score": 0.9,
        }
    ]
    scores = np.array([0.2, 0.9, 0.3], dtype=np.float32)

    out = _score_peak_records_from_array(
        records,
        scores,
        "chr1",
        output_stride=16,
        stat="summit",
    )

    assert out[0] == pytest.approx(0.9)
