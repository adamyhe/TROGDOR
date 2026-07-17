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
    compute_fdr,
    score_peaks_from_array,
    select_fdr_threshold,
    shuffle_peaks_within_intervals,
    subtract_intervals_df,
)
from chiaroscuro.calibration import (
    null_log_records,
    score_peak_records_from_array,
    write_calibration_figure,
    write_null_log,
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


def test_quantile_fdr_grid_resolves_saturated_tail():
    real_scores = np.concatenate(
        [
            np.linspace(0.25, 0.995, 500),
            np.array([0.9991, 0.9992, 0.9993, 0.9994, 0.9995]),
        ]
    )
    null_scores = np.concatenate(
        [
            np.linspace(0.25, 0.995, 500),
            np.array([0.9991, 0.9992]),
        ]
    )

    linear = compute_fdr(
        real_scores,
        null_scores,
        n_shuffle=1,
        n_thresholds=5,
        threshold_grid="linear",
    )
    quantile = compute_fdr(
        real_scores,
        null_scores,
        n_shuffle=1,
        n_thresholds=100,
        threshold_grid="quantile",
    )

    linear_threshold, linear_n = select_fdr_threshold(
        linear[0], linear[1], linear[3], 0.5
    )
    quantile_threshold, quantile_n = select_fdr_threshold(
        quantile[0], quantile[1], quantile[3], 0.5
    )

    assert linear_threshold == pytest.approx(real_scores.max())
    assert linear_n == 1
    assert quantile_threshold < real_scores.max()
    assert quantile_n > 1


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

    out = score_peak_records_from_array(
        records,
        scores,
        "chr1",
        output_stride=16,
        stat="summit",
    )

    assert out[0] == pytest.approx(0.9)


def test_smoothed_summit_calibration_averages_local_window():
    records = [
        {
            "chrom": "chr1",
            "start": 0,
            "end": 80,
            "score": 0.9,
            "summit_start": 32,
            "summit_end": 48,
            "summit_score": 0.9,
        }
    ]
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.4], dtype=np.float32)

    out = score_peak_records_from_array(
        records,
        scores,
        "chr1",
        output_stride=16,
        stat="smoothed_summit",
        smooth_bins=3,
    )

    assert out[0] == pytest.approx((0.2 + 0.9 + 0.3) / 3)


def test_write_calibration_figure(tmp_path):
    pytest.importorskip("matplotlib")

    path = tmp_path / "calibration.png"
    real_scores = np.array([0.9, 0.8, 0.4], dtype=np.float32)
    null_scores = np.array([0.7, 0.3, 0.2], dtype=np.float32)
    thresholds = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    n_real = np.array([3, 2, 2], dtype=float)
    fdr = np.array([1.0, 0.25, 0.0], dtype=float)

    write_calibration_figure(
        path,
        real_scores,
        null_scores,
        thresholds,
        n_real,
        fdr,
        "summit",
        0.05,
        0.8,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_null_log_records_pairs_positions_with_scores_and_drops_nan():
    null_df = pd.DataFrame(
        [("chr1", 0, 16), ("chr1", 32, 48), ("chr1", 64, 80)],
        columns=["chrom", "start", "end"],
    )
    scores = np.array([0.9, np.nan, 0.4], dtype=np.float32)

    records = null_log_records(null_df, scores)

    assert records == [("chr1", 0, 16, pytest.approx(0.9)), ("chr1", 64, 80, pytest.approx(0.4))]


def test_write_null_log_round_trips(tmp_path):
    path = tmp_path / "null_log.tsv"
    records = [("chr1", 0, 16, 0.9), ("chr2", 100, 116, 0.4)]

    write_null_log(path, records)

    table = pd.read_csv(path, sep="\t")
    assert list(table.columns) == ["chrom", "start", "end", "score"]
    assert len(table) == 2
    assert table.iloc[0]["chrom"] == "chr1"
    assert table.iloc[1]["score"] == pytest.approx(0.4)


def _bed(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def test_subtract_intervals_df_no_overlap_is_unchanged():
    allowed = _bed([("chr1", 0, 10)])
    exclude = _bed([("chr1", 50, 60)])

    out = subtract_intervals_df(allowed, exclude)

    assert list(zip(out["chrom"], out["start"], out["end"])) == [("chr1", 0, 10)]


def test_subtract_intervals_df_full_swallow_removes_interval():
    allowed = _bed([("chr1", 10, 20)])
    exclude = _bed([("chr1", 0, 100)])

    out = subtract_intervals_df(allowed, exclude)

    assert len(out) == 0


def test_subtract_intervals_df_splits_middle_overlap():
    allowed = _bed([("chr1", 0, 100)])
    exclude = _bed([("chr1", 40, 60)])

    out = subtract_intervals_df(allowed, exclude)

    assert list(zip(out["start"], out["end"])) == [(0, 40), (60, 100)]


def test_subtract_intervals_df_truncates_partial_overlap():
    allowed = _bed([("chr1", 0, 100)])
    exclude = _bed([("chr1", 80, 150)])

    out = subtract_intervals_df(allowed, exclude)

    assert list(zip(out["start"], out["end"])) == [(0, 80)]


def test_subtract_intervals_df_handles_gap_spanning_exclude():
    allowed = _bed([("chr1", 0, 10), ("chr1", 20, 30)])
    exclude = _bed([("chr1", 5, 25)])

    out = subtract_intervals_df(allowed, exclude)

    assert list(zip(out["start"], out["end"])) == [(0, 5), (25, 30)]


def test_subtract_intervals_df_margin_expands_exclusion():
    allowed = _bed([("chr1", 0, 100)])
    exclude = _bed([("chr1", 40, 60)])

    out = subtract_intervals_df(allowed, exclude, margin=10)

    assert list(zip(out["start"], out["end"])) == [(0, 30), (70, 100)]


def test_subtract_intervals_df_margin_merges_adjacent_excludes():
    allowed = _bed([("chr1", 0, 100)])
    exclude = _bed([("chr1", 20, 30), ("chr1", 40, 50)])

    # margin=6 expands to [14,36) and [34,56), which now overlap and merge
    out = subtract_intervals_df(allowed, exclude, margin=6)

    assert list(zip(out["start"], out["end"])) == [(0, 14), (56, 100)]


def test_subtract_intervals_df_ignores_other_chromosomes():
    allowed = _bed([("chr1", 0, 100)])
    exclude = _bed([("chr2", 0, 100)])

    out = subtract_intervals_df(allowed, exclude)

    assert list(zip(out["chrom"], out["start"], out["end"])) == [("chr1", 0, 100)]


def test_subtract_intervals_df_rejects_negative_margin():
    with pytest.raises(ValueError, match="margin"):
        subtract_intervals_df(_bed([("chr1", 0, 10)]), _bed([]), margin=-1)
