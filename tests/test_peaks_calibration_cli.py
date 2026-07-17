"""Integration tests for ``trogdor peaks --calibrate`` against a real bigWig.

These exercise ``cmd_peaks``'s calibrated branch end-to-end: reading scores
from an on-disk bigWig, calling profile peaks, building an empirical null by
shuffling, and writing raw/calibrated BED outputs. Marked ``integration``
since they do real bigWig I/O (see tests/conftest.py).
"""

import argparse

import pandas as pd
import pybigtools
import pytest

from cli.commands import cmd_peaks

pytestmark = pytest.mark.integration


def _write_bigwig(path, chrom, chrom_len, stride, records):
    """Write a dense bigWig covering [0, chrom_len) in ``stride``-bp bins.

    ``records`` is a list of (start, end, score) overrides; any bin not
    covered by a record gets ``background_score``.
    """
    bw = pybigtools.open(str(path), "w")
    bw.write({chrom: chrom_len}, [(chrom, s, e, float(v)) for s, e, v in records])
    bw.close()


def _dense_intervals(chrom_len, stride, background_score, bumps):
    """Build stride-aligned (start, end, score) triples covering the whole
    chromosome, with ``bumps`` (start, end, score) overrides layered on top."""
    intervals = []
    for start in range(0, chrom_len, stride):
        end = start + stride
        score = background_score
        for b_start, b_end, b_score in bumps:
            if b_start <= start < b_end:
                score = b_score
                break
        intervals.append((start, end, score))
    return intervals


def _base_peaks_args(bw_path, output, **overrides):
    args = dict(
        input=str(bw_path),
        output=str(output),
        min_score=0.9,
        mode="profile",
        max_gap=0,
        min_width=0,
        max_width=None,
        seed_score=None,
        smooth_bins=3,
        valley_fraction=0.5,
        boundary_fraction=0.0,
        min_support_signal=0.0,
        support_plus_bigwig=None,
        support_minus_bigwig=None,
        calibrate=True,
        raw_output=None,
        calibration_fdr_target=0.5,
        calibration_curve=None,
        calibration_figure=None,
        threshold_grid="quantile",
        calibration_plot_scale="logit",
        calibration_stat="summit",
        null_scope="genome",
        n_shuffle=5,
        n_thresholds=20,
        calibration_seed=0,
        verbose=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _read_bed(path):
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [line.rstrip("\n").split("\t") for line in fh if line.strip()]


def test_cmd_peaks_calibrate_genome_null_end_to_end(tmp_path):
    chrom, chrom_len, stride = "chr1", 16000, 16
    bw_path = tmp_path / "scores.bw"
    intervals = _dense_intervals(
        chrom_len, stride, background_score=0.05, bumps=[(8000, 8048, 0.97)]
    )
    _write_bigwig(bw_path, chrom, chrom_len, stride, intervals)

    output = tmp_path / "peaks.calibrated.bed.gz"
    raw_output = tmp_path / "peaks.raw.bed.gz"
    curve_path = tmp_path / "fdr_curve.tsv"

    args = _base_peaks_args(
        bw_path,
        output,
        raw_output=str(raw_output),
        calibration_curve=str(curve_path),
        null_scope="genome",
    )
    cmd_peaks(args)

    assert raw_output.exists()
    assert output.exists()
    assert curve_path.exists()

    raw_rows = _read_bed(raw_output)
    calibrated_rows = _read_bed(output)

    assert len(raw_rows) == 1
    assert len(calibrated_rows) <= len(raw_rows)
    # profile mode writes chrom/start/end/score/summit_start/summit_end/summit_score
    assert len(raw_rows[0]) == 7

    curve = curve_path.read_text().splitlines()
    assert curve[0].split("\t") == [
        "threshold",
        "n_real",
        "n_null",
        "fdr",
        "recall_proxy",
    ]
    assert len(curve) > 1


def test_cmd_peaks_calibrate_writes_null_log(tmp_path):
    chrom, chrom_len, stride = "chr1", 16000, 16
    bw_path = tmp_path / "scores.bw"
    intervals = _dense_intervals(
        chrom_len,
        stride,
        background_score=0.05,
        bumps=[(8000, 8048, 0.97), (2000, 2064, 0.55)],
    )
    _write_bigwig(bw_path, chrom, chrom_len, stride, intervals)

    output = tmp_path / "peaks.calibrated.bed.gz"
    raw_output = tmp_path / "peaks.raw.bed.gz"
    null_log_path = tmp_path / "null_log.tsv"

    args = _base_peaks_args(
        bw_path,
        output,
        raw_output=str(raw_output),
        seed_score=0.5,
        null_scope="candidate",
        n_shuffle=3,
        calibration_null_log=str(null_log_path),
    )
    cmd_peaks(args)

    assert null_log_path.exists()
    table = pd.read_csv(null_log_path, sep="\t")
    assert list(table.columns) == ["chrom", "start", "end", "score"]
    # one raw peak x 3 shuffles worth of null draws logged
    assert len(table) == 3
    assert (table["chrom"] == chrom).all()
    assert (table["end"] > table["start"]).all()


def _raw_peaks_df(raw_output):
    return pd.read_csv(
        raw_output,
        sep="\t",
        header=None,
        names=[
            "chrom",
            "start",
            "end",
            "score",
            "summit_start",
            "summit_end",
            "summit_score",
        ],
    )


def _null_overlaps_any_peak_margin(null_df, peaks_df, margin):
    for _, peak in peaks_df.iterrows():
        lo, hi = peak["start"] - margin, peak["end"] + margin
        same_chrom = null_df[null_df["chrom"] == peak["chrom"]]
        if len(same_chrom[(same_chrom["end"] > lo) & (same_chrom["start"] < hi)]) > 0:
            return True
    return False


def _ramp_bumps(start, stride, scores):
    return [
        (start + i * stride, start + (i + 1) * stride, score)
        for i, score in enumerate(scores)
    ]


def _two_peaks_plus_distractor_bumps(stride):
    # Two single-summit ramps (seed_score=0.5 candidate blocks, each 5 bins
    # wide) plus a flat distractor block that clears seed_score but never
    # min_score, so it never becomes a peak — genuine non-peak candidate
    # territory for null placement to fall back on once both real peaks
    # (plus margin) are excluded.
    return (
        _ramp_bumps(2000, stride, [0.55, 0.75, 0.97, 0.85, 0.6])
        + _ramp_bumps(8000, stride, [0.55, 0.75, 0.96, 0.85, 0.6])
        + [(4000, 4400, 0.6)]
    )


def test_cmd_peaks_calibrate_null_exclusion_margin_keeps_null_away_from_peaks(tmp_path):
    chrom, chrom_len, stride = "chr1", 16000, 16
    bw_path = tmp_path / "scores.bw"
    # Mirrors the real self-referential-null scenario where candidate space
    # is mostly real peaks' own footprints, but keeps one non-peak
    # distractor block so there's still somewhere valid to place null draws
    # after both real peaks are excluded with a margin.
    intervals = _dense_intervals(
        chrom_len,
        stride,
        background_score=0.05,
        bumps=_two_peaks_plus_distractor_bumps(stride),
    )
    _write_bigwig(bw_path, chrom, chrom_len, stride, intervals)

    output = tmp_path / "peaks.calibrated.bed.gz"
    raw_output = tmp_path / "peaks.raw.bed.gz"
    null_log_path = tmp_path / "null_log.tsv"
    margin = 200

    args = _base_peaks_args(
        bw_path,
        output,
        raw_output=str(raw_output),
        seed_score=0.5,
        null_scope="candidate",
        n_shuffle=50,
        calibration_null_log=str(null_log_path),
        null_exclusion_margin=margin,
        calibration_seed=1,
    )
    cmd_peaks(args)

    raw = _raw_peaks_df(raw_output)
    null = pd.read_csv(null_log_path, sep="\t")
    assert len(raw) == 2
    assert len(null) > 0
    assert not _null_overlaps_any_peak_margin(null, raw, margin)


def test_cmd_peaks_calibrate_without_null_exclusion_margin_null_overlaps_peaks(tmp_path):
    # Same setup as above but with the fix disabled (the default) — shows
    # this is a meaningful regression test, not one that trivially passes
    # regardless of the flag.
    chrom, chrom_len, stride = "chr1", 16000, 16
    bw_path = tmp_path / "scores.bw"
    intervals = _dense_intervals(
        chrom_len,
        stride,
        background_score=0.05,
        bumps=_two_peaks_plus_distractor_bumps(stride),
    )
    _write_bigwig(bw_path, chrom, chrom_len, stride, intervals)

    output = tmp_path / "peaks.calibrated.bed.gz"
    raw_output = tmp_path / "peaks.raw.bed.gz"
    null_log_path = tmp_path / "null_log.tsv"

    args = _base_peaks_args(
        bw_path,
        output,
        raw_output=str(raw_output),
        seed_score=0.5,
        null_scope="candidate",
        n_shuffle=50,
        calibration_null_log=str(null_log_path),
        null_exclusion_margin=None,
        calibration_seed=1,
    )
    cmd_peaks(args)

    raw = _raw_peaks_df(raw_output)
    null = pd.read_csv(null_log_path, sep="\t")
    assert _null_overlaps_any_peak_margin(null, raw, margin=0)


def test_cmd_peaks_calibrate_candidate_null_runs(tmp_path):
    chrom, chrom_len, stride = "chr1", 16000, 16
    bw_path = tmp_path / "scores.bw"
    # A called peak (clears min_score) plus a separate, weaker candidate
    # region that never clears min_score but still seeds a block — this
    # gives the candidate-scope null somewhere else to place shuffled peaks
    # besides the called peak's own footprint.
    intervals = _dense_intervals(
        chrom_len,
        stride,
        background_score=0.05,
        bumps=[(8000, 8048, 0.97), (2000, 2064, 0.55)],
    )
    _write_bigwig(bw_path, chrom, chrom_len, stride, intervals)

    output = tmp_path / "peaks.calibrated.bed.gz"
    raw_output = tmp_path / "peaks.raw.bed.gz"

    args = _base_peaks_args(
        bw_path,
        output,
        raw_output=str(raw_output),
        seed_score=0.5,
        null_scope="candidate",
    )
    cmd_peaks(args)

    raw_rows = _read_bed(raw_output)
    calibrated_rows = _read_bed(output)

    assert len(raw_rows) == 1
    assert len(calibrated_rows) <= len(raw_rows)


def test_cmd_peaks_calibrate_validates_args(tmp_path):
    bw_path = tmp_path / "scores.bw"
    _write_bigwig(
        bw_path, "chr1", 1024, 16, _dense_intervals(1024, 16, 0.05, [])
    )
    output = tmp_path / "peaks.bed.gz"

    args = _base_peaks_args(bw_path, output, n_shuffle=0)
    with pytest.raises(ValueError, match="n_shuffle"):
        cmd_peaks(args)
