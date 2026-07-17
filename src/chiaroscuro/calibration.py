"""Reusable helpers for empirical peak calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import score_peaks, score_peaks_from_array
from .utils import merge_intervals


def records_to_bed3(records):
    """Convert peak records to a BED3 DataFrame."""
    return pd.DataFrame(
        [(r["chrom"], int(r["start"]), int(r["end"])) for r in records],
        columns=["chrom", "start", "end"],
    )


def records_to_summit_bed3(records):
    """Convert peak records to summit-sized BED3 intervals."""
    return pd.DataFrame(
        [
            (r["chrom"], int(r["summit_start"]), int(r["summit_end"]))
            for r in records
        ],
        columns=["chrom", "start", "end"],
    )


def candidate_intervals_to_bed3(chrom, intervals):
    """Convert scored candidate intervals to merged BED3 intervals."""
    if not intervals:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    rows = [
        (chrom, int(start), int(end))
        for start, end, _ in merge_intervals(intervals)
    ]
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def finite_scores(scores):
    """Return finite calibration scores as float32."""
    scores = np.asarray(scores, dtype=np.float32)
    return scores[~np.isnan(scores)]


def score_centered_windows_from_array(
    intervals_df, scores, chrom, output_stride, smooth_bins
):
    """Score summit-centered windows from a per-chromosome score array."""
    sub = intervals_df[intervals_df["chrom"] == chrom]
    out = np.full(len(sub), np.nan, dtype=np.float32)
    left = (smooth_bins - 1) // 2
    right = smooth_bins // 2
    scores = np.asarray(scores, dtype=np.float32)
    for j, (_, row) in enumerate(sub.iterrows()):
        center_bp = (int(row["start"]) + int(row["end"]) - 1) // 2
        center_bin = max(0, center_bp // output_stride)
        lo = max(0, center_bin - left)
        hi = min(len(scores), center_bin + right + 1)
        vals = scores[lo:hi]
        if len(vals) == 0:
            continue
        out[j] = np.nan_to_num(vals).mean()
    return out


def score_centered_windows_from_bigwig(
    intervals_df, bw, chrom_sizes, chrom, output_stride, smooth_bins
):
    """Score summit-centered windows from a bigWig probability track."""
    sub = intervals_df[intervals_df["chrom"] == chrom]
    out = np.full(len(sub), np.nan, dtype=np.float32)
    if chrom not in chrom_sizes:
        return out
    chrom_len = chrom_sizes[chrom]
    half_left = ((smooth_bins - 1) // 2) * output_stride
    half_right = (smooth_bins // 2 + 1) * output_stride
    for j, (_, row) in enumerate(sub.iterrows()):
        center_bp = (int(row["start"]) + int(row["end"]) - 1) // 2
        center_bin_start = (center_bp // output_stride) * output_stride
        start = max(0, center_bin_start - half_left)
        end = min(chrom_len, center_bin_start + half_right)
        if start >= end:
            continue
        vals = np.nan_to_num(np.array(bw.values(chrom, start, end), dtype=np.float32))
        if len(vals) == 0:
            continue
        out[j] = vals.mean()
    return out


def score_peak_records_from_array(
    records, scores, chrom, output_stride, stat, smooth_bins=1
):
    """Score peak records from a per-chromosome score array."""
    if stat == "summit":
        return np.asarray([r["summit_score"] for r in records], dtype=np.float32)
    if stat == "smoothed_summit":
        summit_df = records_to_summit_bed3(records)
        return score_centered_windows_from_array(
            summit_df, scores, chrom, output_stride, smooth_bins
        )
    peaks_df = records_to_bed3(records)
    return score_peaks_from_array(scores, peaks_df, chrom, output_stride, stat)


def score_peak_records_from_bigwig(
    records, bw, chrom_sizes, chrom, stat, output_stride=16, smooth_bins=1
):
    """Score peak records from a bigWig probability track."""
    if stat == "summit":
        return np.asarray([r["summit_score"] for r in records], dtype=np.float32)
    if stat == "smoothed_summit":
        summit_df = records_to_summit_bed3(records)
        return score_centered_windows_from_bigwig(
            summit_df, bw, chrom_sizes, chrom, output_stride, smooth_bins
        )
    peaks_df = records_to_bed3(records)
    return score_peaks(bw, peaks_df, chrom_sizes, stat, [chrom])


def null_log_records(null_df, scores):
    """Pair null-draw positions with their scores for diagnostic logging.

    ``null_df`` and ``scores`` must be row-aligned (as returned together by a
    ``shuffle_peaks_within_intervals`` call and the scoring function applied
    to it). Rows with a non-finite score are dropped.

    Returns
    -------
    list of (chrom, start, end, score)
    """
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.isfinite(scores)
    if not mask.any():
        return []
    chroms = null_df["chrom"].to_numpy()[mask]
    starts = null_df["start"].to_numpy(dtype=np.int64)[mask]
    ends = null_df["end"].to_numpy(dtype=np.int64)[mask]
    return list(zip(chroms.tolist(), starts.tolist(), ends.tolist(), scores[mask].tolist()))


def write_null_log(path, records):
    """Write logged null-draw positions and scores to a TSV (optionally .gz).

    ``records`` is a list of (chrom, start, end, score) tuples, typically
    accumulated across all shuffles/chromosomes via ``null_log_records``.
    Intended for diagnosing calibration (e.g. checking whether high-scoring
    null draws cluster near real peak summits) rather than for production
    use, so no bgzip/tabix support — this can be large in candidate-null
    scope with many shuffles.
    """
    table = pd.DataFrame(records, columns=["chrom", "start", "end", "score"])
    table.to_csv(path, sep="\t", index=False, float_format="%.6g")


def write_calibration_table(path, thresholds, n_real, n_null, fdr, n_total):
    """Write an empirical FDR curve TSV."""
    recall = np.divide(
        n_real,
        n_total,
        out=np.zeros_like(n_real, dtype=float),
        where=n_total > 0,
    )
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "n_real": n_real.astype(int),
            "n_null": n_null,
            "fdr": fdr,
            "recall_proxy": recall,
        }
    )
    table.to_csv(path, sep="\t", index=False, float_format="%.6g")


def write_calibration_figure(
    path,
    real_scores,
    null_scores,
    thresholds,
    n_real,
    fdr,
    stat,
    fdr_target,
    threshold_at_target,
    plot_scale="logit",
):
    """Write a diagnostic empirical FDR figure."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "--calibration_figure requires matplotlib; install TROGDOR with "
            "plotting/dev dependencies or omit this option."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_scale == "logit":
        if (
            min(real_scores.min(), null_scores.min() if len(null_scores) else 1) < 0
            or max(real_scores.max(), null_scores.max() if len(null_scores) else 0) > 1
        ):
            raise ValueError(
                "--calibration_plot_scale logit requires scores in [0, 1]."
            )

        def _plot_x(x):
            x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))

        x_label = f"Logit peak score ({stat})"
    else:
        def _plot_x(x):
            return np.asarray(x, dtype=np.float64)

        x_label = f"Peak score ({stat})"

    real_plot = _plot_x(real_scores)
    null_plot = _plot_x(null_scores) if len(null_scores) else null_scores
    thresholds_plot = _plot_x(thresholds)
    threshold_at_target_plot = (
        float(_plot_x([threshold_at_target])[0])
        if not np.isnan(threshold_at_target)
        else float("nan")
    )

    recall = np.divide(
        n_real,
        len(real_scores),
        out=np.zeros_like(n_real, dtype=float),
        where=len(real_scores) > 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Empirical calibration ({stat})", fontsize=10)

    ax = axes[0]
    if len(thresholds_plot) > 1 and thresholds_plot[0] != thresholds_plot[-1]:
        bins = np.linspace(thresholds_plot[0], thresholds_plot[-1], 60)
    else:
        center = float(thresholds_plot[0]) if len(thresholds_plot) else 0.0
        bins = np.linspace(center - 0.5, center + 0.5, 20)
    ax.hist(
        real_plot,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="real",
    )
    if len(null_scores) > 0:
        ax.hist(
            null_plot,
            bins=bins,
            density=True,
            alpha=0.5,
            color="salmon",
            label="null",
        )
    if not np.isnan(threshold_at_target):
        ax.axvline(
            threshold_at_target_plot,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"t={threshold_at_target:.6g}",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title("Score distributions")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(thresholds_plot, fdr, color="black", linewidth=1.5, label="FDR")
    ax.axhline(
        fdr_target,
        color="firebrick",
        linestyle="--",
        linewidth=0.8,
        label=f"FDR={fdr_target:.3f}",
    )
    if not np.isnan(threshold_at_target):
        ax.axvline(
            threshold_at_target_plot,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label=f"t={threshold_at_target:.6g}",
        )
    ax.set_xlabel(x_label.replace("Peak score", "Score threshold"))
    ax.set_ylabel("Empirical FDR")
    ax.set_title("FDR and retained fraction")
    ax.set_ylim(0, 1.05)

    ax2 = ax.twinx()
    ax2.plot(
        thresholds_plot,
        recall,
        color="steelblue",
        linewidth=1.5,
        label="Retained fraction",
    )
    ax2.set_ylabel("Retained fraction", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
