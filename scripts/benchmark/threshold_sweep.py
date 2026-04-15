#!/usr/bin/env python3
# threshold_sweep.py
# Author: Adam He <adamyhe@gmail.com>

"""Sweep a probability threshold and compute peak-level recall and FDR
against a ground truth BED at each level.

`benchmark_bw.py` operates at the bin level (AUROC/AUPRC).  This script
operates at the *peak level*: for each threshold it calls peaks from a
probability bigWig (by merging bins that pass), then compares those peaks to
a ground truth BED using a centre-window hit test.

  Recall  = fraction of GT peaks hit by ≥1 predicted peak centre-window
  FDR     = fraction of predicted peaks whose centre-window hits no GT peak
            = 1 − precision

Two input modes are supported (exactly one must be provided):

  -b / --bigwig   TROGDOR probability bigWig (output of ``trogdor score``).
                  The sweep is limited to the score range stored in the file;
                  re-score with a lower ``--min_score`` for a wider sweep.

  --dreg          dREG scored BED file.  dREG stores the centre 1 bp of each
                  100 bp scored window as each BED entry; ``--dreg_window``
                  controls the expansion half-width (default 50 bp).

Examples
--------
# TROGDOR bigWig
python scripts/benchmark/threshold_sweep.py \\
    -b predictions.prob.bw \\
    -t data/K562.positive.bed.gz \\
    --chroms chr1 chr2 \\
    --output sweep.tsv \\
    --figure sweep.png \\
    -v

# dREG scored BED
python scripts/benchmark/threshold_sweep.py \\
    --dreg sample.dREG.scores.bed.gz \\
    -t data/K562.positive.bed.gz \\
    --chroms chr1 chr2 \\
    --output dreg_sweep.tsv \\
    --figure dreg_sweep.png \\
    -v
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pybigtools


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep probability threshold; compute peak-level recall and FDR vs. ground truth."
    )
    # ---- Input (mutually exclusive; exactly one required) -------------------
    p.add_argument(
        "-b",
        "--bigwig",
        default=None,
        help="Probability bigWig (output of `trogdor score`). Mutually exclusive with --dreg.",
    )
    p.add_argument(
        "--dreg",
        default=None,
        metavar="BED",
        help="dREG scored BED file (centre 1 bp per scored region). Mutually exclusive with -b.",
    )
    p.add_argument(
        "--dreg_window",
        type=int,
        default=50,
        help="Half-width in bp to expand each dREG centre entry into its scored region "
        "(default: 50, giving 100 bp windows).",
    )
    p.add_argument(
        "--score_col",
        type=int,
        default=3,
        help="0-based column index of the score in the dREG BED (default: 3).",
    )
    # ---- Common arguments ---------------------------------------------------
    p.add_argument(
        "-t",
        "--truth",
        required=True,
        help="Ground truth peak BED file (≥3 cols; chrom/start/end; optionally gzipped)",
    )
    p.add_argument(
        "--n_thresholds",
        type=int,
        default=100,
        help="Number of evenly spaced threshold levels to evaluate (default: 100)",
    )
    p.add_argument(
        "--score_floor",
        type=float,
        default=None,
        help="Minimum score threshold to include in the sweep.  Useful for dREG logit input "
             "where very low thresholds produce enormous merged peaks that break the "
             "centre-window hit test.  Default: None (use the minimum observed score).",
    )
    p.add_argument(
        "--min_peaks",
        type=int,
        default=10,
        help="Minimum number of predicted peaks required to include a threshold level in the "
             "PRC plot and AUPRC calculation.  Thresholds producing fewer peaks have noisy "
             "precision estimates and cause the curve to collapse toward (0, 0).  "
             "Default: 10.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=200,
        help="Half-width in bp of the centre-window hit test (default: 200)",
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all chromosomes in the scored input)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write results to this TSV path (default: print to stdout only)",
    )
    p.add_argument(
        "--figure",
        default=None,
        help="Save a 3-panel summary figure to this path (e.g. sweep.png)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-chromosome progress",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_truth(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
        dtype={"chrom": str, "start": int, "end": int},
    )


def read_bigwig_intervals(bw_path, chroms, verbose):
    """Return a dict mapping chrom → sorted list of (start, end, score) tuples.

    Only intervals explicitly stored in the bigWig are returned; the caller is
    responsible for filtering by threshold.
    """
    bw = pybigtools.open(bw_path)
    chrom_sizes = dict(bw.chroms())
    scored = {}

    for chrom in chroms:
        if chrom not in chrom_sizes:
            if verbose:
                print(f"  Skipping {chrom} (not in bigWig)", flush=True)
            continue
        chrom_len = chrom_sizes[chrom]
        if verbose:
            print(f"  Reading {chrom} ({chrom_len:,} bp)...", flush=True)
        ivals = [
            (int(s), int(e), float(v))
            for s, e, v in bw.records(chrom, 0, chrom_len)
            if not (v != v)  # drop NaN
        ]
        if ivals:
            scored[chrom] = sorted(ivals)  # sorted by start (already is, but be safe)

    bw.close()
    return chrom_sizes, scored


def read_dreg_intervals(bed_path, chroms, score_col, verbose):
    """Read a dREG scored BED into the same per-chrom interval format.

    dREG stores the centre 1 bp of each scored window as each BED entry.
    Expansion from 1 bp to the full scored region happens at peak-calling
    time via ``call_peaks_at_threshold(expand=dreg_window)``.

    Parameters
    ----------
    bed_path : str
        Path to the dREG scored BED file (optionally gzipped).
    chroms : list of str
        Chromosomes to load.
    score_col : int
        0-based column index of the score (default 3 = 4th column).
    verbose : bool

    Returns
    -------
    scored : dict[str, list of (int, int, float)]
        Per-chrom sorted lists of (start, end, score) tuples.
    """
    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        compression="infer",
        usecols=[0, 1, 2, score_col],
        dtype={0: str, 1: int, 2: int, score_col: float},
    )
    df.columns = ["chrom", "start", "end", "score"]
    df = df.dropna(subset=["score"])

    scored = {}
    for chrom in chroms:
        sub = df[df["chrom"] == chrom].sort_values("start")
        if len(sub) == 0:
            if verbose:
                print(f"  Skipping {chrom} (not in dREG BED)", flush=True)
            continue
        if verbose:
            print(f"  {chrom}: {len(sub):,} dREG entries", flush=True)
        scored[chrom] = list(zip(sub["start"], sub["end"], sub["score"]))
    return scored


# ---------------------------------------------------------------------------
# Vectorised peak calling and hit test
# ---------------------------------------------------------------------------


def _merge_peaks_np(starts, ends, scores, threshold, expand=0):
    """Filter to score ≥ threshold, expand, and merge into peaks.

    Parameters
    ----------
    starts, ends, scores : np.ndarray
        Pre-sorted (by start) interval arrays for one chromosome.
    threshold : float
    expand : int
        Bases to add on each side before merging (dREG window expansion).

    Returns
    -------
    peak_starts, peak_ends, peak_scores : np.ndarray
        Merged peak arrays.  Empty arrays (length 0) when nothing passes.
    """
    _empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))
    mask = scores >= threshold
    if not mask.any():
        return _empty
    ps = starts[mask] - expand
    pe = ends[mask] + expand
    pv = scores[mask]
    # Group boundaries: new group wherever next start > current end
    group_idx = np.concatenate([[0], np.where(ps[1:] > pe[:-1])[0] + 1])
    return (
        ps[group_idx],
        np.maximum.reduceat(pe, group_idx),
        np.maximum.reduceat(pv, group_idx),
    )


def _centre_window_hits_np(peak_starts, peak_ends, gt_starts, gt_ends, window):
    """Vectorised centre-window hit test.

    Parameters
    ----------
    peak_starts, peak_ends : np.ndarray
        Merged predicted peak coordinates for one chromosome.
    gt_starts, gt_ends : np.ndarray
        Ground-truth peak coordinates, sorted by start.
    window : int
        Half-width of the centre window in bp.

    Returns
    -------
    pred_hits : np.ndarray of bool, shape (n_pred,)
    gt_recalled : np.ndarray of bool, shape (n_gt,)
    """
    n_pred = len(peak_starts)
    n_gt = len(gt_starts)
    if n_pred == 0:
        return np.empty(0, bool), np.zeros(n_gt, bool)
    if n_gt == 0:
        return np.zeros(n_pred, bool), np.empty(0, bool)

    centres = (peak_starts + peak_ends) // 2
    ws = centres - window
    we = centres + window  # half-open [ws, we)

    # Vectorised binary search over all peaks at once
    lo = np.searchsorted(gt_ends,    ws + 1, side="left")   # ge > ws
    hi = np.searchsorted(gt_starts,  we,     side="left")   # gs < we

    pred_hits   = lo < hi
    gt_recalled = np.zeros(n_gt, bool)
    for i in np.where(pred_hits)[0]:                        # loop only over TPs
        gt_recalled[lo[i] : hi[i]] = True

    return pred_hits, gt_recalled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # ---- Validate mutually exclusive inputs --------------------------------
    if (args.bigwig is None) == (args.dreg is None):
        print("Exactly one of -b/--bigwig or --dreg must be provided.", file=sys.stderr)
        sys.exit(1)

    expand = args.dreg_window if args.dreg else 0

    # ---- Load truth --------------------------------------------------------
    truth_df = read_truth(args.truth)
    truth_chroms = set(truth_df["chrom"].unique())

    # ---- Determine chromosome list and load scored intervals ---------------
    if args.bigwig:
        bw = pybigtools.open(args.bigwig)
        all_chroms = sorted(bw.chroms().keys())
        bw.close()
        chroms = args.chroms if args.chroms is not None else all_chroms
        chroms = [c for c in chroms if c in truth_chroms]
        if not chroms:
            print(
                "No chromosomes shared between bigWig and ground truth BED.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.verbose:
            print(
                f"Loading bigWig intervals for {len(chroms)} chromosome(s)...",
                flush=True,
            )
        _, scored = read_bigwig_intervals(args.bigwig, chroms, args.verbose)
    else:
        # dREG: derive chromosome list from truth (or user whitelist)
        df_tmp = pd.read_csv(
            args.dreg,
            sep="\t",
            header=None,
            compression="infer",
            usecols=[0],
            dtype={0: str},
        )
        dreg_chroms = set(df_tmp.iloc[:, 0].unique())
        all_chroms = sorted(dreg_chroms)
        chroms = args.chroms if args.chroms is not None else all_chroms
        chroms = [c for c in chroms if c in truth_chroms]
        if not chroms:
            print(
                "No chromosomes shared between dREG BED and ground truth BED.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.verbose:
            print(
                f"Loading dREG intervals for {len(chroms)} chromosome(s)...", flush=True
            )
        scored = read_dreg_intervals(args.dreg, chroms, args.score_col, args.verbose)

    if not scored:
        print(
            "No scored intervals found for the requested chromosomes.", file=sys.stderr
        )
        sys.exit(1)

    # ---- Pre-process: convert interval lists to sorted numpy arrays --------
    # Done once here so the threshold sweep only does fast numpy operations.
    processed = {}
    for chrom, ivals in scored.items():
        arr = np.array(ivals, dtype=np.float64)          # (N, 3): start, end, score
        order = np.argsort(arr[:, 0], kind="stable")     # sort by start (usually already sorted)
        processed[chrom] = (
            arr[order, 0].astype(np.int64),
            arr[order, 1].astype(np.int64),
            arr[order, 2].astype(np.float32),
        )

    # Pre-compute GT arrays per chrom (avoids repeated pandas queries in the hot loop)
    gt_arrays = {}
    for chrom in processed:
        gt_sub = truth_df[truth_df["chrom"] == chrom].sort_values("start")
        gt_arrays[chrom] = (
            gt_sub["start"].to_numpy(dtype=np.int64),
            gt_sub["end"].to_numpy(dtype=np.int64),
        )

    # ---- Build threshold grid from observed score range --------------------
    all_scores = np.concatenate([scores for _, _, scores in processed.values()])
    score_min = float(all_scores.min())
    score_max = float(all_scores.max())

    sweep_min = score_min if args.score_floor is None else max(score_min, args.score_floor)
    if sweep_min >= score_max:
        print(
            f"--score_floor ({args.score_floor}) is >= maximum observed score "
            f"({score_max:.4f}); no thresholds to evaluate.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verbose:
        source = "bigWig" if args.bigwig else "dREG BED"
        floor_note = f"  sweep from {sweep_min:.4f}" if args.score_floor is not None else ""
        print(
            f"Score range in {source}: [{score_min:.4f}, {score_max:.4f}]"
            f"{floor_note}  ({len(all_scores):,} entries)",
            flush=True,
        )

    thresholds = np.linspace(sweep_min, score_max, args.n_thresholds)

    # ---- Total GT peak count (fixed across thresholds) --------------------
    truth_on_chroms = truth_df[truth_df["chrom"].isin(chroms)]
    n_gt_total = len(truth_on_chroms)

    if n_gt_total == 0:
        print(
            "Ground truth BED has no peaks on the scored chromosomes.", file=sys.stderr
        )
        sys.exit(1)

    if args.verbose:
        print(
            f"Ground truth: {n_gt_total:,} peaks on {len(chroms)} chromosome(s)",
            flush=True,
        )

    # ---- Threshold sweep ---------------------------------------------------
    records = []

    for thr in thresholds:
        n_pred_total = 0
        n_pred_hits = 0  # predicted peaks that hit a GT peak (TP)
        gt_recalled_flags = []

        for chrom, (starts, ends, scores) in processed.items():
            gt_s, gt_e = gt_arrays[chrom]
            peak_s, peak_e, _ = _merge_peaks_np(starts, ends, scores, thr, expand)

            if len(peak_s) == 0:
                gt_recalled_flags.append(np.zeros(len(gt_s), bool))
                continue

            pred_hits, gt_recalled = _centre_window_hits_np(
                peak_s, peak_e, gt_s, gt_e, args.window
            )
            n_pred_total += len(peak_s)
            n_pred_hits += int(pred_hits.sum())
            gt_recalled_flags.append(gt_recalled)

        gt_recalled_all = (
            np.concatenate(gt_recalled_flags)
            if gt_recalled_flags
            else np.array([], dtype=bool)
        )
        n_gt_recalled = int(gt_recalled_all.sum())

        recall = n_gt_recalled / n_gt_total
        precision = (n_pred_hits / n_pred_total) if n_pred_total > 0 else float("nan")
        fdr = 1.0 - precision if n_pred_total > 0 else float("nan")

        records.append(
            {
                "threshold": thr,
                "n_pred_peaks": n_pred_total,
                "n_gt_recalled": n_gt_recalled,
                "recall": recall,
                "precision": precision,
                "peak_fdr": fdr,
            }
        )

    results = pd.DataFrame(records)

    # ---- Print / save table ------------------------------------------------
    header = (
        f"{'threshold':>12}  {'n_pred':>10}  {'n_recalled':>12}  "
        f"{'recall':>8}  {'precision':>10}  {'peak_fdr':>10}"
    )
    print(header)
    print("-" * len(header))
    for _, row in results.iterrows():
        print(
            f"{row['threshold']:12.6f}  {int(row['n_pred_peaks']):10,}  "
            f"{int(row['n_gt_recalled']):12,}  "
            f"{row['recall']:8.4f}  {row['precision']:10.4f}  {row['peak_fdr']:10.4f}"
        )

    if args.output is not None:
        results.to_csv(args.output, sep="\t", index=False, float_format="%.6g")
        if args.verbose:
            print(f"\nResults written to {args.output}")

    # ---- Optional figure ---------------------------------------------------
    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Peak-level threshold sweep  |  window=±{args.window} bp  |  "
            f"{n_gt_total:,} GT peaks  |  {len(chroms)} chroms",
            fontsize=9,
        )

        valid = results.dropna(subset=["recall", "precision"])

        # Panel 1: recall and FDR vs. threshold
        ax = axes[0]
        ax.plot(valid["threshold"], valid["recall"], color="steelblue", label="Recall")
        ax.plot(valid["threshold"], valid["peak_fdr"], color="tomato", label="Peak FDR")
        ax.set_xlabel("Score threshold")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title("Recall and FDR vs. threshold", fontsize=9)

        # Panel 2: n_pred_peaks vs. threshold
        ax = axes[1]
        ax.plot(valid["threshold"], valid["n_pred_peaks"], color="dimgray")
        ax.set_xlabel("Score threshold")
        ax.set_ylabel("# predicted peaks")
        ax.set_title("Peak count vs. threshold", fontsize=9)

        # Panel 3: precision–recall curve (peak-level)
        # Drop the high-threshold tail where too few peaks are called (noisy precision),
        # then anchor the curve at (recall=0, precision=1) — the conventional endpoint
        # when no peaks are predicted (zero false positives by definition).
        ax = axes[2]
        from sklearn.metrics import auc

        prc_data = valid[valid["n_pred_peaks"] >= args.min_peaks].copy()
        anchor = pd.DataFrame([{"recall": 0.0, "precision": 1.0}])
        prc_sorted = pd.concat([anchor, prc_data.sort_values("recall")], ignore_index=True)
        auprc = auc(prc_sorted["recall"].to_numpy(), prc_sorted["precision"].to_numpy())
        ax.plot(
            prc_sorted["recall"],
            prc_sorted["precision"],
            color="steelblue",
            marker="o",
            markersize=2,
        )
        ax.text(
            0.97,
            0.97,
            f"AUPRC = {auprc:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision (1 − FDR)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_title("Peak-level precision–recall curve", fontsize=9)

        plt.tight_layout()
        fig.savefig(args.figure, dpi=150)
        if args.verbose:
            print(f"Figure saved to {args.figure}")


if __name__ == "__main__":
    main()
