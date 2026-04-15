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

from chiaroscuro.utils import merge_intervals


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep probability threshold; compute peak-level recall and FDR vs. ground truth."
    )
    # ---- Input (mutually exclusive; exactly one required) -------------------
    p.add_argument(
        "-b", "--bigwig", default=None,
        help="Probability bigWig (output of `trogdor score`). Mutually exclusive with --dreg.",
    )
    p.add_argument(
        "--dreg", default=None, metavar="BED",
        help="dREG scored BED file (centre 1 bp per scored region). Mutually exclusive with -b.",
    )
    p.add_argument(
        "--dreg_window", type=int, default=50,
        help="Half-width in bp to expand each dREG centre entry into its scored region "
             "(default: 50, giving 100 bp windows).",
    )
    p.add_argument(
        "--score_col", type=int, default=3,
        help="0-based column index of the score in the dREG BED (default: 3).",
    )
    # ---- Common arguments ---------------------------------------------------
    p.add_argument(
        "-t", "--truth", required=True,
        help="Ground truth peak BED file (≥3 cols; chrom/start/end; optionally gzipped)",
    )
    p.add_argument(
        "--n_thresholds", type=int, default=100,
        help="Number of evenly spaced threshold levels to evaluate (default: 100)",
    )
    p.add_argument(
        "--window", type=int, default=200,
        help="Half-width in bp of the centre-window hit test (default: 200)",
    )
    p.add_argument(
        "--chroms", nargs="+", default=None,
        help="Chromosome whitelist (default: all chromosomes in the scored input)",
    )
    p.add_argument(
        "--output", default=None,
        help="Write results to this TSV path (default: print to stdout only)",
    )
    p.add_argument(
        "--figure", default=None,
        help="Save a 3-panel summary figure to this path (e.g. sweep.png)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
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
            if not (v != v)   # drop NaN
        ]
        if ivals:
            scored[chrom] = sorted(ivals)   # sorted by start (already is, but be safe)

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
# Peak calling at a threshold
# ---------------------------------------------------------------------------


def _merge_overlapping(intervals):
    """Merge overlapping or abutting sorted intervals, keeping max score.

    Unlike ``merge_intervals`` in utils.py (which handles only exactly-abutting
    intervals), this function handles the general overlapping case needed after
    dREG entries are expanded to their full window width.
    """
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e, v in intervals[1:]:
        if s <= merged[-1][1]:          # overlaps or abuts
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], v)
        else:
            merged.append([s, e, v])
    return merged


def call_peaks_at_threshold(scored_intervals, threshold, expand=0):
    """Filter intervals to score ≥ threshold, optionally expand, and merge.

    Parameters
    ----------
    scored_intervals : list of (start, end, score)
        Already sorted by start.
    threshold : float
    expand : int, optional
        Number of bp to add on each side of every passing interval before
        merging.  Use ``dreg_window`` for dREG input; leave 0 for bigWig
        input where bins are already full-width and exactly abutting.

    Returns
    -------
    list of [start, end, max_score]
        Merged peak intervals.
    """
    passing = [(s - expand, e + expand, v)
               for s, e, v in scored_intervals if v >= threshold]
    if expand > 0:
        return _merge_overlapping(sorted(passing))
    return merge_intervals(passing)


# ---------------------------------------------------------------------------
# Centre-window hit test
# ---------------------------------------------------------------------------


def centre_window_hits(pred_peaks, truth_df, chrom, window):
    """Return hit arrays for predicted peaks and ground truth peaks on *chrom*.

    A predicted peak is a "hit" (TP) if a ±window bp window around its centre
    overlaps any GT interval.  A GT peak is "recalled" if it is overlapped by
    at least one predicted centre window.

    Parameters
    ----------
    pred_peaks : list of [start, end, score]
        Merged predicted peaks on this chromosome.
    truth_df : pd.DataFrame
        Columns chrom/start/end for all chromosomes.
    chrom : str
    window : int

    Returns
    -------
    pred_hits : np.ndarray of bool, shape (n_pred,)
    gt_recalled : np.ndarray of bool, shape (n_gt_on_chrom,)
    """
    gt = truth_df[truth_df["chrom"] == chrom].sort_values("start")
    n_gt = len(gt)
    n_pred = len(pred_peaks)

    if n_pred == 0:
        return np.empty(0, dtype=bool), np.zeros(n_gt, dtype=bool)
    if n_gt == 0:
        return np.zeros(n_pred, dtype=bool), np.empty(0, dtype=bool)

    gt_starts = gt["start"].to_numpy()
    gt_ends   = gt["end"].to_numpy()

    pred_hits  = np.zeros(n_pred, dtype=bool)
    gt_recalled = np.zeros(n_gt,   dtype=bool)

    for i, (ps, pe, _) in enumerate(pred_peaks):
        centre = (ps + pe) // 2
        ws, we = centre - window, centre + window   # half-open [ws, we)

        # GT intervals [gs, ge) that overlap [ws, we): gs < we AND ge > ws
        lo = np.searchsorted(gt_ends,   ws + 1, side="left")   # ge > ws
        hi = np.searchsorted(gt_starts, we,     side="left")   # gs < we

        if lo < hi:
            pred_hits[i]     = True
            gt_recalled[lo:hi] = True

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
            print("No chromosomes shared between bigWig and ground truth BED.", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Loading bigWig intervals for {len(chroms)} chromosome(s)...", flush=True)
        _, scored = read_bigwig_intervals(args.bigwig, chroms, args.verbose)
    else:
        # dREG: derive chromosome list from truth (or user whitelist)
        df_tmp = pd.read_csv(
            args.dreg, sep="\t", header=None, compression="infer",
            usecols=[0], dtype={0: str},
        )
        dreg_chroms = set(df_tmp.iloc[:, 0].unique())
        all_chroms = sorted(dreg_chroms)
        chroms = args.chroms if args.chroms is not None else all_chroms
        chroms = [c for c in chroms if c in truth_chroms]
        if not chroms:
            print("No chromosomes shared between dREG BED and ground truth BED.", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Loading dREG intervals for {len(chroms)} chromosome(s)...", flush=True)
        scored = read_dreg_intervals(args.dreg, chroms, args.score_col, args.verbose)

    if not scored:
        print("No scored intervals found for the requested chromosomes.", file=sys.stderr)
        sys.exit(1)

    # ---- Build threshold grid from observed score range --------------------
    all_scores = np.concatenate([
        np.array([v for _, _, v in ivals], dtype=np.float32)
        for ivals in scored.values()
    ])
    score_min = float(all_scores.min())
    score_max = float(all_scores.max())

    if args.verbose:
        source = "bigWig" if args.bigwig else "dREG BED"
        print(
            f"Score range in {source}: [{score_min:.4f}, {score_max:.4f}] "
            f"across {len(all_scores):,} entries",
            flush=True,
        )

    thresholds = np.linspace(score_min, score_max, args.n_thresholds)

    # ---- Total GT peak count (fixed across thresholds) --------------------
    truth_on_chroms = truth_df[truth_df["chrom"].isin(chroms)]
    n_gt_total = len(truth_on_chroms)

    if n_gt_total == 0:
        print("Ground truth BED has no peaks on the scored chromosomes.", file=sys.stderr)
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
        n_pred_hits  = 0   # predicted peaks that hit a GT peak (TP)
        gt_recalled_flags = []

        for chrom, ivals in scored.items():
            peaks = call_peaks_at_threshold(ivals, thr, expand=expand)
            if len(peaks) == 0:
                # Count GT peaks on this chrom as un-recalled
                n_gt_chrom = (truth_df["chrom"] == chrom).sum()
                gt_recalled_flags.append(np.zeros(n_gt_chrom, dtype=bool))
                continue

            pred_hits, gt_recalled = centre_window_hits(peaks, truth_df, chrom, args.window)
            n_pred_total += len(peaks)
            n_pred_hits  += int(pred_hits.sum())
            gt_recalled_flags.append(gt_recalled)

        gt_recalled_all = np.concatenate(gt_recalled_flags) if gt_recalled_flags else np.array([], dtype=bool)
        n_gt_recalled = int(gt_recalled_all.sum())

        recall    = n_gt_recalled / n_gt_total
        precision = (n_pred_hits / n_pred_total) if n_pred_total > 0 else float("nan")
        fdr       = 1.0 - precision if n_pred_total > 0 else float("nan")

        records.append({
            "threshold":     thr,
            "n_pred_peaks":  n_pred_total,
            "n_gt_recalled": n_gt_recalled,
            "recall":        recall,
            "precision":     precision,
            "peak_fdr":      fdr,
        })

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
        ax.plot(valid["threshold"], valid["recall"],   color="steelblue", label="Recall")
        ax.plot(valid["threshold"], valid["peak_fdr"], color="tomato",    label="Peak FDR")
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
        ax = axes[2]
        ax.plot(valid["recall"], valid["precision"], color="steelblue", marker="o", markersize=2)
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
