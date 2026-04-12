#!/usr/bin/env python3
# compare_peaks.py
# Author: Adam He <adamyhe@gmail.com>

"""
compare_peaks.py — Compare a candidate peak BED against a ground truth BED.

Because peak callers produce peaks of varying widths, naïve per-interval
overlap counts are length-biased.  This script addresses that with two
complementary metric tracks:

  Bin-level   Both peak sets are rasterised to binary bin vectors at
              --output_stride resolution (default 16 bp).  Precision,
              recall, F1, and Jaccard are computed from these vectors.
              This is length-fair because each genomic position contributes
              equally regardless of the peak it falls in.

  Peak-level  For each ground truth peak, the fraction of its bases covered
  (length-    by candidate peaks is computed (sensitivity / recall).  For
  normalised) each candidate peak, the fraction covered by ground truth is
              computed (PPV / precision).  Both binary hit-rates and mean
              coverage fractions are reported.

  Center-     A candidate peak is a "good call" if a ±--window bp region
  window      around its centre intersects any ground truth peak.
              Sensitivity = fraction of GT peaks hit by ≥1 candidate window.
              Specificity = fraction of candidate windows that hit GT.

Both BED files are assumed to have exactly three columns: chrom, start, end.

Usage
-----
python compare_peaks.py \\
    -c candidate.bed.gz \\
    -t truth.bed.gz \\
    -g hg38.chrom.sizes \\
    --output_stride 16 \\
    --figure comparison.png \\
    -v
"""

import argparse
import sys

import numpy as np
import pandas as pd

from chiaroscuro.utils import encode_labels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare a candidate peak BED against a ground truth BED."
    )
    p.add_argument(
        "-c", "--candidate", required=True,
        help="Candidate peak BED file (chrom/start/end, optionally gzipped)",
    )
    p.add_argument(
        "-t", "--truth", required=True,
        help="Ground truth peak BED file (chrom/start/end, optionally gzipped)",
    )
    p.add_argument(
        "-g", "--chrom_sizes", required=True,
        help="Tab-separated chrom.sizes file (e.g. hg38.chrom.sizes)",
    )
    p.add_argument(
        "--output_stride", type=int, default=16,
        help="Bin size in bp for bin-level metrics (default: 16)",
    )
    p.add_argument(
        "--chroms", nargs="+", default=None,
        help="Chromosome whitelist (default: all chromosomes in chrom_sizes)",
    )
    p.add_argument(
        "--window", type=int, default=200,
        help="Half-width in bp of the centre window for sensitivity/specificity (default: 200)",
    )
    p.add_argument(
        "--figure", default=None,
        help="Save a summary figure to this path (e.g. comparison.png)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-chromosome progress",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# BED I/O
# ---------------------------------------------------------------------------


def read_bed(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
        dtype={"chrom": str, "start": int, "end": int},
    )


# ---------------------------------------------------------------------------
# Peak-level length-normalised overlap
# ---------------------------------------------------------------------------


def coverage_fractions(query_df, subject_df, chrom):
    """Return a float32 array of per-peak coverage fractions for query peaks
    on ``chrom``, where coverage is the number of query bases overlapped by
    any subject peak divided by the query peak length.

    Parameters
    ----------
    query_df : pd.DataFrame
        Columns chrom/start/end.  Fractions are computed for peaks on ``chrom``.
    subject_df : pd.DataFrame
        Columns chrom/start/end.  These are the peaks providing coverage.
    chrom : str

    Returns
    -------
    np.ndarray of float32, shape (n_query_on_chrom,)
        Coverage fraction in [0, 1] for each query peak.
    """
    q = query_df[query_df["chrom"] == chrom].sort_values("start")
    s = subject_df[subject_df["chrom"] == chrom].sort_values("start")

    if len(q) == 0:
        return np.empty(0, dtype=np.float32)

    q_starts = q["start"].to_numpy()
    q_ends = q["end"].to_numpy()
    q_lens = (q_ends - q_starts).astype(np.float32)

    if len(s) == 0:
        return np.zeros(len(q), dtype=np.float32)

    s_starts = s["start"].to_numpy()
    s_ends = s["end"].to_numpy()

    fracs = np.empty(len(q), dtype=np.float32)
    for i in range(len(q)):
        qs, qe = q_starts[i], q_ends[i]
        # subject peaks that could overlap [qs, qe)
        # a subject peak [ss, se) overlaps if ss < qe and se > qs
        lo = np.searchsorted(s_ends, qs + 1, side="left")   # se > qs
        hi = np.searchsorted(s_starts, qe, side="left")      # ss < qe
        if lo >= hi:
            fracs[i] = 0.0
            continue
        # sum intersection lengths
        covered = np.sum(
            np.minimum(s_ends[lo:hi], qe) - np.maximum(s_starts[lo:hi], qs)
        )
        fracs[i] = covered / q_lens[i]

    return fracs


# ---------------------------------------------------------------------------
# Center-window hits
# ---------------------------------------------------------------------------


def center_window_hits(query_df, subject_df, chrom, window):
    """Return boolean arrays indicating which query peaks and subject peaks
    have a centre-window hit.

    For each query peak a window [centre - window, centre + window) is built,
    where centre = (start + end) // 2.  A hit is recorded when the window
    overlaps at least one subject interval.

    Parameters
    ----------
    query_df : pd.DataFrame
        Columns chrom/start/end.  Centres are taken from these peaks.
    subject_df : pd.DataFrame
        Columns chrom/start/end.  These provide the target intervals.
    chrom : str
    window : int
        Half-width of the centre window in base pairs.

    Returns
    -------
    q_hits : np.ndarray of bool, shape (n_query_on_chrom,)
        True where query centre window overlaps a subject interval.
    s_hits : np.ndarray of bool, shape (n_subject_on_chrom,)
        True where subject interval is overlapped by at least one query window.
    """
    q = query_df[query_df["chrom"] == chrom].sort_values("start")
    s = subject_df[subject_df["chrom"] == chrom].sort_values("start")

    n_q = len(q)
    n_s = len(s)

    if n_q == 0:
        return np.empty(0, dtype=bool), np.zeros(n_s, dtype=bool)
    if n_s == 0:
        return np.zeros(n_q, dtype=bool), np.empty(0, dtype=bool)

    centers = ((q["start"].to_numpy() + q["end"].to_numpy()) // 2)
    w_starts = centers - window
    w_ends   = centers + window   # half-open: [centre-W, centre+W)

    s_starts = s["start"].to_numpy()
    s_ends   = s["end"].to_numpy()

    q_hits = np.zeros(n_q, dtype=bool)
    s_hits = np.zeros(n_s, dtype=bool)

    for i in range(n_q):
        ws, we = w_starts[i], w_ends[i]
        # subject intervals [ss, se) overlap [ws, we) iff ss < we and se > ws
        lo = np.searchsorted(s_ends,    ws + 1, side="left")   # se > ws
        hi = np.searchsorted(s_starts,  we,     side="left")   # ss < we
        if lo < hi:
            q_hits[i] = True
            s_hits[lo:hi] = True

    return q_hits, s_hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Read chrom sizes
    chrom_sizes = pd.read_csv(
        args.chrom_sizes, sep="\t", header=None, names=["chrom", "length"],
        dtype={"chrom": str, "length": int},
    )
    chrom_sizes = dict(zip(chrom_sizes["chrom"], chrom_sizes["length"]))

    cand_df = read_bed(args.candidate)
    truth_df = read_bed(args.truth)

    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())
    # Restrict to chromosomes present in both BEDs and chrom_sizes
    cand_chroms = set(cand_df["chrom"].unique())
    truth_chroms = set(truth_df["chrom"].unique())
    chroms = [c for c in chroms if c in chrom_sizes and c in cand_chroms and c in truth_chroms]

    if not chroms:
        print("No chromosomes found in both BED files and chrom_sizes.", file=sys.stderr)
        sys.exit(1)

    # Accumulate bin-level arrays
    bin_cand_all = []
    bin_truth_all = []

    # Accumulate peak-level coverage fractions
    gt_fracs_all = []    # GT peak covered by candidates
    cand_fracs_all = []  # candidate peak covered by GT

    # Accumulate center-window hits
    cand_win_hits_all = []  # per-candidate: centre window hits GT
    gt_win_hits_all   = []  # per-GT: hit by at least one candidate window

    for chrom in chroms:
        chrom_len = chrom_sizes[chrom]
        if args.verbose:
            n_c = (cand_df["chrom"] == chrom).sum()
            n_t = (truth_df["chrom"] == chrom).sum()
            print(f"  {chrom}: {n_t:,} GT peaks, {n_c:,} candidate peaks", flush=True)

        # --- Bin-level ---
        bin_truth = encode_labels(truth_df, chrom, chrom_len, args.output_stride)
        bin_cand = encode_labels(cand_df, chrom, chrom_len, args.output_stride)
        bin_truth_all.append(bin_truth)
        bin_cand_all.append(bin_cand)

        # --- Peak-level ---
        gt_fracs_all.append(coverage_fractions(truth_df, cand_df, chrom))
        cand_fracs_all.append(coverage_fractions(cand_df, truth_df, chrom))

        # --- Center-window ---
        cw, gw = center_window_hits(cand_df, truth_df, chrom, args.window)
        cand_win_hits_all.append(cw)
        gt_win_hits_all.append(gw)

    # --- Bin-level metrics ---
    bin_truth = np.concatenate(bin_truth_all).astype(bool)
    bin_cand = np.concatenate(bin_cand_all).astype(bool)

    tp = np.sum(bin_cand & bin_truth)
    fp = np.sum(bin_cand & ~bin_truth)
    fn = np.sum(~bin_cand & bin_truth)

    bin_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    bin_recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    bin_f1        = (
        2 * bin_precision * bin_recall / (bin_precision + bin_recall)
        if (bin_precision + bin_recall) > 0 else float("nan")
    )
    bin_jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

    # --- Peak-level metrics ---
    gt_fracs   = np.concatenate(gt_fracs_all)
    cand_fracs = np.concatenate(cand_fracs_all)

    n_gt   = len(gt_fracs)
    n_cand = len(cand_fracs)

    gt_hit_rate   = np.mean(gt_fracs   > 0) if n_gt   > 0 else float("nan")
    cand_hit_rate = np.mean(cand_fracs > 0) if n_cand > 0 else float("nan")
    gt_mean_cov   = float(np.mean(gt_fracs))   if n_gt   > 0 else float("nan")
    cand_mean_cov = float(np.mean(cand_fracs)) if n_cand > 0 else float("nan")

    n_gt_hit   = int(np.sum(gt_fracs   > 0))
    n_cand_hit = int(np.sum(cand_fracs > 0))

    # --- Center-window metrics ---
    cand_win_hits = np.concatenate(cand_win_hits_all)
    gt_win_hits   = np.concatenate(gt_win_hits_all)

    cw_sensitivity   = float(np.mean(gt_win_hits))   if len(gt_win_hits)   > 0 else float("nan")
    cw_specificity   = float(np.mean(cand_win_hits))  if len(cand_win_hits) > 0 else float("nan")
    n_gt_win_hit     = int(np.sum(gt_win_hits))
    n_cand_win_hit   = int(np.sum(cand_win_hits))

    # --- Print summary ---
    print(f"Chromosomes evaluated:  {len(chroms)}")
    print(f"Ground truth peaks:     {n_gt:,}")
    print(f"Candidate peaks:        {n_cand:,}")
    print()
    print(f"--- Bin-level ({args.output_stride} bp bins) ---")
    print(f"Precision:   {bin_precision:.4f}")
    print(f"Recall:      {bin_recall:.4f}")
    print(f"F1:          {bin_f1:.4f}")
    print(f"Jaccard:     {bin_jaccard:.4f}")
    print()
    print("--- Peak-level (length-normalised) ---")
    print(f"Sensitivity (GT covered ≥1 bp):        {gt_hit_rate:.4f}  ({n_gt_hit:,}/{n_gt:,} peaks)")
    print(f"Mean GT coverage fraction:              {gt_mean_cov:.4f}")
    print(f"PPV (candidate covered ≥1 bp):         {cand_hit_rate:.4f}  ({n_cand_hit:,}/{n_cand:,} peaks)")
    print(f"Mean candidate coverage fraction:       {cand_mean_cov:.4f}")
    print()
    print(f"--- Center-window (±{args.window} bp around candidate centre) ---")
    print(f"Sensitivity (GT hit by ≥1 window):      {cw_sensitivity:.4f}  ({n_gt_win_hit:,}/{len(gt_win_hits):,} peaks)")
    print(f"Specificity (candidate window hits GT):  {cw_specificity:.4f}  ({n_cand_win_hit:,}/{len(cand_win_hits):,} peaks)")

    # --- Optional figure ---
    if args.figure is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(
            f"Candidate vs Ground Truth  |  {len(chroms)} chroms  |  "
            f"bin={args.output_stride} bp  |  window=±{args.window} bp",
            fontsize=9,
        )

        # Left: stacked bar chart of bin-level overlap composition
        ax = axes[0]
        tp_val = int(np.sum(bin_cand & bin_truth))
        fp_val = int(np.sum(bin_cand & ~bin_truth))
        fn_val = int(np.sum(~bin_cand & bin_truth))
        categories = ["Candidate bins", "Ground truth bins"]
        tp_frac_c = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
        tp_frac_g = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        ax.bar(categories, [tp_frac_c, tp_frac_g], color=["steelblue", "salmon"], alpha=0.8)
        ax.set_ylabel("Fraction overlapping the other set")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Bin-level overlap\n"
            f"Precision={bin_precision:.3f}  Recall={bin_recall:.3f}  "
            f"F1={bin_f1:.3f}  Jaccard={bin_jaccard:.3f}",
            fontsize=8,
        )
        for i, v in enumerate([tp_frac_c, tp_frac_g]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        # Right: histogram of per-peak coverage fractions
        ax = axes[1]
        bins = np.linspace(0, 1, 41)
        ax.hist(gt_fracs,   bins=bins, alpha=0.6, density=True, color="salmon",
                label=f"GT coverage (mean={gt_mean_cov:.3f})")
        ax.hist(cand_fracs, bins=bins, alpha=0.6, density=True, color="steelblue",
                label=f"Candidate coverage (mean={cand_mean_cov:.3f})")
        ax.set_xlabel("Fraction of peak covered by the other set")
        ax.set_ylabel("Density")
        ax.set_title("Per-peak length-normalised coverage", fontsize=9)
        ax.legend(fontsize=8)

        # Third panel: center-window sensitivity / specificity
        ax = axes[2]
        labels_cw = ["Sensitivity\n(GT recalled)", "Specificity\n(candidate precise)"]
        values_cw = [cw_sensitivity, cw_specificity]
        colors_cw = ["salmon", "steelblue"]
        ax.bar(labels_cw, values_cw, color=colors_cw, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title(f"Center-window (±{args.window} bp)", fontsize=9)
        for i, v in enumerate(values_cw):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"\nSaved figure to {args.figure}")


if __name__ == "__main__":
    main()
