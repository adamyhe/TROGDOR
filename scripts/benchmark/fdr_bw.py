#!/usr/bin/env python3
# fdr_bw.py
# Author: Adam He <adamyhe@gmail.com>

"""
fdr_bigwig.py — Estimate empirical FDR for candidate peaks scored by a
bigWig probability track, without assuming the scores are calibrated p-values.

Strategy:
    1. Summarise bigWig scores over candidate peaks (e.g. mean, max)
    2. Generate a null peak set by shuffling candidate peaks
    3. Summarise bigWig scores over null peaks the same way
    4. Estimate FDR(t) = min(1, N_null(t) / N_real(t)) at each threshold

The score threshold corresponding to a target FDR (default 0.05) is printed to
stdout along with summary statistics.  Pass --output to save a TSV table and
--figure to save an FDR-vs-threshold plot.

Example
-------
python scripts/benchmark/fdr_bw.py \\
  -b predictions.prob.bw \\
  -t peaks.bed.gz \\
  --fdr_target 0.05 \\
  --output fdr_table.tsv \\
  --figure fdr_curve.png \\
  --chroms chr1 chr2 \\
  -v
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pybigtools


def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate empirical FDR from a probability bigWig and a candidate peak BED."
    )
    p.add_argument("-b", "--bigwig", required=True, help="Probability bigWig file")
    p.add_argument(
        "-t", "--peaks", required=True, help="Candidate peak BED file (optionally .gz)"
    )
    p.add_argument(
        "--stat",
        choices=["max", "mean"],
        default="max",
        help="Summary statistic per peak (default: max)",
    )
    p.add_argument(
        "--n_shuffle",
        type=int,
        default=1,
        help="Number of independent genome shuffles to average over (default: 1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0)",
    )
    p.add_argument(
        "--fdr_target",
        type=float,
        default=0.05,
        help="FDR target for reporting the score threshold (default: 0.05)",
    )
    p.add_argument(
        "--n_thresholds",
        type=int,
        default=200,
        help="Number of evenly-spaced thresholds to evaluate (default: 200)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write TSV table of threshold/FDR/N_real/N_null to this path",
    )
    p.add_argument(
        "--figure",
        default=None,
        help="Save FDR-vs-threshold plot to this path (e.g. fdr_curve.png)",
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all in bigWig)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-chromosome progress"
    )
    return p.parse_args()


def score_peaks(bw, peaks_df, chrom_sizes, stat, chroms, verbose):
    """Return an array of per-peak summary scores from the bigWig.

    Peaks not present in the bigWig or with zero length are assigned NaN.
    """
    scores = np.full(len(peaks_df), np.nan, dtype=np.float32)
    for chrom in chroms:
        if chrom not in chrom_sizes:
            continue
        chrom_len = chrom_sizes[chrom]
        mask = peaks_df["chrom"] == chrom
        if not mask.any():
            continue
        if verbose:
            n = mask.sum()
            print(f"  Scoring {n} peaks on {chrom}...", flush=True)
        for idx, row in peaks_df[mask].iterrows():
            start = int(row["start"])
            end = min(int(row["end"]), chrom_len)
            if start >= end:
                continue
            vals = np.nan_to_num(
                np.array(bw.values(chrom, start, end), dtype=np.float32)
            )
            if len(vals) == 0:
                continue
            scores[idx] = vals.max() if stat == "max" else vals.mean()
    return scores


def shuffle_peaks(peaks_df, chrom_sizes, chroms, rng):
    """Shuffle peak start positions uniformly within chromosome bounds.

    Width is preserved; peaks that cannot fit are dropped.
    """
    rows = []
    for chrom in chroms:
        if chrom not in chrom_sizes:
            continue
        chrom_len = chrom_sizes[chrom]
        sub = peaks_df[peaks_df["chrom"] == chrom].copy()
        widths = (sub["end"] - sub["start"]).values.astype(int)
        max_starts = chrom_len - widths
        keep = max_starts > 0
        if not keep.any():
            continue
        sub = sub[keep].copy()
        widths = widths[keep]
        max_starts = max_starts[keep]
        new_starts = rng.integers(0, max_starts, endpoint=False)
        sub["start"] = new_starts
        sub["end"] = new_starts + widths
        rows.append(sub)
    if not rows:
        return peaks_df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def main():
    args = parse_args()

    bw = pybigtools.open(args.bigwig)
    chrom_sizes = dict(bw.chroms())
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())

    peaks_df = pd.read_csv(
        args.peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
    )
    peaks_df = peaks_df[peaks_df["chrom"].isin(chroms)].reset_index(drop=True)

    if len(peaks_df) == 0:
        print(
            "No peaks found on the requested chromosomes.", file=sys.stderr
        )
        sys.exit(1)

    if args.verbose:
        print(
            f"Scoring {len(peaks_df):,} real peaks from {args.bigwig}...",
            flush=True,
        )
    real_scores = score_peaks(bw, peaks_df, chrom_sizes, args.stat, chroms, args.verbose)
    real_scores = real_scores[~np.isnan(real_scores)]

    if len(real_scores) == 0:
        print("All real peak scores are NaN; check bigWig coverage.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    null_score_lists = []
    for i in range(args.n_shuffle):
        if args.verbose:
            print(f"  Shuffle {i + 1}/{args.n_shuffle}...", flush=True)
        null_df = shuffle_peaks(peaks_df, chrom_sizes, chroms, rng)
        s = score_peaks(bw, null_df, chrom_sizes, args.stat, chroms, False)
        null_score_lists.append(s[~np.isnan(s)])
    bw.close()

    null_scores = np.concatenate(null_score_lists)

    # ---- FDR curve ----
    t_min = min(real_scores.min(), null_scores.min()) if len(null_scores) else real_scores.min()
    t_max = max(real_scores.max(), null_scores.max()) if len(null_scores) else real_scores.max()
    thresholds = np.linspace(t_min, t_max, args.n_thresholds)

    n_real = np.array([(real_scores >= t).sum() for t in thresholds], dtype=float)
    if len(null_scores) > 0:
        # Average over shuffles: total null hits / n_shuffle
        n_null_total = np.array([(null_scores >= t).sum() for t in thresholds], dtype=float)
        n_null = n_null_total / args.n_shuffle
    else:
        n_null = np.zeros(len(thresholds))

    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(n_real > 0, np.minimum(1.0, n_null / n_real), 1.0)

    # ---- Find threshold at FDR target ----
    passing = np.where(fdr <= args.fdr_target)[0]
    if len(passing) == 0:
        threshold_at_target = float("nan")
        n_peaks_at_target = 0
    else:
        first_pass = passing[0]
        threshold_at_target = float(thresholds[first_pass])
        n_peaks_at_target = int(n_real[first_pass])

    # ---- Print summary ----
    print(f"Real peaks:       {len(real_scores):,}")
    print(f"Null peaks:       {len(null_scores):,} ({args.n_shuffle} shuffle(s))")
    print(f"Score stat:       {args.stat}")
    print(f"FDR target:       {args.fdr_target:.3f}")
    if np.isnan(threshold_at_target):
        print(f"Score threshold:  N/A (FDR {args.fdr_target} never reached)")
    else:
        print(f"Score threshold:  {threshold_at_target:.6f}")
        print(f"Peaks at target:  {n_peaks_at_target:,}")

    # ---- Optional TSV output ----
    if args.output is not None:
        table = pd.DataFrame(
            {
                "threshold": thresholds,
                "n_real": n_real.astype(int),
                "n_null": n_null,
                "fdr": fdr,
            }
        )
        table.to_csv(args.output, sep="\t", index=False, float_format="%.6g")
        print(f"Saved FDR table to {args.output}")

    # ---- Optional figure ----
    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(args.bigwig, fontsize=9)

        # Left — score distributions
        ax = axes[0]
        bins = np.linspace(t_min, t_max, 60)
        ax.hist(real_scores, bins=bins, density=True, alpha=0.6, color="steelblue", label="real")
        if len(null_scores) > 0:
            ax.hist(null_scores, bins=bins, density=True, alpha=0.5, color="salmon", label="null")
        if not np.isnan(threshold_at_target):
            ax.axvline(
                threshold_at_target,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"t={threshold_at_target:.3f} (FDR={args.fdr_target})",
            )
        ax.set_xlabel(f"Peak score ({args.stat})")
        ax.set_ylabel("Density")
        ax.set_title("Score distributions")
        ax.legend(fontsize=8)

        # Right — FDR curve
        ax = axes[1]
        ax.plot(thresholds, fdr, color="black", linewidth=1.5)
        ax.axhline(args.fdr_target, color="firebrick", linestyle="--", linewidth=0.8, label=f"FDR={args.fdr_target}")
        if not np.isnan(threshold_at_target):
            ax.axvline(
                threshold_at_target,
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label=f"t={threshold_at_target:.3f}",
            )
        ax.set_xlabel(f"Score threshold ({args.stat})")
        ax.set_ylabel("Empirical FDR")
        ax.set_title("FDR vs threshold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
