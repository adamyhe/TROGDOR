#!/usr/bin/env python3
# logit_dist.py
# Author: Adam He <adamyhe@gmail.com>

"""Logit score distribution diagnostic for a probability bigWig.

Reads per-base predicted probabilities, bins them, converts to logits, and
produces a 2-panel figure:

  Left  — Histogram + KDE with quantile reference lines
  Right — Empirical CDF with quantile reference lines

Summary statistics are printed to stdout.

Example
-------
python scripts/benchmark/logit_dist.py \\
  -b predictions.prob.bw \\
  -o logit_dist.png \\
  --chroms chr22 \\
  -v
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybigtools
import scipy.special
import scipy.stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Logit score distribution diagnostic for a probability bigWig."
    )
    p.add_argument("-b", "--bigwig", required=True, help="Probability bigWig file")
    p.add_argument(
        "-o", "--output", required=True, help="Output image path (e.g. logit_dist.png)"
    )
    p.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Bin size in bp; probabilities are max-pooled to this resolution (default: 16)",
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all in bigWig)",
    )
    p.add_argument(
        "--max_points",
        type=int,
        default=500_000,
        help="Downsample to at most this many points for plotting (default: 500000)",
    )
    p.add_argument(
        "--keep_zeros",
        action="store_true",
        help="Include zero-probability bins (excluded by default as unscored background)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure suptitle (default: bigWig filename)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-chromosome progress"
    )
    return p.parse_args()


def bin_probs(raw, output_stride):
    n_bins = len(raw) // output_stride
    return raw[: n_bins * output_stride].reshape(n_bins, output_stride).max(axis=1)


def main():
    args = parse_args()

    prob_bw = pybigtools.open(args.bigwig)
    chrom_sizes = prob_bw.chroms()
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())

    all_probs = []

    for chrom in chroms:
        if chrom not in chrom_sizes:
            if args.verbose:
                print(f"  Skipping {chrom} (not in bigWig)", flush=True)
            continue

        chrom_len = chrom_sizes[chrom]
        if args.verbose:
            print(f"  Reading {chrom} ({chrom_len:,} bp)...", flush=True)

        raw = np.nan_to_num(
            np.array(prob_bw.values(chrom, 0, chrom_len), dtype=np.float32)
        )
        all_probs.append(bin_probs(raw, args.output_stride))

    if not all_probs:
        print(
            "No chromosomes scored. Check --chroms and bigWig contents.",
            file=sys.stderr,
        )
        sys.exit(1)

    probs = np.concatenate(all_probs)

    if not args.keep_zeros:
        probs = probs[probs > 0]
        if args.verbose:
            print(f"  {len(probs):,} bins with p > 0", flush=True)

    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = scipy.special.logit(probs)

    if len(logits) > args.max_points:
        rng = np.random.default_rng(0)
        logits = rng.choice(logits, size=args.max_points, replace=False)
        if args.verbose:
            print(f"  Downsampled to {args.max_points:,} points", flush=True)

    # ---- Summary statistics ----
    q05, q25, q50, q75, q95 = np.quantile(logits, [0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"n_bins:   {len(logits):,}")
    print(f"min:      {logits.min():.4f}")
    print(f"q05:      {q05:.4f}")
    print(f"q25:      {q25:.4f}")
    print(f"median:   {q50:.4f}")
    print(f"q75:      {q75:.4f}")
    print(f"q95:      {q95:.4f}")
    print(f"max:      {logits.max():.4f}")
    print(f"mean:     {logits.mean():.4f}")
    print(f"std:      {logits.std():.4f}")

    # ---- Figure ----
    title = args.title if args.title is not None else args.bigwig
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=10)

    quantile_vals = [q05, q25, q50, q75, q95]
    quantile_labels = ["q05", "q25", "q50", "q75", "q95"]

    # Left panel — Histogram + KDE
    ax_hist.hist(logits, bins=100, density=True, color="lightgrey")
    kde = scipy.stats.gaussian_kde(logits)
    z = np.linspace(logits.min(), logits.max(), 500)
    ax_hist.plot(z, kde(z), color="black", linewidth=1.5)
    for qv, ql in zip(quantile_vals, quantile_labels):
        ax_hist.axvline(qv, color="grey", linestyle="--", linewidth=0.8, label=ql)
    ax_hist.axvline(0, color="firebrick", linestyle="--", linewidth=0.8)
    ax_hist.set_xlabel("Logit score")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Histogram + KDE")
    ax_hist.legend(fontsize=7)

    # Right panel — Empirical CDF
    logits_sorted = np.sort(logits)
    cdf_y = np.linspace(0, 1, len(logits_sorted))
    ax_cdf.plot(logits_sorted, cdf_y, color="black", linewidth=1.2)
    for qv, ql in zip(quantile_vals, quantile_labels):
        ax_cdf.axvline(qv, color="grey", linestyle="--", linewidth=0.8, label=ql)
    ax_cdf.axvline(0, color="firebrick", linestyle="--", linewidth=0.8)
    ax_cdf.set_xlabel("Logit score")
    ax_cdf.set_ylabel("Cumulative probability")
    ax_cdf.set_title("Empirical CDF")
    ax_cdf.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
