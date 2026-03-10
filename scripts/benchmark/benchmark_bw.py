#!/usr/bin/env python3
# benchmark_bw.py
# Author: Adam He <adamyhe@gmail.com>

"""Genome-wide benchmarking from a pre-computed probability bigWig.

Reads per-base (or pre-binned) predicted probabilities from a bigWig file,
bins them at --output_stride resolution by taking the per-bin maximum, converts
ground-truth peak BED regions to the same per-bin labels, and reports
AUROC/AUPRC.

Example
-------
python scripts/benchmark/benchmark_bw.py \
  -b predictions.bw \
  -t data/K562.positive.bed.gz \
  --chroms chr1 chr2 \
  -v
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pybigtools
import torch
from torcheval.metrics.functional import binary_auprc, binary_auroc

from chiaroscuro.utils import encode_labels


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark a pre-computed probability bigWig against ground-truth peaks."
    )
    p.add_argument("-b", "--bigwig", required=True, help="Probability bigWig file")
    p.add_argument("-t", "--peaks", required=True, help="Ground-truth peak BED file")
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
        "-v", "--verbose", action="store_true", help="Print per-chromosome progress"
    )
    return p.parse_args()


def bin_probs(raw, output_stride):
    """Max-pool raw per-base probabilities into output_stride-sized bins.

    Positions beyond the last complete bin are discarded, matching the label
    encoding in encode_labels which also uses chrom_len // output_stride bins.
    """
    n_bins = len(raw) // output_stride
    return raw[: n_bins * output_stride].reshape(n_bins, output_stride).max(axis=1)


def main():
    args = parse_args()

    prob_bw = pybigtools.open(args.bigwig)
    chrom_sizes = prob_bw.chroms()
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())

    peaks_df = pd.read_csv(
        args.peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
    )

    all_probs = []
    all_labels = []

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

        probs = bin_probs(raw, args.output_stride)
        labels = encode_labels(peaks_df, chrom, chrom_len, args.output_stride)

        all_probs.append(probs)
        all_labels.append(labels)

    if not all_probs:
        print(
            "No chromosomes scored. Check --chroms and bigWig contents.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    probs_t = torch.tensor(all_probs)
    labels_t = torch.tensor(all_labels).long()

    auroc = binary_auroc(probs_t, labels_t).item()
    auprc = binary_auprc(probs_t, labels_t).item()

    n_total = len(all_labels)
    n_pos = int(all_labels.sum())
    pct = 100.0 * n_pos / n_total

    print(f"Bins:  {n_total:,} total, {n_pos:,} positive ({pct:.3f}%)")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")


if __name__ == "__main__":
    main()
