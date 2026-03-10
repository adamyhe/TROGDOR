#!/usr/bin/env python3
# frip.py
# Author: Adam He <adamyhe@gmail.com>

"""Calculate FRIP (Fraction of Reads In Peaks) from stranded nascent RNA bigWigs.

Two variants are reported:
  Raw FRIP:        signal_in_peaks / total_signal
  Normalized FRIP: raw_frip / (peak_length / effective_genome_length)
               — corrects for peak set size; equivalent to fold-enrichment
                 over uniform expectation.

Example
-------
python scripts/benchmark/frip.py \
  -p plus.bw -m minus.bw -t peaks.bed.gz --chroms chr1 -v
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pybigtools


def parse_args():
    p = argparse.ArgumentParser(
        description="Calculate FRIP from stranded nascent RNA bigWigs and a peak BED file."
    )
    p.add_argument(
        "-p", "--pl_bigwig", required=True, help="Plus-strand coverage bigWig"
    )
    p.add_argument(
        "-m", "--mn_bigwig", required=True, help="Minus-strand coverage bigWig"
    )
    p.add_argument(
        "-t", "--peaks", required=True, help="Peak BED file (optionally .gz)"
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


def main():
    args = parse_args()

    pl_bw = pybigtools.open(args.pl_bigwig)
    mn_bw = pybigtools.open(args.mn_bigwig)
    chrom_sizes = pl_bw.chroms()
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())

    peaks_df = pd.read_csv(
        args.peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
    )

    total_signal = 0.0
    peak_signal = 0.0
    peak_length = 0
    effective_genome_length = 0

    for chrom in chroms:
        if chrom not in chrom_sizes:
            if args.verbose:
                print(f"  Skipping {chrom} (not in bigWig)", flush=True)
            continue

        chrom_len = chrom_sizes[chrom]
        if args.verbose:
            print(f"  Reading {chrom} ({chrom_len:,} bp)...", flush=True)

        pl = np.abs(
            np.nan_to_num(np.array(pl_bw.values(chrom, 0, chrom_len), dtype=np.float32))
        )
        mn = np.abs(
            np.nan_to_num(np.array(mn_bw.values(chrom, 0, chrom_len), dtype=np.float32))
        )
        coverage = pl + mn

        total_signal += coverage.sum()
        effective_genome_length += chrom_len

        chrom_peaks = peaks_df[peaks_df["chrom"] == chrom]
        for _, row in chrom_peaks.iterrows():
            start = int(row["start"])
            end = min(int(row["end"]), chrom_len)
            if start >= end:
                continue
            peak_signal += coverage[start:end].sum()
            peak_length += end - start

    if effective_genome_length == 0:
        print(
            "No chromosomes processed. Check --chroms and bigWig contents.",
            file=sys.stderr,
        )
        sys.exit(1)

    if total_signal == 0:
        print("Total signal is zero; cannot compute FRIP.", file=sys.stderr)
        sys.exit(1)

    raw_frip = peak_signal / total_signal
    peak_pct = 100.0 * peak_length / effective_genome_length
    normalized_frip = (
        raw_frip / (peak_length / effective_genome_length)
        if peak_length > 0
        else float("nan")
    )

    print(f"Peak length:      {peak_length:,} bp ({peak_pct:.3f}% of genome)")
    print(f"Total signal:     {total_signal:.3e}")
    print(f"Peak signal:      {peak_signal:.3e}")
    print(f"Raw FRIP:         {raw_frip:.4f}")
    print(f"Normalized FRIP:  {normalized_frip:.4f}")


if __name__ == "__main__":
    main()
