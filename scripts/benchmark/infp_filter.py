#!/usr/bin/env python3
# infp_filter.py
# Author: Adam He <adamyhe@gmail.com>

"""Apply a dREG-style informative-positions filter to a TROGDOR probability bigWig.

A genomic position is "informative" if it meets either heuristic (dREG paper):
  1. >3 reads in a 100 bp window on either strand
  2. >1 read in a 1 kbp window on both strands

Per-base coverage is loaded from stranded bigWigs and rolling sums are computed
with the O(n) cumsum trick. The resulting boolean mask is used to:

  --output_bw   Re-emit only prob.bw bins that overlap ≥1 informative position
                (drop background bins that inflated the FDR null distribution).
  --output_bed  Write merged informative intervals as a BED file.

At least one of --output_bw / --output_bed must be specified.

Example
-------
python scripts/benchmark/infp_filter.py \\
    -p GM12878_groseq.plus.bw \\
    -m GM12878_groseq.minus.bw \\
    -i GM12878_groseq.prob.bw \\
    --output_bw GM12878_groseq.infp.prob.bw \\
    --output_bed GM12878_groseq.infp.bed \\
    --chroms chr1 chr2 -v
"""

import argparse
import sys

import numpy as np
import pybigtools


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply dREG-style informative-positions filter to a TROGDOR prob.bw."
    )
    p.add_argument("-p", "--plus", required=True, help="Plus-strand coverage bigWig")
    p.add_argument("-m", "--minus", required=True, help="Minus-strand coverage bigWig")
    p.add_argument(
        "-i", "--input_bw", default=None, help="TROGDOR prob.bw to mask (required with --output_bw)"
    )
    p.add_argument("--output_bw", default=None, help="Masked output prob.bw")
    p.add_argument("--output_bed", default=None, help="Informative regions BED")
    p.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Bin size in bp for prob.bw masking (default: 16)",
    )
    p.add_argument(
        "--window1",
        type=int,
        default=96,
        help="Window size in bp for heuristic 1; should be a multiple of --output_stride (default: 96)",
    )
    p.add_argument(
        "--thresh1",
        type=int,
        default=3,
        help="Read count threshold for heuristic 1 (default: 3)",
    )
    p.add_argument(
        "--window2",
        type=int,
        default=992,
        help="Window size in bp for heuristic 2; should be a multiple of --output_stride (default: 992)",
    )
    p.add_argument(
        "--thresh2",
        type=int,
        default=1,
        help="Read count threshold for heuristic 2 (default: 1)",
    )
    p.add_argument(
        "--chroms", nargs="+", default=None, help="Chromosome whitelist (default: all)"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print per-chromosome progress")
    args = p.parse_args()

    if args.output_bw is None and args.output_bed is None:
        p.error("At least one of --output_bw or --output_bed must be specified.")
    if args.output_bw is not None and args.input_bw is None:
        p.error("--input_bw is required when --output_bw is specified.")

    return args


def _informative_mask(plus_bins, minus_bins, w1_bins, t1, w2_bins, t2):
    """Compute per-bin informative-positions boolean mask.

    A bin j is informative if:
      h1: sum(plus_bins[j-w1//2 : j+w1//2]) > t1  OR  same for minus
      h2: sum(plus_bins[j-w2//2 : j+w2//2]) > t2  AND same for minus

    Rolling sums use the O(n) cumsum trick on bin-level arrays (already downsampled
    from per-base coverage by the caller), which is ~output_stride× faster than
    operating at per-base resolution.

    Parameters
    ----------
    plus_bins, minus_bins : np.ndarray, float64
        Per-bin summed coverage arrays, shape (n_bins,).
    w1_bins, w2_bins : int
        Rolling-window sizes in bins.
    t1, t2 : numeric
        Read-count thresholds for heuristics 1 and 2.

    Returns
    -------
    np.ndarray, bool, shape (n_bins,)
    """
    n = len(plus_bins)
    mask = np.zeros(n, dtype=bool)

    def _roll(arr, w):
        cs = np.empty(len(arr) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(arr, out=cs[1:])
        return cs[w:] - cs[:-w]   # rolling[i] = sum(arr[i : i+w])

    half1 = w1_bins // 2
    if n >= w1_bins:
        h1 = (_roll(plus_bins, w1_bins) > t1) | (_roll(minus_bins, w1_bins) > t1)
        mask[half1 : half1 + len(h1)] |= h1

    half2 = w2_bins // 2
    if n >= w2_bins:
        h2 = (_roll(plus_bins, w2_bins) > t2) & (_roll(minus_bins, w2_bins) > t2)
        mask[half2 : half2 + len(h2)] |= h2

    return mask


def _mask_to_intervals(mask):
    """Convert a boolean array to a list of (start, end) intervals (0-based, half-open).

    Parameters
    ----------
    mask : np.ndarray, bool

    Returns
    -------
    list of (int, int)
    """
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.view(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    args = parse_args()

    pl_bw = pybigtools.open(args.plus)
    mn_bw = pybigtools.open(args.minus)
    pl_sizes = dict(pl_bw.chroms())
    mn_sizes = dict(mn_bw.chroms())
    chrom_sizes = {c: s for c, s in pl_sizes.items() if c in mn_sizes}

    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())
    chroms = [c for c in chroms if c in chrom_sizes]

    if not chroms:
        print("No chromosomes to process.", file=sys.stderr)
        sys.exit(1)

    in_bw = pybigtools.open(args.input_bw) if args.input_bw else None

    bed_out = None
    if args.output_bed:
        bed_out = open(args.output_bed, "w")

    all_intervals = []   # (chrom, start, end, value) for masked bigWig
    total_bp = 0
    total_informative = 0
    total_bins_in = 0
    total_bins_out = 0

    for chrom in chroms:
        chrom_len = chrom_sizes[chrom]
        if args.verbose:
            print(f"  {chrom} ({chrom_len:,} bp)...", flush=True)

        plus_arr = np.abs(np.nan_to_num(
            np.array(pl_bw.values(chrom, 0, chrom_len), dtype=np.float32)
        ))
        minus_arr = np.abs(np.nan_to_num(
            np.array(mn_bw.values(chrom, 0, chrom_len), dtype=np.float32)
        ))

        # Downsample to bin resolution before rolling sums (~output_stride× faster)
        n_bins = chrom_len // args.output_stride
        plus_bins = (
            plus_arr[: n_bins * args.output_stride]
            .reshape(n_bins, args.output_stride)
            .sum(axis=1)
            .astype(np.float64)
        )
        minus_bins = (
            minus_arr[: n_bins * args.output_stride]
            .reshape(n_bins, args.output_stride)
            .sum(axis=1)
            .astype(np.float64)
        )

        w1_bins = max(1, args.window1 // args.output_stride)
        w2_bins = max(1, args.window2 // args.output_stride)

        bin_inf_mask = _informative_mask(
            plus_bins, minus_bins, w1_bins, args.thresh1, w2_bins, args.thresh2
        )

        total_bp += chrom_len
        total_informative += int(bin_inf_mask.sum()) * args.output_stride

        if args.output_bw and in_bw is not None:
            recs = list(in_bw.records(chrom, 0, chrom_len))
            if recs:
                r_starts = np.array([r[0] for r in recs], dtype=np.int64)
                r_ends   = np.array([r[1] for r in recs], dtype=np.int64)
                r_vals   = np.array([r[2] for r in recs], dtype=np.float32)
                finite = ~np.isnan(r_vals)
                bin_idx = (r_starts[finite] // args.output_stride).clip(0, n_bins - 1)
                keep = bin_inf_mask[bin_idx]
                for s, e, v in zip(
                    r_starts[finite][keep].tolist(),
                    r_ends[finite][keep].tolist(),
                    r_vals[finite][keep].tolist(),
                ):
                    all_intervals.append((chrom, s, e, v))
                total_bins_in  += int(finite.sum())
                total_bins_out += int(keep.sum())

        if bed_out is not None:
            # Convert bin mask back to bp coordinates for BED output
            for bin_start, bin_end in _mask_to_intervals(bin_inf_mask):
                bed_out.write(
                    f"{chrom}\t{bin_start * args.output_stride}\t{bin_end * args.output_stride}\n"
                )

    pl_bw.close()
    mn_bw.close()
    if in_bw is not None:
        in_bw.close()
    if bed_out is not None:
        bed_out.close()

    eff_w1 = max(1, args.window1 // args.output_stride) * args.output_stride
    eff_w2 = max(1, args.window2 // args.output_stride) * args.output_stride
    print(
        f"Effective windows: {eff_w1} bp (h1, thresh>{args.thresh1}), "
        f"{eff_w2} bp (h2, thresh>{args.thresh2}) at {args.output_stride} bp bins"
    )
    pct = 100.0 * total_informative / total_bp if total_bp > 0 else 0.0
    print(f"Informative positions: {total_informative:,} / {total_bp:,} bp ({pct:.2f}%)")

    if args.output_bw:
        bins_dropped = total_bins_in - total_bins_out
        pct_kept = 100.0 * total_bins_out / total_bins_in if total_bins_in > 0 else 0.0
        print(
            f"Prob.bw bins: {total_bins_in:,} in → {total_bins_out:,} kept, "
            f"{bins_dropped:,} dropped ({pct_kept:.1f}% retained)"
        )
        out_bw = pybigtools.open(args.output_bw, "w")
        out_bw.write(
            {c: chrom_sizes[c] for c in chroms},
            iter(all_intervals),
        )
        print(f"Wrote masked bigWig to {args.output_bw}")

    if args.output_bed:
        print(f"Wrote informative regions BED to {args.output_bed}")


if __name__ == "__main__":
    main()
