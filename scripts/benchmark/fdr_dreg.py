#!/usr/bin/env python3
# fdr_dreg.py
# Author: Adam He <adamyhe@gmail.com>

"""Shuffle-based empirical FDR estimation for dREG BED scores.

Analogous to ``trogdor fdr`` but accepts dREG scored BED files (centre 1 bp
per 100 bp scored window) instead of a probability bigWig. Peak scoring uses
a ±dreg_window overlap test with np.searchsorted for efficiency.

Example
-------
python scripts/benchmark/fdr_dreg.py \\
  --dreg GM12878_groseq.dREG.scores.bed.gz \\
  --peaks GM12878_groseq.dREG.peaks.bed.gz \\
  --fdr_target 0.05 --n_shuffle 3 --chroms chr22 \\
  --output fdr_dreg_test.tsv --figure fdr_dreg_test.png -v
"""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../src"))

from chiaroscuro.stats import compute_fdr, shuffle_peaks


def _build_covered_intervals(dreg_data, dreg_window):
    """Build per-chromosome merged covered intervals from dREG entries.

    Each dREG entry at position ``start`` contributes the interval
    ``[start - dreg_window, start + dreg_window)``.  Overlapping/adjacent
    intervals are merged.

    Parameters
    ----------
    dreg_data : dict[str, (np.ndarray, np.ndarray)]
        Per-chromosome (entry_starts, scores) from _load_dreg.
    dreg_window : int
        Half-width of each dREG scored region.

    Returns
    -------
    dict[str, (np.ndarray, np.ndarray)]
        Maps chrom -> (ci_starts, ci_ends), both int64, sorted, non-overlapping.
    """
    covered = {}
    for chrom, (entry_starts, _) in dreg_data.items():
        raw_s = entry_starts - dreg_window
        raw_e = entry_starts + dreg_window
        # Merge overlapping intervals (entry_starts is already sorted)
        ci_starts, ci_ends = [raw_s[0]], [raw_e[0]]
        for s, e in zip(raw_s[1:], raw_e[1:]):
            if s <= ci_ends[-1]:
                ci_ends[-1] = max(ci_ends[-1], e)
            else:
                ci_starts.append(s)
                ci_ends.append(e)
        covered[chrom] = (
            np.maximum(0, np.array(ci_starts, dtype=np.int64)),
            np.array(ci_ends, dtype=np.int64),
        )
    return covered


def _shuffle_peaks_covered(peaks_df, covered_intervals, chroms, rng):
    """Shuffle peaks uniformly within dREG-covered intervals.

    Each peak is placed so that it is fully contained within a covered interval.
    Peaks wider than every covered interval on their chromosome are dropped.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Columns chrom, start, end.
    covered_intervals : dict[str, (np.ndarray, np.ndarray)]
        Per-chromosome (ci_starts, ci_ends) from _build_covered_intervals.
    chroms : list of str
        Chromosomes to process.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    pd.DataFrame
        Shuffled peaks with the same columns as peaks_df.
    """
    rows = []
    for chrom in chroms:
        if chrom not in covered_intervals:
            continue
        ci_starts, ci_ends = covered_intervals[chrom]
        sub = peaks_df[peaks_df["chrom"] == chrom].copy()
        if len(sub) == 0:
            continue
        widths = (sub["end"] - sub["start"]).values.astype(np.int64)
        new_starts = np.full(len(sub), -1, dtype=np.int64)

        # Vectorize over unique peak widths to avoid per-peak Python loops
        unique_widths, width_inverse = np.unique(widths, return_inverse=True)
        for ui, w in enumerate(unique_widths):
            # Valid start range for containment: [ci_s, ci_e - w)
            vs = ci_starts
            ve = ci_ends - w
            valid = ve > vs
            if not valid.any():
                continue
            vs, ve = vs[valid], ve[valid]
            lengths = ve - vs  # number of valid start positions per interval
            cumlen = np.cumsum(lengths)
            total = int(cumlen[-1])
            peak_mask = width_inverse == ui
            n = int(peak_mask.sum())
            r = rng.integers(0, total, size=n)
            idx = np.searchsorted(cumlen, r, side="right")
            idx = np.clip(idx, 0, len(cumlen) - 1)
            prev_cum = np.concatenate([[0], cumlen[:-1]])
            offsets = r - prev_cum[idx]
            new_starts[peak_mask] = vs[idx] + offsets

        keep = new_starts >= 0
        sub_keep = sub[keep].copy()
        sub_keep = sub_keep.reset_index(drop=True)
        sub_keep["start"] = new_starts[keep]
        sub_keep["end"] = new_starts[keep] + widths[keep]
        rows.append(sub_keep)

    if not rows:
        return peaks_df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate shuffle-based empirical FDR for dREG scored peaks."
    )
    p.add_argument("--dreg", required=True, help="dREG scored BED file (centre 1 bp per window)")
    p.add_argument("--peaks", required=True, help="Candidate peak BED file")
    p.add_argument(
        "--chrom_sizes",
        default=None,
        help="Tab-separated chrom-sizes file (optional; derived from BED if omitted)",
    )
    p.add_argument(
        "--score_col",
        type=int,
        default=3,
        help="0-based score column in dREG BED (default: 3)",
    )
    p.add_argument(
        "--dreg_window",
        type=int,
        default=50,
        help="Half-width of each dREG scored region in bp (default: 50)",
    )
    p.add_argument(
        "--stat",
        choices=["max", "mean"],
        default="max",
        help="Summary statistic per peak: max or mean (default: max)",
    )
    p.add_argument(
        "--n_shuffle",
        type=int,
        default=1,
        help="Number of independent genome shuffles (default: 1)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument(
        "--fdr_target",
        type=float,
        default=0.05,
        help="FDR target for threshold reporting (default: 0.05)",
    )
    p.add_argument(
        "--n_thresholds",
        type=int,
        default=200,
        help="Threshold grid resolution (default: 200)",
    )
    p.add_argument("--output", default=None, help="TSV output path (optional)")
    p.add_argument("--figure", default=None, help="Figure output path PNG/PDF (optional)")
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all chromosomes in dREG BED)",
    )
    p.add_argument(
        "--restrict_to_covered",
        action="store_true",
        help=(
            "Shuffle peaks only within dREG-covered intervals (union of entry "
            "positions ± dreg_window) rather than across full chromosome lengths. "
            "Ensures every shuffled peak receives a score; recommended when the "
            "dREG-covered genome is a small fraction of the total."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print per-step progress")
    return p.parse_args()


def _load_dreg(bed_path, chroms, score_col):
    """Load dREG scored BED into per-chromosome sorted arrays.

    Parameters
    ----------
    bed_path : str
        Path to dREG scored BED (optionally gzip-compressed).
    chroms : list of str or None
        Chromosome whitelist; if None, load all.
    score_col : int
        0-based column index for scores.

    Returns
    -------
    dict[str, (np.ndarray, np.ndarray)]
        Maps chrom -> (entry_starts, scores), both sorted by entry_starts.
    """
    usecols = sorted({0, 1, score_col})
    col_names = {0: "chrom", 1: "start", score_col: "score"}
    # If score_col < 2 use positional names; expand to include end col
    usecols = [0, 1, score_col] if score_col != 1 else [0, 1]
    usecols = sorted(set([0, 1, score_col]))

    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=usecols,
        compression="infer",
    )
    # Rename columns by position index
    rename = {0: "chrom", 1: "start", score_col: "score"}
    df = df.rename(columns=rename)

    if chroms is not None:
        df = df[df["chrom"].isin(chroms)]

    dreg_data = {}
    for chrom, grp in df.groupby("chrom", sort=False):
        grp_sorted = grp.sort_values("start")
        dreg_data[chrom] = (
            grp_sorted["start"].to_numpy(dtype=np.int64),
            grp_sorted["score"].to_numpy(dtype=np.float32),
        )
    return dreg_data


def _score_peaks_dreg(dreg_data, peaks_df, chroms, dreg_window, stat):
    """Score candidate peaks using overlapping dREG entries.

    Each peak [p_start, p_end) is scored by finding dREG entries whose
    ±dreg_window expanded window overlaps the peak:

        overlap condition: entry_start ∈ [p_start − dreg_window + 1, p_end + dreg_window)

    Score is max or mean of overlapping entry scores. NaN if no entries overlap.

    Parameters
    ----------
    dreg_data : dict[str, (np.ndarray, np.ndarray)]
        Per-chromosome (entry_starts, scores) from _load_dreg.
    peaks_df : pd.DataFrame
        Columns chrom, start, end.
    chroms : list of str
        Chromosomes to process.
    dreg_window : int
        Half-width of each dREG scored region.
    stat : {"max", "mean"}
        Summary statistic per peak.

    Returns
    -------
    np.ndarray of float32, shape (len(peaks_df),)
    """
    scores = np.full(len(peaks_df), np.nan, dtype=np.float32)

    for chrom in chroms:
        if chrom not in dreg_data:
            continue
        entry_starts, entry_scores = dreg_data[chrom]
        mask = peaks_df["chrom"] == chrom
        if not mask.any():
            continue

        chrom_peaks = peaks_df[mask]
        p_starts = chrom_peaks["start"].to_numpy(dtype=np.int64)
        p_ends = chrom_peaks["end"].to_numpy(dtype=np.int64)
        indices = chrom_peaks.index.to_numpy()

        # Overlap condition: entry_start ∈ [p_start - dreg_window + 1, p_end + dreg_window)
        lo_bound = p_starts - dreg_window + 1
        hi_bound = p_ends + dreg_window

        lo = np.searchsorted(entry_starts, lo_bound, side="left")
        hi = np.searchsorted(entry_starts, hi_bound, side="left")

        for i, (li, hi_i, idx) in enumerate(zip(lo, hi, indices)):
            if li >= hi_i:
                continue
            vals = entry_scores[li:hi_i]
            scores[idx] = vals.max() if stat == "max" else vals.mean()

    return scores


def main():
    args = parse_args()

    # Load dREG BED
    if args.verbose:
        print(f"Loading dREG scores from {args.dreg}...", flush=True)
    # First pass: determine available chroms
    dreg_data = _load_dreg(args.dreg, args.chroms, args.score_col)

    if len(dreg_data) == 0:
        print("No dREG entries found. Check --chroms and BED contents.", file=sys.stderr)
        sys.exit(1)

    chroms = args.chroms if args.chroms is not None else sorted(dreg_data.keys())
    chroms = [c for c in chroms if c in dreg_data]

    if args.verbose:
        n_entries = sum(len(v[0]) for v in dreg_data.values())
        print(f"  Loaded {n_entries:,} dREG entries across {len(dreg_data)} chromosomes.", flush=True)

    # Load peaks BED
    if args.verbose:
        print(f"Loading peaks from {args.peaks}...", flush=True)
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
        print("No peaks found on the requested chromosomes.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"  {len(peaks_df):,} peaks loaded.", flush=True)

    # Derive or load chrom sizes
    if args.chrom_sizes is not None:
        cs_df = pd.read_csv(args.chrom_sizes, sep="\t", header=None, names=["chrom", "size"])
        chrom_sizes = dict(zip(cs_df["chrom"], cs_df["size"].astype(int)))
    else:
        # Derive from dREG BED: max(end) + 2 * dreg_window per chromosome
        if args.verbose:
            print("Deriving chrom sizes from dREG BED...", flush=True)
        # Load end column too for size estimation
        raw_df = pd.read_csv(
            args.dreg,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
            compression="infer",
        )
        if args.chroms is not None:
            raw_df = raw_df[raw_df["chrom"].isin(chroms)]
        chrom_sizes = (
            raw_df.groupby("chrom")["end"].max() + 2 * args.dreg_window
        ).astype(int).to_dict()

    # Score real peaks
    if args.verbose:
        print(f"Scoring {len(peaks_df):,} real peaks...", flush=True)
    real_scores = _score_peaks_dreg(dreg_data, peaks_df, chroms, args.dreg_window, args.stat)
    real_scores = real_scores[~np.isnan(real_scores)]

    if len(real_scores) == 0:
        print("All real peak scores are NaN; check dREG BED coverage.", file=sys.stderr)
        sys.exit(1)

    # Build covered intervals for restricted shuffling if requested
    covered_intervals = None
    if args.restrict_to_covered:
        covered_intervals = _build_covered_intervals(dreg_data, args.dreg_window)
        if args.verbose:
            total_covered = sum(
                int((e - s).sum()) for s, e in covered_intervals.values()
            )
            print(f"  Covered genome: {total_covered:,} bp across {len(covered_intervals)} chromosomes.", flush=True)

    # Score shuffled null sets
    rng = np.random.default_rng(args.seed)
    null_score_lists = []
    for i in range(args.n_shuffle):
        if args.verbose:
            print(f"  Shuffle {i + 1}/{args.n_shuffle}...", flush=True)
        if covered_intervals is not None:
            null_df = _shuffle_peaks_covered(peaks_df, covered_intervals, chroms, rng)
        else:
            null_df = shuffle_peaks(peaks_df, chrom_sizes, chroms, rng)
        s = _score_peaks_dreg(dreg_data, null_df, chroms, args.dreg_window, args.stat)
        null_score_lists.append(s[~np.isnan(s)])

    null_scores = np.concatenate(null_score_lists) if null_score_lists else np.empty(0, dtype=np.float32)

    # Compute FDR curve
    thresholds, n_real, n_null, fdr = compute_fdr(
        real_scores, null_scores, args.n_shuffle, args.n_thresholds
    )

    # Find threshold at FDR target
    passing = np.where(fdr <= args.fdr_target)[0]
    if len(passing) == 0:
        threshold_at_target = float("nan")
        n_peaks_at_target = 0
    else:
        first_pass = passing[0]
        threshold_at_target = float(thresholds[first_pass])
        n_peaks_at_target = int(n_real[first_pass])

    # Print summary
    print(f"Real peaks:       {len(real_scores):,}")
    print(f"Null peaks:       {len(null_scores):,} ({args.n_shuffle} shuffle(s))")
    print(f"Score stat:       {args.stat}")
    print(f"FDR target:       {args.fdr_target:.3f}")
    if np.isnan(threshold_at_target):
        print(f"Score threshold:  N/A (FDR {args.fdr_target} never reached)")
    else:
        print(f"Score threshold:  {threshold_at_target:.6f}")
        print(f"Peaks at target:  {n_peaks_at_target:,}")

    # Optional TSV output
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

    # Optional figure
    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(args.dreg, fontsize=9)

        # Left: score distributions
        ax = axes[0]
        bins = np.linspace(thresholds[0], thresholds[-1], 60)
        ax.hist(
            real_scores,
            bins=bins,
            density=True,
            alpha=0.6,
            color="steelblue",
            label="real",
        )
        if len(null_scores) > 0:
            ax.hist(
                null_scores,
                bins=bins,
                density=True,
                alpha=0.5,
                color="salmon",
                label="null",
            )
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

        recall = n_real / len(real_scores)

        # Right: FDR curve overlaid with recall
        ax = axes[1]
        ax.plot(thresholds, fdr, color="black", linewidth=1.5, label="FDR")
        ax.axhline(
            args.fdr_target,
            color="firebrick",
            linestyle="--",
            linewidth=0.8,
            label=f"FDR={args.fdr_target}",
        )
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
        ax.set_title("FDR and recall vs threshold")
        ax.set_ylim(0, 1.05)

        ax2 = ax.twinx()
        ax2.plot(thresholds, recall, color="steelblue", linewidth=1.5, linestyle="-", label="Recall")
        ax2.set_ylabel("Recall (fraction of real peaks)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(0, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        plt.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
