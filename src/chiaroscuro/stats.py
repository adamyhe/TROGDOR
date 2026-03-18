# stats.py
# Author: Adam He <adamyhe@gmail.com>

"""
Statistical helpers for empirical FDR estimation from bigWig probability
tracks and candidate peak BED files.
"""

import numpy as np
import pandas as pd


def score_peaks(bw, peaks_df, chrom_sizes, stat, chroms, verbose):
    """Return an array of per-peak summary scores from the bigWig.

    Peaks not present in the bigWig or with zero length are assigned NaN.

    Parameters
    ----------
    bw : pybigtools file handle
        Open probability bigWig.
    peaks_df : pd.DataFrame
        Columns ``chrom``, ``start``, ``end``.
    chrom_sizes : dict
        Mapping of chromosome name to length.
    stat : {"max", "mean"}
        Summary statistic to apply over each peak interval.
    chroms : list of str
        Chromosomes to process.
    verbose : bool
        Whether to print per-chromosome progress.

    Returns
    -------
    np.ndarray of float32, shape (len(peaks_df),)
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

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Columns ``chrom``, ``start``, ``end``.
    chrom_sizes : dict
        Mapping of chromosome name to length.
    chroms : list of str
        Chromosomes to process.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    pd.DataFrame
        Shuffled peaks with the same columns as ``peaks_df``.
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


def compute_fdr(real_scores, null_scores, n_shuffle, n_thresholds):
    """Compute an empirical FDR curve from real and null peak scores.

    Parameters
    ----------
    real_scores : np.ndarray
        Scores for real peaks (NaN already removed).
    null_scores : np.ndarray
        Concatenated scores from all shuffles (NaN already removed).
    n_shuffle : int
        Number of shuffles used to produce ``null_scores``; used to average
        the null count.
    n_thresholds : int
        Number of evenly-spaced thresholds to evaluate.

    Returns
    -------
    thresholds : np.ndarray, shape (n_thresholds,)
    n_real : np.ndarray, shape (n_thresholds,)
    n_null : np.ndarray, shape (n_thresholds,)
        Average null count per shuffle.
    fdr : np.ndarray, shape (n_thresholds,)
        Estimated FDR at each threshold, clipped to [0, 1].
    """
    t_min = (
        min(real_scores.min(), null_scores.min())
        if len(null_scores)
        else real_scores.min()
    )
    t_max = (
        max(real_scores.max(), null_scores.max())
        if len(null_scores)
        else real_scores.max()
    )
    thresholds = np.linspace(t_min, t_max, n_thresholds)

    n_real = np.array([(real_scores >= t).sum() for t in thresholds], dtype=float)
    if len(null_scores) > 0:
        n_null_total = np.array(
            [(null_scores >= t).sum() for t in thresholds], dtype=float
        )
        n_null = n_null_total / n_shuffle
    else:
        n_null = np.zeros(len(thresholds))

    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(n_real > 0, np.minimum(1.0, n_null / n_real), 1.0)

    return thresholds, n_real, n_null, fdr
