# stats.py
# Author: Adam He <adamyhe@gmail.com>

"""
Statistical helpers for empirical FDR estimation from bigWig probability
tracks and candidate peak BED files.
"""

import numpy as np
import pandas as pd


def read_bed3(path):
    """Read the first three columns of a BED-like file."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
        dtype={"chrom": str, "start": int, "end": int},
    )


def merge_intervals_df(intervals_df, chroms=None):
    """Merge BED intervals per chromosome.

    Parameters
    ----------
    intervals_df : pd.DataFrame
        Columns ``chrom``, ``start``, ``end``.
    chroms : list of str or None
        Optional chromosome order/subset.

    Returns
    -------
    pd.DataFrame
        Merged intervals with columns ``chrom``, ``start``, ``end``.
    """
    rows = []
    if chroms is None:
        chroms = sorted(intervals_df["chrom"].unique())
    for chrom in chroms:
        sub = intervals_df[intervals_df["chrom"] == chrom].sort_values(
            ["start", "end"]
        )
        if len(sub) == 0:
            continue
        cur_start = int(sub.iloc[0]["start"])
        cur_end = int(sub.iloc[0]["end"])
        for _, row in sub.iloc[1:].iterrows():
            start = int(row["start"])
            end = int(row["end"])
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                rows.append((chrom, cur_start, cur_end))
                cur_start, cur_end = start, end
        rows.append((chrom, cur_start, cur_end))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def _remerge_sorted_intervals(starts, ends):
    """Merge possibly-overlapping interval arrays into disjoint, ascending
    ``(starts, ends)`` int64 arrays. Used after margin expansion, which can
    make previously-separate intervals overlap or touch."""
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    ends = ends[order]
    merged_starts = [int(starts[0])]
    merged_ends = [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        s, e = int(s), int(e)
        if s <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], e)
        else:
            merged_starts.append(s)
            merged_ends.append(e)
    return (
        np.array(merged_starts, dtype=np.int64),
        np.array(merged_ends, dtype=np.int64),
    )


def subtract_intervals_df(allowed_df, exclude_df, margin=0, chroms=None):
    """Remove ``exclude_df`` intervals (each expanded by ``margin`` bp on
    both sides) from ``allowed_df``, per chromosome.

    Used to keep null placement (e.g. for ``--null_scope candidate``) from
    landing on or immediately beside a real peak's own footprint — see
    ``docs/trogdor_dreg_peak_calling_findings.md``'s "Null-Log Diagnostic"
    section for why an unmodified candidate footprint is nearly all real
    peak territory with no independent "elsewhere" to draw from.

    Parameters
    ----------
    allowed_df, exclude_df : pd.DataFrame
        Columns ``chrom``, ``start``, ``end``.
    margin : int
        Symmetric bp expansion applied to each ``exclude_df`` interval
        before subtracting. ``0`` excludes only the exact footprint.
    chroms : list of str or None
        Chromosomes to process; defaults to the union of both inputs'
        chromosomes.

    Returns
    -------
    pd.DataFrame
        Remaining allowed intervals, columns ``chrom``, ``start``, ``end``.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0.")
    if chroms is None:
        chroms = sorted(
            set(allowed_df["chrom"].unique()) | set(exclude_df["chrom"].unique())
        )

    allowed = merge_intervals_df(allowed_df, chroms)
    exclude = merge_intervals_df(exclude_df, chroms)

    rows = []
    for chrom in chroms:
        a = allowed[allowed["chrom"] == chrom]
        if len(a) == 0:
            continue
        a_starts = a["start"].to_numpy(dtype=np.int64)
        a_ends = a["end"].to_numpy(dtype=np.int64)

        e = exclude[exclude["chrom"] == chrom]
        if len(e) == 0:
            rows.extend((chrom, int(s), int(en)) for s, en in zip(a_starts, a_ends))
            continue

        e_starts = np.maximum(0, e["start"].to_numpy(dtype=np.int64) - margin)
        e_ends = e["end"].to_numpy(dtype=np.int64) + margin
        e_starts, e_ends = _remerge_sorted_intervals(e_starts, e_ends)

        j = 0
        for a_start, a_end in zip(a_starts.tolist(), a_ends.tolist()):
            cur = a_start
            while j < len(e_ends) and e_ends[j] <= cur:
                j += 1
            k = j
            while k < len(e_starts) and e_starts[k] < a_end and cur < a_end:
                if e_starts[k] > cur:
                    rows.append((chrom, cur, min(int(e_starts[k]), a_end)))
                cur = max(cur, int(e_ends[k]))
                k += 1
            if cur < a_end:
                rows.append((chrom, cur, a_end))

    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def shuffle_peaks_within_intervals(peaks_df, allowed_df, chroms, rng):
    """Shuffle peaks so each peak remains fully contained in allowed intervals.

    Width and chromosome assignment are preserved. Peaks wider than every
    allowed interval on their chromosome are dropped.
    """
    allowed = merge_intervals_df(allowed_df, chroms)
    rows = []
    for chrom in chroms:
        sub = peaks_df[peaks_df["chrom"] == chrom].copy()
        allowed_chrom = allowed[allowed["chrom"] == chrom]
        if len(sub) == 0 or len(allowed_chrom) == 0:
            continue

        a_starts = allowed_chrom["start"].to_numpy(dtype=np.int64)
        a_ends = allowed_chrom["end"].to_numpy(dtype=np.int64)
        widths = (sub["end"] - sub["start"]).to_numpy(dtype=np.int64)
        new_starts = np.full(len(sub), -1, dtype=np.int64)

        unique_widths, width_inverse = np.unique(widths, return_inverse=True)
        for ui, width in enumerate(unique_widths):
            valid_ends = a_ends - width
            valid = valid_ends >= a_starts
            if not valid.any():
                continue
            starts = a_starts[valid]
            ends = valid_ends[valid]
            lengths = ends - starts + 1
            cumlen = np.cumsum(lengths)
            total = int(cumlen[-1])
            peak_mask = width_inverse == ui
            n = int(peak_mask.sum())
            r = rng.integers(0, total, size=n)
            idx = np.searchsorted(cumlen, r, side="right")
            idx = np.clip(idx, 0, len(cumlen) - 1)
            prev_cum = np.concatenate([[0], cumlen[:-1]])
            offsets = r - prev_cum[idx]
            new_starts[peak_mask] = starts[idx] + offsets

        keep = new_starts >= 0
        if keep.any():
            sub_keep = sub[keep].copy().reset_index(drop=True)
            sub_keep["start"] = new_starts[keep]
            sub_keep["end"] = new_starts[keep] + widths[keep]
            rows.append(sub_keep)

    if not rows:
        return peaks_df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def score_peaks(bw, peaks_df, chrom_sizes, stat, chroms, verbose=False):
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


def score_peaks_from_array(scores, peaks_df, chrom, output_stride, stat):
    """Score intervals from a per-chromosome score array."""
    sub = peaks_df[peaks_df["chrom"] == chrom]
    out = np.full(len(sub), np.nan, dtype=np.float32)
    for j, (_, row) in enumerate(sub.iterrows()):
        start_bin = max(0, int(row["start"]) // output_stride)
        end_bin = int(np.ceil(int(row["end"]) / output_stride))
        vals = scores[start_bin:end_bin]
        if len(vals) == 0:
            continue
        vals = np.nan_to_num(np.asarray(vals, dtype=np.float32))
        out[j] = vals.max() if stat == "max" else vals.mean()
    return out


def _logit(x, eps=1e-6):
    x = np.clip(np.asarray(x, dtype=np.float64), eps, 1 - eps)
    return np.log(x / (1 - x))


def _inv_logit(x):
    x = np.asarray(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))


def _thresholds_from_scores(real_scores, null_scores, n_thresholds, threshold_grid):
    scores = (
        np.concatenate([real_scores, null_scores])
        if len(null_scores)
        else np.asarray(real_scores)
    )
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        raise ValueError("Cannot compute FDR with no finite scores.")

    if threshold_grid == "linear":
        return np.linspace(scores.min(), scores.max(), n_thresholds)

    if threshold_grid == "logit":
        if scores.min() < 0 or scores.max() > 1:
            raise ValueError("--threshold_grid logit requires scores in [0, 1].")
        logits = _logit(scores)
        return _inv_logit(np.linspace(logits.min(), logits.max(), n_thresholds))

    if threshold_grid == "unique":
        unique = np.unique(scores)
        if len(unique) <= n_thresholds:
            return unique
        idx = np.linspace(0, len(unique) - 1, n_thresholds).round().astype(int)
        return unique[np.unique(idx)]

    if threshold_grid == "quantile":
        n_body = max(2, n_thresholds // 2)
        n_tail = max(2, n_thresholds - n_body)
        body_q = np.linspace(0, 1, n_body)
        # Enrich the high-score tail, where saturated TROGDOR probabilities
        # often make the empirical FDR decision.
        min_tail_step = max(1 / max(len(scores), 1), 1e-8)
        tail_q = 1 - np.geomspace(1e-2, min_tail_step, n_tail)
        q = np.unique(np.concatenate([body_q, tail_q, [0.0, 1.0]]))
        return np.unique(np.quantile(scores, q))

    raise ValueError(f"Unknown threshold_grid: {threshold_grid}")


def compute_fdr(
    real_scores,
    null_scores,
    n_shuffle,
    n_thresholds,
    threshold_grid="quantile",
):
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
        Number of thresholds to evaluate.
    threshold_grid : {"quantile", "linear", "logit", "unique"}
        Strategy used to choose score thresholds.

    Returns
    -------
    thresholds : np.ndarray, shape (n_thresholds,)
    n_real : np.ndarray, shape (n_thresholds,)
    n_null : np.ndarray, shape (n_thresholds,)
        Average null count per shuffle.
    fdr : np.ndarray, shape (n_thresholds,)
        Estimated FDR at each threshold, clipped to [0, 1].
    """
    real_scores = np.asarray(real_scores, dtype=np.float64)
    null_scores = np.asarray(null_scores, dtype=np.float64)
    thresholds = _thresholds_from_scores(
        real_scores, null_scores, n_thresholds, threshold_grid
    )

    real_sorted = np.sort(real_scores)
    n_real = len(real_sorted) - np.searchsorted(real_sorted, thresholds, side="left")
    n_real = n_real.astype(float)
    if len(null_scores) > 0:
        null_sorted = np.sort(null_scores)
        n_null_total = len(null_sorted) - np.searchsorted(
            null_sorted, thresholds, side="left"
        )
        n_null_total = n_null_total.astype(float)
        n_null = n_null_total / n_shuffle
    else:
        n_null = np.zeros(len(thresholds))

    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(n_real > 0, np.minimum(1.0, n_null / n_real), 1.0)

    return thresholds, n_real, n_null, fdr


def select_fdr_threshold(thresholds, n_real, fdr, fdr_target):
    """Return the first threshold whose empirical FDR is at or below target."""
    passing = np.where(fdr <= fdr_target)[0]
    if len(passing) == 0:
        return float("nan"), 0
    idx = int(passing[0])
    return float(thresholds[idx]), int(n_real[idx])
