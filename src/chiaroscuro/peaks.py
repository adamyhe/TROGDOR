"""Peak-calling helpers for TROGDOR probability intervals."""

from __future__ import annotations

import numpy as np


def _as_bin_arrays(intervals):
    """Normalise scored bins to ``(starts, ends, values)`` NumPy arrays.

    Accepts either a ``(starts, ends, values)`` tuple of arrays -- what
    ``cmd_peaks`` passes, so a dense genome never becomes per-bin Python
    objects -- or the historical iterable of ``(start, end, score)`` tuples.
    Values are float64 so threshold comparisons match the tuple path's Python
    floats: under NumPy 2 promotion a float32 array compares against a Python
    float in float32, which would let e.g. a stored 0.95f pass ``>= 0.95``.
    """
    if (
        isinstance(intervals, tuple)
        and len(intervals) == 3
        and all(isinstance(a, np.ndarray) for a in intervals)
    ):
        starts, ends, values = intervals
    else:
        intervals = list(intervals)
        if not intervals:
            starts, ends, values = [], [], []
        else:
            starts, ends, values = zip(*intervals)
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def call_peaks(intervals, min_score=0.95, max_gap=0, min_width=0):
    """Call merged peak intervals from sorted scored bins.

    Parameters
    ----------
    intervals : iterable of (start, end, score), or (starts, ends, values) arrays
        Sorted scored intervals, usually fixed-width output bins.
    min_score : float
        Minimum score for an interval to seed or extend a peak.
    max_gap : int
        Maximum gap in bp allowed between passing intervals before they are
        merged into the same peak. ``0`` preserves the historical behaviour:
        only directly abutting bins merge.
    min_width : int
        Minimum final peak width in bp. Peaks narrower than this are discarded.

    Returns
    -------
    list of dict
        Each dict has ``start``, ``end``, ``score``, ``summit_start``,
        ``summit_end``, and ``summit_score``.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be >= 0.")
    if min_width < 0:
        raise ValueError("min_width must be >= 0.")

    starts, ends, values = _as_bin_arrays(intervals)
    keep = values >= min_score
    starts, ends, values = starts[keep], ends[keep], values[keep]
    if len(starts) == 0:
        return []

    # A peak's end is the max end merged so far, which is what each new bin's
    # gap is measured from.
    prev_ends = np.maximum.accumulate(ends)
    first = np.r_[0, np.flatnonzero(starts[1:] > prev_ends[:-1] + max_gap) + 1]
    peak_ends = np.maximum.reduceat(ends, first)
    peak_scores = np.maximum.reduceat(values, first)
    # Summit = first bin reaching the peak's max score.
    group = np.repeat(np.arange(len(first)), np.diff(np.r_[first, len(starts)]))
    at_max = np.flatnonzero(values == peak_scores[group])
    summits = at_max[np.unique(group[at_max], return_index=True)[1]]

    peaks = []
    for i, end, score, j in zip(
        first.tolist(), peak_ends.tolist(), peak_scores.tolist(), summits.tolist()
    ):
        start = int(starts[i])
        if end - start < min_width:
            continue
        peaks.append(
            {
                "start": start,
                "end": end,
                "score": score,
                "summit_start": int(starts[j]),
                "summit_end": int(ends[j]),
                "summit_score": float(values[j]),
            }
        )
    return peaks
