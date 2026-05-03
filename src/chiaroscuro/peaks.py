"""Peak-calling helpers for TROGDOR probability intervals."""

from __future__ import annotations


def call_peaks(intervals, min_score=0.95, max_gap=0, min_width=0):
    """Call merged peak intervals from sorted scored bins.

    Parameters
    ----------
    intervals : iterable of (start, end, score)
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

    peaks = []
    current = None

    for start, end, score in intervals:
        if score < min_score:
            continue

        start = int(start)
        end = int(end)
        score = float(score)

        if current is None or start > current["end"] + max_gap:
            if current is not None and current["end"] - current["start"] >= min_width:
                peaks.append(current)
            current = {
                "start": start,
                "end": end,
                "score": score,
                "summit_start": start,
                "summit_end": end,
                "summit_score": score,
            }
            continue

        current["end"] = max(current["end"], end)
        if score > current["score"]:
            current["score"] = score
        if score > current["summit_score"]:
            current["summit_start"] = start
            current["summit_end"] = end
            current["summit_score"] = score

    if current is not None and current["end"] - current["start"] >= min_width:
        peaks.append(current)

    return peaks
