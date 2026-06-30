"""Peak-calling helpers for TROGDOR probability intervals."""

from __future__ import annotations


def _validate_profile_params(
    seed_score,
    min_score,
    max_gap,
    min_width,
    max_width,
    smooth_bins,
    valley_fraction,
    boundary_fraction,
):
    if max_gap < 0:
        raise ValueError("max_gap must be >= 0.")
    if min_width < 0:
        raise ValueError("min_width must be >= 0.")
    if max_width is not None and max_width < 0:
        raise ValueError("max_width must be >= 0.")
    if max_width is not None and min_width > max_width:
        raise ValueError("min_width must be <= max_width.")
    if smooth_bins < 1:
        raise ValueError("smooth_bins must be >= 1.")
    if not 0 <= valley_fraction <= 1:
        raise ValueError("valley_fraction must be in [0, 1].")
    if not 0 <= boundary_fraction <= 1:
        raise ValueError("boundary_fraction must be in [0, 1].")
    if seed_score > min_score:
        raise ValueError("seed_score must be <= min_score.")


def _peak_from_bins(bins):
    max_i = max(range(len(bins)), key=lambda i: bins[i][2])
    return {
        "start": int(bins[0][0]),
        "end": int(bins[-1][1]),
        "score": max(float(v) for _, _, v in bins),
        "summit_start": int(bins[max_i][0]),
        "summit_end": int(bins[max_i][1]),
        "summit_score": float(bins[max_i][2]),
    }


def _merge_seed_blocks(intervals, seed_score, max_gap):
    blocks = []
    current = []
    for start, end, score in intervals:
        if score < seed_score:
            continue
        item = (int(start), int(end), float(score))
        if current and item[0] > current[-1][1] + max_gap:
            blocks.append(current)
            current = []
        current.append(item)
    if current:
        blocks.append(current)
    return blocks


def _smooth_scores(scores, smooth_bins):
    if smooth_bins <= 1 or len(scores) <= 2:
        return [float(s) for s in scores]

    radius = smooth_bins // 2
    smoothed = []
    for i in range(len(scores)):
        lo = max(0, i - radius)
        hi = min(len(scores), i + radius + 1)
        smoothed.append(float(sum(scores[lo:hi]) / (hi - lo)))
    return smoothed


def _local_maxima(scores):
    if not scores:
        return []
    if len(scores) == 1:
        return [0]

    maxima = []
    for i, score in enumerate(scores):
        left = scores[i - 1] if i > 0 else float("-inf")
        right = scores[i + 1] if i + 1 < len(scores) else float("-inf")
        if score >= left and score >= right and (score > left or score > right):
            maxima.append(i)

    if maxima:
        return maxima
    return [max(range(len(scores)), key=lambda i: scores[i])]


def _segments_from_valleys(block, scores, min_score, valley_fraction):
    summit_idxs = [i for i in _local_maxima(scores) if block[i][2] >= min_score]
    if not summit_idxs:
        return []
    if len(summit_idxs) == 1:
        return [(0, len(block) - 1)]

    cut_points = []
    for left, right in zip(summit_idxs[:-1], summit_idxs[1:]):
        if right - left <= 1:
            continue
        valley_i = min(range(left + 1, right), key=lambda i: scores[i])
        weaker_summit = min(scores[left], scores[right])
        if scores[valley_i] <= weaker_summit * valley_fraction:
            cut_points.append(valley_i)

    if not cut_points:
        return [(0, len(block) - 1)]

    segments = []
    start_i = 0
    for cut_i in cut_points:
        left_end = max(start_i, cut_i - 1)
        if any(block[i][2] >= min_score for i in range(start_i, left_end + 1)):
            segments.append((start_i, left_end))
        start_i = min(len(block) - 1, cut_i + 1)

    if any(block[i][2] >= min_score for i in range(start_i, len(block))):
        segments.append((start_i, len(block) - 1))
    return segments


def _trim_segment(block, start_i, end_i, boundary_fraction):
    segment = block[start_i : end_i + 1]
    summit_score = max(v for _, _, v in segment)
    threshold = summit_score * boundary_fraction
    keep = [i for i in range(start_i, end_i + 1) if block[i][2] >= threshold]
    if not keep:
        return start_i, end_i
    return min(keep), max(keep)


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


def call_profile_peaks(
    intervals,
    min_score=0.95,
    seed_score=None,
    max_gap=0,
    min_width=0,
    max_width=None,
    smooth_bins=1,
    valley_fraction=0.5,
    boundary_fraction=0.0,
):
    """Call score-profile-aware peaks from sorted scored bins.

    This caller keeps the current TROGDOR score as a non-parametric ranking
    signal. It does not compute p-values. Candidate blocks are seeded from a
    permissive score threshold, then nearby local maxima are split when the
    intervening valley is deep enough.
    """
    if seed_score is None:
        seed_score = min_score

    _validate_profile_params(
        seed_score,
        min_score,
        max_gap,
        min_width,
        max_width,
        smooth_bins,
        valley_fraction,
        boundary_fraction,
    )

    blocks = _merge_seed_blocks(intervals, seed_score, max_gap)
    peaks = []
    for block in blocks:
        raw_scores = [score for _, _, score in block]
        smooth = _smooth_scores(raw_scores, smooth_bins)
        for start_i, end_i in _segments_from_valleys(
            block, smooth, min_score, valley_fraction
        ):
            start_i, end_i = _trim_segment(block, start_i, end_i, boundary_fraction)
            peak = _peak_from_bins(block[start_i : end_i + 1])
            width = peak["end"] - peak["start"]
            if width < min_width:
                continue
            if max_width is not None and width > max_width:
                continue
            if peak["summit_score"] < min_score:
                continue
            peaks.append(peak)

    return peaks
