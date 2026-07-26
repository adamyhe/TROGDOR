"""Peak-calling helpers for TROGDOR probability intervals."""

from __future__ import annotations

import math


def resolve_seed_score(min_score, seed_score):
    """Resolve the profile caller's permissive candidate-seed threshold.

    When unset, seeds at half of ``min_score`` (capped at 0.5) so local-maxima
    and valley splitting has candidate blocks wider than a single bin to work
    with, instead of silently collapsing to ``min_score``.
    """
    if seed_score is None:
        return min(min_score, 0.5)
    return seed_score


def _validate_profile_params(
    seed_score,
    min_score,
    max_gap,
    min_width,
    max_width,
    smooth_bins,
    valley_fraction,
    boundary_fraction,
    split_merge_rule="threshold",
    split_merge_cutoff=0.5,
    max_merge_distance=None,
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
    if split_merge_rule not in {"threshold", "learned"}:
        raise ValueError("split_merge_rule must be 'threshold' or 'learned'.")
    if not 0 <= split_merge_cutoff <= 1:
        raise ValueError("split_merge_cutoff must be in [0, 1].")
    if max_merge_distance is not None and max_merge_distance <= 0:
        raise ValueError("max_merge_distance must be > 0.")


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


FEATURE_NAMES = ("dist", "r1", "r2", "y1", "y2", "maxy", "d1", "d2", "d3", "dr")


def _pairwise_features(block, scores, left, valley_i, right):
    """Return the 10 dREG-equivalent split/merge features for one adjacent
    summit pair, in ``FEATURE_NAMES`` order.

    ``block`` gives real bp coordinates (its raw score, position 2, is not
    used here); ``scores`` is the smoothed score list, index-aligned with
    ``block`` — matching ``_segments_from_valleys``'s existing convention of
    computing valley depth on smoothed scores, not raw ones.
    """
    x_left, x_right, x_valley = block[left][0], block[right][0], block[valley_i][0]
    y1, y2, valley_score = scores[left], scores[right], scores[valley_i]

    dist = float(x_right - x_left)
    r1 = float(x_valley - x_left)
    r2 = float(x_right - x_valley)
    maxy = max(y1, y2)
    d1 = abs(y1 - y2)
    d2 = max(0.0, min(y1, y2) - valley_score)
    d3 = valley_score
    denom = d1 + d3
    dr = d2 / denom if denom > 1e-9 else 0.0
    return (dist, r1, r2, y1, y2, maxy, d1, d2, d3, dr)


# NOT RECOMMENDED FOR PRODUCTION -- retained as scaffolding only. Fitted by
# scripts/train/fit_split_merge_model.py on K562 pairs labeled by borrowing
# K562.positive.bed.gz (groHMM+DNase) truth-interval membership. That label
# source turned out to be distance-confounded by construction (dist/r2 alone
# perfectly separate merge/split with zero value-range overlap), and the
# actual winner-selection picked a full-feature model that degenerates to a
# trivial single-distance threshold rather than genuine valley-shape
# reasoning -- see docs/trogdor_dreg_peak_calling_findings.md's "Multi-Feature
# Split/Merge Fitting: Distance-Confounded Labels" and
# docs/peak_calling_handoff.md item 2c. Do not treat split_merge_rule="learned"
# as validated until the label source is rebuilt from point-resolution (e.g.
# PRO-cap) TSS calls instead of truth-interval membership.
_SPLIT_MERGE_WEIGHTS = (
    0.0020617864,
    0.0035780178,
    0.0036693964,
    0.5511516,
    2.7351534,
    13.326024,
    103.02077,
    4.0272446,
    -4.012432,
    1.7906594,
)
_SPLIT_MERGE_BIAS = -16.161508


def _predict_split(features, weights=_SPLIT_MERGE_WEIGHTS, bias=_SPLIT_MERGE_BIAS):
    """Return P(split) for one adjacent-summit pair's ``FEATURE_NAMES`` tuple."""
    z = bias + sum(w * f for w, f in zip(weights, features))
    return 1.0 / (1.0 + math.exp(-z))


def _segments_from_valleys(
    block,
    scores,
    min_score,
    valley_fraction,
    split_merge_rule="threshold",
    split_merge_cutoff=0.5,
    max_merge_distance=None,
):
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
        if split_merge_rule == "learned":
            features = _pairwise_features(block, scores, left, valley_i, right)
            should_split = _predict_split(features) >= split_merge_cutoff
        else:
            weaker_summit = min(scores[left], scores[right])
            should_split = scores[valley_i] <= weaker_summit * valley_fraction
        # Hard sanity cap, independent of the rule above: never merge summits
        # farther apart than this, no matter how shallow the valley looks --
        # applies on top of either the threshold or learned decision.
        if max_merge_distance is not None:
            dist = block[right][0] - block[left][0]
            should_split = should_split or dist >= max_merge_distance
        if should_split:
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


def _trim_segment(block, start_i, end_i, seed_score, boundary_fraction):
    """Trim a segment's low-confidence shoulders.

    The keep-threshold interpolates between ``seed_score`` (every bin in the
    segment already cleared this during seeding, so ``boundary_fraction=0``
    is a true no-op) and the segment's own summit score
    (``boundary_fraction=1`` keeps only bins at the summit). Anchoring at
    ``seed_score`` instead of ``0`` keeps ``boundary_fraction`` meaningful
    across its whole range regardless of how far ``seed_score`` sits below
    the summit — anchoring at ``0`` made most of the range a no-op whenever
    the summit was much higher than ``seed_score``.
    """
    segment = block[start_i : end_i + 1]
    summit_score = max(v for _, _, v in segment)
    threshold = seed_score + boundary_fraction * (summit_score - seed_score)
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
    split_merge_rule="threshold",
    split_merge_cutoff=0.5,
    max_merge_distance=None,
):
    """Call score-profile-aware peaks from sorted scored bins.

    This caller keeps the current TROGDOR score as a non-parametric ranking
    signal. It does not compute p-values. Candidate blocks are seeded from a
    permissive score threshold, then nearby local maxima are split when the
    intervening valley is deep enough.

    ``split_merge_rule="learned"`` replaces the fixed ``valley_fraction``
    threshold with a pretrained classifier (see ``_predict_split``) over the
    same dREG-equivalent geometry features (``_pairwise_features``),
    splitting when its predicted split-probability is >= ``split_merge_cutoff``.

    ``max_merge_distance``, if set, forces a split whenever two adjacent
    summits are at least this many bp apart, regardless of valley depth or
    ``split_merge_rule`` -- a hard cap against merging distant summits into
    one implausibly wide peak.
    """
    seed_score = resolve_seed_score(min_score, seed_score)

    _validate_profile_params(
        seed_score,
        min_score,
        max_gap,
        min_width,
        max_width,
        smooth_bins,
        valley_fraction,
        boundary_fraction,
        split_merge_rule,
        split_merge_cutoff,
        max_merge_distance,
    )

    blocks = _merge_seed_blocks(intervals, seed_score, max_gap)
    peaks = []
    for block in blocks:
        raw_scores = [score for _, _, score in block]
        smooth = _smooth_scores(raw_scores, smooth_bins)
        for start_i, end_i in _segments_from_valleys(
            block,
            smooth,
            min_score,
            valley_fraction,
            split_merge_rule,
            split_merge_cutoff,
            max_merge_distance,
        ):
            start_i, end_i = _trim_segment(
                block, start_i, end_i, seed_score, boundary_fraction
            )
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
