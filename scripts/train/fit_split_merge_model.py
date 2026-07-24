#!/usr/bin/env python3
"""Fit a split/merge classifier for TROGDOR's `profile` peak caller.

Motivation
----------
`_segments_from_valleys` (chiaroscuro.peaks) currently decides whether two
adjacent local maxima are one peak or two using a single fixed threshold
(`valley_fraction`). dREG replaced an identical single-threshold rule with a
random forest over 10 features of the local score-profile geometry (summit
heights, valley depth, distances) -- see docs/trogdor_dreg_peak_calling_findings.md
for the exact feature derivation, verified against dREG's own R source.

This script fits three candidate classifiers over the same 10 features
(logistic regression, a depth-limited decision tree, and a random forest),
using ONLY K562 data, and reports held-out (by chromosome) precision/recall/
F1 for each so a winner can be picked on evidence rather than assumption.
GM12878 is never touched here -- it stays independent for the downstream
`compare_peaks.py` benchmark against dREG, matching this project's existing
K562-develop / GM12878-validate split discipline (the base TROGDOR model
itself is trained/validated on K562 only, per scripts/train/train.py).

This is a *dev*-time script: scikit-learn is a dev dependency only, never a
runtime one. The winning model's fitted parameters get pasted into
chiaroscuro.peaks as a small constant, evaluated at runtime with pure Python
(no sklearn/numpy needed for inference).

Usage
-----
python scripts/train/fit_split_merge_model.py \\
    --prob_bigwigs data/G1.prob.bw data/G2.prob.bw data/G3.prob.bw \\
        data/G5.prob.bw data/G6.prob.bw \\
    --tss_bed data/K562.positive.bed.gz -v

Each bigwig in --prob_bigwigs must have been written by `trogdor score` with
a storage threshold no higher than the intended --seed_score (default 0.5),
same requirement as the CLI's profile caller -- otherwise candidate blocks
will be missing the low-confidence shoulders needed for valley-splitting.
"""

import argparse
import os

import numpy as np
import pybigtools
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from chiaroscuro.peaks import (
    FEATURE_NAMES,
    _local_maxima,
    _merge_seed_blocks,
    _pairwise_features,
    _smooth_scores,
    resolve_seed_score,
)
from chiaroscuro.stats import read_bed3

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _label_pair(truth_starts, truth_ends, left_pos, right_pos):
    """Label one adjacent-summit pair by ground-truth-interval overlap.

    Returns 0 (merge), 1 (split), or None (ambiguous -- excluded from
    training). Span is [left_pos, right_pos); truth_starts/truth_ends must
    be sorted, non-overlapping arrays for this chromosome.

    Rule: exactly 1 truth interval overlaps the span AND both summits fall
    inside it -> merge. >=2 distinct truth intervals overlap the span ->
    split. Anything else (0 intervals; or exactly 1 but only one summit
    inside it) -> excluded, rather than guessed.
    """
    lo = int(np.searchsorted(truth_ends, left_pos, side="right"))
    hi = int(np.searchsorted(truth_starts, right_pos, side="left"))
    n_overlap = 0
    left_inside = False
    right_inside = False
    for i in range(lo, hi):
        s, e = truth_starts[i], truth_ends[i]
        if s < right_pos and e > left_pos:
            n_overlap += 1
            if s <= left_pos < e:
                left_inside = True
            if s <= right_pos < e:
                right_inside = True
    if n_overlap >= 2:
        return 1
    if n_overlap == 1 and left_inside and right_inside:
        return 0
    return None


def _export_linear(clf, feature_names, scaler=None):
    """Print the fitted LogisticRegression as raw-feature weights + bias.

    ``clf`` was fit on standardized features (``scaler``, a fitted
    StandardScaler) for numerical stability -- LR's raw feature scales
    (bp distances up to ~thousands vs. probability-scale values in [0,1])
    otherwise cause overflow during optimization. Standardization is affine
    (``z = (x - mean) / scale``), so it folds algebraically into equivalent
    raw-feature weights/bias: no scaler needed at runtime, inference stays a
    plain ``sigmoid(w . x_raw + b)`` dot product.
    """
    coef = clf.coef_[0]
    bias = clf.intercept_[0]
    if scaler is not None:
        raw_coef = coef / scaler.scale_
        raw_bias = bias - float(np.sum(coef * scaler.mean_ / scaler.scale_))
    else:
        raw_coef, raw_bias = coef, bias
    weights = ", ".join(f"{w:.8g}" for w in raw_coef)
    print("\n# --- LogisticRegression export (raw-feature weights, scaler folded in) ---")
    print(f"# feature order: {feature_names}")
    print(f"_SPLIT_MERGE_WEIGHTS = ({weights})")
    print(f"_SPLIT_MERGE_BIAS = {raw_bias:.8g}")


def _export_tree_nodes(tree_, node_id=0):
    """Flatten a fitted sklearn tree into a list of
    (feature_idx, threshold, left, right, leaf_proba) tuples. Internal nodes
    have feature_idx >= 0 and leaf_proba = -1.0; leaves have feature_idx =
    -1, threshold/left/right unused (0), leaf_proba = P(class 1).
    """
    nodes = []

    def _walk(node_id):
        idx = len(nodes)
        nodes.append(None)  # placeholder, fixed up below
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        if left_child == right_child:  # leaf
            counts = tree_.value[node_id][0]
            proba = float(counts[1] / counts.sum()) if counts.sum() > 0 else 0.0
            nodes[idx] = (-1, 0.0, 0, 0, proba)
        else:
            feat = int(tree_.feature[node_id])
            thresh = float(tree_.threshold[node_id])
            left_idx = _walk(left_child)
            right_idx = _walk(right_child)
            nodes[idx] = (feat, thresh, left_idx, right_idx, -1.0)
        return idx

    _walk(node_id)
    return nodes


def _export_tree(clf, feature_names, name="_SPLIT_MERGE_TREE"):
    nodes = _export_tree_nodes(clf.tree_)
    print(f"\n# --- {type(clf).__name__} export ---")
    print(f"# feature order: {feature_names}")
    print(f"# node tuple: (feature_idx, threshold, left, right, leaf_proba)")
    print(f"{name} = [")
    for n in nodes:
        print(f"    {n!r},")
    print("]")


def _export_forest(clf, feature_names):
    print("\n# --- RandomForestClassifier export ---")
    print(f"# feature order: {feature_names}")
    print(f"# each tree: list of (feature_idx, threshold, left, right, leaf_proba)")
    print("_SPLIT_MERGE_FOREST = [")
    for est in clf.estimators_:
        nodes = _export_tree_nodes(est.tree_)
        print(f"    {nodes!r},")
    print("]")


def _print_separability(X, y, feature_names, title):
    """Print per-feature min/mean/max by class, flagging features with no
    overlap between merge and split -- a red flag that the classifier is
    exploiting a labeling-rule artifact (e.g. distance confound) rather than
    genuine valley-shape signal.
    """
    print(f"\n--- {title} ---")
    print(f"{'Feature':<8} {'merge min/mean/max':>28} {'split min/mean/max':>28} {'overlap?':>9}")
    merge_mask, split_mask = y == 0, y == 1
    for j, name in enumerate(feature_names):
        m, s = X[merge_mask, j], X[split_mask, j]
        if len(m) == 0 or len(s) == 0:
            print(f"{name:<8} {'(one class empty)':>28} {'':>28} {'':>9}")
            continue
        m_range = f"{m.min():.4g}/{m.mean():.4g}/{m.max():.4g}"
        s_range = f"{s.min():.4g}/{s.mean():.4g}/{s.max():.4g}"
        overlap = "no" if (m.max() < s.min() or s.max() < m.min()) else "yes"
        print(f"{name:<8} {m_range:>28} {s_range:>28} {overlap:>9}")


def _build_candidates(args):
    return {
        "LogisticRegression": (
            LogisticRegression(
                C=args.lr_C,
                class_weight="balanced",
                max_iter=1000,
                random_state=args.seed,
            ),
            True,
        ),
        f"DecisionTree(depth={args.tree_max_depth})": (
            DecisionTreeClassifier(
                max_depth=args.tree_max_depth,
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced",
                random_state=args.seed,
            ),
            False,
        ),
        f"RandomForest(n={args.n_estimators},depth={args.max_depth})": (
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced",
                random_state=args.seed,
            ),
            False,
        ),
    }


def _fit_and_compare(X_train, y_train, X_val, y_val, args, title):
    """Fit LR/tree/RF on (X_train, y_train), evaluate on (X_val, y_val), print
    a comparison table. Returns {name: (clf, needs_scaling, f1)}, or None if
    either split is empty/single-class (nothing meaningful to fit/evaluate).
    """
    print(f"\n--- {title} ---")
    if len(y_val) == 0 or len(set(y_val.tolist())) < 2:
        print("Skipped: held-out subset is empty or single-class.")
        return None
    if len(y_train) == 0 or len(set(y_train.tolist())) < 2:
        print("Skipped: train subset is empty or single-class.")
        return None

    # LogisticRegression needs standardized features for numerical stability
    # (raw feature scales range from bp distances in the thousands down to
    # probability-scale values in [0,1], which otherwise overflows the
    # solver). Tree-based models are scale-invariant and use raw features.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    candidates = _build_candidates(args)
    print(f"{'Model':<32} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    results = {}
    for name, (clf, scale) in candidates.items():
        clf.fit(X_train_scaled if scale else X_train, y_train)
        pred = clf.predict(X_val_scaled if scale else X_val)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", zero_division=0
        )
        results[name] = (clf, scale, f1)
        print(f"{name:<32} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prob_bigwigs",
        nargs="+",
        required=True,
        help="Scored probability bigWig(s) (output of `trogdor score`), stored "
        "densely enough to cover --seed_score.",
    )
    parser.add_argument(
        "--tss_bed",
        default=os.path.join(DATA_DIR, "K562.positive.bed.gz"),
        help="Ground-truth TSS/TIR BED shared by all --prob_bigwigs. Default: "
        "K562.positive.bed.gz (dREG-derived; matches the shipped model).",
    )
    parser.add_argument("--min_score", type=float, default=0.95)
    parser.add_argument(
        "--seed_score",
        type=float,
        default=None,
        help="Defaults to min(min_score, 0.5), matching resolve_seed_score.",
    )
    parser.add_argument("--max_gap", type=int, default=0)
    parser.add_argument("--smooth_bins", type=int, default=3)
    parser.add_argument(
        "--val_chroms",
        nargs="+",
        default=["chr21", "chr22"],
        help="Chromosomes held out for evaluation only -- never used to fit "
        "any of the three candidate classifiers. Default: chr21 chr22.",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Restrict to these chromosomes overall (default: all chromosomes "
        "in the first --prob_bigwigs file).",
    )
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for the tree/RF candidates (not the exported depth-3 tree; see --tree_max_depth).")
    parser.add_argument("--tree_max_depth", type=int, default=3, help="Max depth for the single-tree candidate.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees for the RF candidate.")
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument(
        "--lr_C",
        type=float,
        default=0.01,
        help="Inverse L2 regularization strength for LogisticRegression "
        "(sklearn's --C; smaller = stronger regularization). Default 0.01, "
        "well below sklearn's default 1.0 -- these 10 engineered features on "
        "a modest pair dataset are prone to near-perfect linear separability, "
        "which otherwise drives LR's coefficients toward infinity (visible as "
        "divide-by-zero/overflow RuntimeWarnings during fitting; verified "
        "0.1 still warned on a single-replicate smoke test, 0.01 did not).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    seed_score = resolve_seed_score(args.min_score, args.seed_score)
    truth = read_bed3(args.tss_bed)

    rows = []  # (feat_0, ..., feat_9, label, is_val)
    diag_total_pairs = 0
    diag_excluded = 0
    diag_blocks_total = 0
    diag_blocks_ge3 = 0

    for bw_idx, bw_path in enumerate(args.prob_bigwigs):
        bw = pybigtools.open(bw_path)
        chrom_sizes = dict(bw.chroms())
        chroms = args.chroms if args.chroms else sorted(chrom_sizes)
        if args.verbose:
            print(f"[{bw_idx + 1}/{len(args.prob_bigwigs)}] {bw_path}: {len(chroms)} chromosomes")
        for chrom in chroms:
            if chrom not in chrom_sizes:
                continue
            chrom_len = chrom_sizes[chrom]
            intervals = [
                (s, e, v)
                for s, e, v in bw.records(chrom, 0, chrom_len)
                if not np.isnan(v)
            ]
            if not intervals:
                continue

            chrom_truth = truth[truth["chrom"] == chrom].sort_values("start")
            truth_starts = chrom_truth["start"].to_numpy()
            truth_ends = chrom_truth["end"].to_numpy()

            blocks = _merge_seed_blocks(intervals, seed_score, args.max_gap)
            is_val = chrom in args.val_chroms
            for block in blocks:
                raw_scores = [s for _, _, s in block]
                smooth = _smooth_scores(raw_scores, args.smooth_bins)
                summit_idxs = [
                    i for i in _local_maxima(smooth) if block[i][2] >= args.min_score
                ]
                diag_blocks_total += 1
                if len(summit_idxs) >= 3:
                    diag_blocks_ge3 += 1
                for left, right in zip(summit_idxs[:-1], summit_idxs[1:]):
                    if right - left <= 1:
                        continue
                    valley_i = min(range(left + 1, right), key=lambda i: smooth[i])
                    diag_total_pairs += 1

                    left_pos = block[left][0]
                    right_pos = block[right][0]
                    label = _label_pair(truth_starts, truth_ends, left_pos, right_pos)
                    if label is None:
                        diag_excluded += 1
                        continue

                    features = _pairwise_features(block, smooth, left, valley_i, right)
                    rows.append((*features, label, is_val))
        bw.close()

    if not rows:
        raise SystemExit("No labeled pairs generated -- check inputs/thresholds.")

    data = np.array(rows, dtype=np.float64)
    X, y, is_val = data[:, :10], data[:, 10], data[:, 11].astype(bool)
    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    print(f"\nTotal pairs generated:      {diag_total_pairs}")
    print(f"Excluded (ambiguous):       {diag_excluded} ({diag_excluded / max(diag_total_pairs, 1):.1%})")
    print(f"Labeled pairs kept:         {len(rows)}")
    print(f"  train (non-val chroms):   {len(y_train)}  (merge={int((y_train == 0).sum())}, split={int((y_train == 1).sum())})")
    print(f"  held-out ({','.join(args.val_chroms)}):  {len(y_val)}  (merge={int((y_val == 0).sum())}, split={int((y_val == 1).sum())})")
    print(f"Blocks with >=3 qualifying summits: {diag_blocks_ge3}/{diag_blocks_total} "
          f"({diag_blocks_ge3 / max(diag_blocks_total, 1):.1%}) "
          "-- sizes the risk of skipping dREG's iterative re-merge loop")

    _print_separability(
        X, y, FEATURE_NAMES,
        "Per-feature separability, ALL labeled pairs (a merge/split range "
        "with no overlap means that single feature alone already perfectly "
        "separates the classes -- a red flag that the *excluded* 96%+ "
        "ambiguous pairs, not this kept subset, are the real test)",
    )

    if len(y_val) == 0 or len(set(y_val.tolist())) < 2:
        raise SystemExit(
            "Held-out set is empty or single-class; pick --val_chroms with "
            "both merge and split examples."
        )
    minority_count = min((y_val == 0).sum(), (y_val == 1).sum())
    if len(y_val) < 30 or minority_count < 5:
        print(
            f"\nWARNING: held-out set is tiny (n={len(y_val)}, minority class "
            f"n={int(minority_count)}) -- precision/recall/F1 below are NOT "
            "reliable evidence of which model is better. This is expected "
            "for a single-replicate smoke test; do not trust these numbers "
            "for a real model-selection decision without more data (the "
            "full G1-G6 set, not one replicate)."
        )

    results = _fit_and_compare(
        X_train, y_train, X_val, y_val, args,
        "Held-out (chromosome-split) comparison, ALL labeled pairs",
    )

    # Distance-matched control: `dist` (and its correlates r1/r2/d2/d3/dr)
    # can be a trivial-separability artifact of the labeling rule itself
    # (merge = both summits inside one truth interval, bounding how far
    # apart they can be; split = summits span two distinct intervals, which
    # tends to mean they're far apart) -- not genuine valley-shape signal.
    # Restricting to the dist range where both classes actually overlap
    # forces the classifier to use the remaining features instead.
    dist_idx = FEATURE_NAMES.index("dist")
    dist = X[:, dist_idx]
    merge_mask, split_mask = y == 0, y == 1
    m_dist, s_dist = dist[merge_mask], dist[split_mask]
    overlap_lo = max(m_dist.min(), s_dist.min())
    overlap_hi = min(m_dist.max(), s_dist.max())
    print(
        f"\n--- Distance-matched control (merge dist=[{m_dist.min():.4g}, "
        f"{m_dist.max():.4g}], split dist=[{s_dist.min():.4g}, {s_dist.max():.4g}]) ---"
    )
    if overlap_lo > overlap_hi:
        print(
            "No overlap in `dist` between classes at all -- every kept pair "
            "is trivially separable by distance alone. This confirms the "
            "labeling rule's classes are distance-confounded by construction "
            "for this run; the held-out F1 above reflects that confound, not "
            "necessarily genuine valley-shape signal. Skipping the "
            "distance-matched fit -- there is no data left once distance is "
            "controlled for."
        )
    else:
        band_mask = (dist >= overlap_lo) & (dist <= overlap_hi)
        Xb, yb, is_val_b = X[band_mask], y[band_mask], is_val[band_mask]
        print(
            f"Overlap band: dist in [{overlap_lo:.4g}, {overlap_hi:.4g}] -- "
            f"{int(band_mask.sum())} pairs ({int((yb == 0).sum())} merge, "
            f"{int((yb == 1).sum())} split)"
        )
        _print_separability(
            Xb, yb, FEATURE_NAMES,
            "Per-feature separability WITHIN the distance-matched band",
        )
        band_results = _fit_and_compare(
            Xb[~is_val_b], yb[~is_val_b], Xb[is_val_b], yb[is_val_b], args,
            "Held-out comparison WITHIN distance-matched band "
            "(tests for signal beyond raw distance)",
        )
        if band_results is None:
            print(
                "Not enough data in the distance-matched band's train/held-out "
                "split to fit -- can't confirm or rule out signal beyond "
                "distance from this run alone."
            )

    # Shape-only ablation: drop dist/r1/r2 (and, since they're derived from
    # position rather than score, keep d1/d2/d3/dr/y1/y2/maxy) to test
    # directly whether the classifiers are exploiting real valley-shape
    # signal or just riding distance/r2's trivial separability. This is a
    # cleaner test than the distance-matched band above when that band turns
    # out empty (no dist overlap at all) -- it doesn't need any overlap to
    # exist, since it removes the confounded features outright instead of
    # conditioning on them.
    shape_feature_names = tuple(n for n in FEATURE_NAMES if n not in ("dist", "r1", "r2"))
    shape_idxs = [FEATURE_NAMES.index(n) for n in shape_feature_names]
    Xs = X[:, shape_idxs]
    print(
        f"\n--- Shape-only ablation (dropped dist/r1/r2; kept "
        f"{shape_feature_names}) ---"
    )
    _print_separability(
        Xs, y, shape_feature_names,
        "Per-feature separability, shape-only features",
    )
    shape_results = _fit_and_compare(
        Xs[~is_val], y[~is_val], Xs[is_val], y[is_val], args,
        "Held-out comparison, shape-only features "
        "(tests for signal independent of distance/r1/r2)",
    )
    if shape_results is None:
        print(
            "Not enough data to fit/evaluate the shape-only subset from this "
            "run alone."
        )

    if results is None:
        raise SystemExit(
            "Held-out set is empty or single-class; pick --val_chroms with "
            "both merge and split examples."
        )
    winner_name = max(results, key=lambda n: results[n][2])
    winner_clf, winner_scale, _ = results[winner_name]
    print(f"\nWinner (highest held-out F1): {winner_name}")
    print("Refitting winner on ALL K562 pairs (train + held-out) for the shipped model...")
    final_scaler = None
    if winner_scale:
        final_scaler = StandardScaler().fit(X)
        winner_clf.fit(final_scaler.transform(X), y)
    else:
        winner_clf.fit(X, y)

    if isinstance(winner_clf, LogisticRegression):
        _export_linear(winner_clf, FEATURE_NAMES, scaler=final_scaler)
    elif isinstance(winner_clf, RandomForestClassifier):
        _export_forest(winner_clf, FEATURE_NAMES)
    elif isinstance(winner_clf, DecisionTreeClassifier):
        _export_tree(winner_clf, FEATURE_NAMES)


if __name__ == "__main__":
    main()
