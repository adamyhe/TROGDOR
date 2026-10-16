# Peak-Calling Handoff

## Context

TROGDOR currently turns model scores into peaks with a simple
threshold-and-merge caller. The default `trogdor peaks` path loads scored
bigWig intervals, keeps bins with `score >= --min_score`, merges directly
abutting passing bins, and writes BED intervals with the peak max score.
The optional refined mode adds `max_gap`, `min_width`, summit columns, and
optional raw plus/minus support, but it is still fundamentally a
score-threshold caller.

This simplicity is likely contributing to the benchmark mismatch: TROGDOR
scores can rank true regions well, while the final called BED can include many
extra or poorly shaped intervals.

Important constraint: TROGDOR logits do not follow a known parametric
distribution. Do not add Gaussian, Laplace, Poisson, ZIP, or similar p-values
for TROGDOR scores/logits. Any significance/FDR layer for TROGDOR scores must
be empirical or non-parametric.

## What dREG Does

The current Danko-Lab dREG pipeline is more than thresholding. Its recommended
caller directly produces narrow peaks with peak score, probability-like value,
and center position. The older legacy path produced broad thresholded dREG
regions and then relied on dREG-HD-style refinement.

From the public dREG source:

- It first scores informative positions rather than every genomic base.
- It fills gaps between nearby high-scoring informative sites.
- It builds broad candidate regions by expanding scored sites and merging
  nearby signal blocks.
- It densifies scoring inside promising broad candidates.
- It smooths local score profiles, finds local maxima and valleys, and uses a
  random forest to split or merge adjacent local maxima.
- It filters peak candidates using a probability-like statistic and multiple
  testing correction.

Useful ideas for TROGDOR:

- Candidate generation should be broader than "passing adjacent bins".
- Boundary refinement should use local maxima, valleys, and summit-centered
  geometry.
- Adjacent maxima should be split only when the valley evidence supports
  distinct peaks.
- Very narrow or low-support artifacts should be filtered.

Do not copy directly:

- dREG's Laplace-style significance model. TROGDOR logits are non-parametric,
  so this would be unjustified.

## What PINTS Does

PINTS is a strand-aware nascent-TSS/TRE caller. It works from raw stranded
coverage rather than classifier logits and returns divergent, bidirectional,
and unidirectional TREs.

From the PINTS source and README:

- It finds strand-specific subpeaks from raw coverage using local peak finding
  and peak-width estimates.
- It merges adjacent subpeaks when the merged density remains high enough.
- It tests candidate windows against local flanking background after removing
  likely peak signal from the local environment.
- It uses Poisson/ZIP models and stratified multiple-testing correction for
  short versus broader candidates.
- It pairs plus- and minus-strand peaks within a configurable distance to
  label divergent/bidirectional/unidirectional elements.

Useful ideas for TROGDOR:

- Use raw plus/minus coverage to refine support and annotate directionality.
- Treat summit location as a first-class output field.
- Pair opposite-strand signal near a peak to distinguish bidirectional or
  divergent TREs from single-strand calls.
- Separate candidate geometry from statistical calibration.

Do not copy directly:

- PINTS' Poisson/ZIP p-values for TROGDOR logits. These models are about read
  count generation, not model-score generation.

## Recommended TROGDOR Caller Direction

The next caller should preserve TROGDOR's neural score as a non-parametric
ranking signal, while improving candidate geometry and empirical calibration.

Proposed staged plan:

1. Add a new caller mode, leaving the current default unchanged.
   Suggested mode name: `profile` or `local`.

2. Candidate generation:
   - Start from score bins above a permissive seed threshold.
   - Allow gaps up to a configurable distance.
   - Keep summit score and summit bin.
   - Apply `min_width`, optional `max_width`, and optional raw-signal support.

3. Boundary refinement:
   - Within each candidate block, smooth score bins lightly.
   - Find local maxima.
   - Split adjacent maxima only if the intervening valley is sufficiently low
     relative to the weaker summit.
   - Bound peaks by score crossing, valley position, or a summit-relative
     fraction of peak height.

4. Strand-aware annotation:
   - Optionally read plus/minus bigWigs.
   - Report local plus/minus support near each summit.
   - Optionally label candidates as bidirectional, plus-supported,
     minus-supported, or ambiguous.

5. Empirical calibration:
   - Score each final candidate by a small summit-centered smoothed score by
     default; keep raw summit score, max score, mean score, width, and
     optional raw-signal support available as diagnostics.
   - Build null candidates by shuffling/circular-shifting candidate locations
     within chromosomes or within informative/signal-covered regions.
   - Preserve widths, chromosome assignment, and ideally local mappability or
     signal-coverage constraints.
   - Estimate empirical FDR from null versus real candidate score ranks.

6. Benchmarking:
   - Compare default threshold-and-merge, refined mode, and the new profile
     caller on the same score tracks.
   - Report bin overlap, peak overlap, center-window precision/recall, peak
     widths, call counts, and empirical FDR/recall curves.
   - Re-run `compare_peaks.py` after the nested-overlap fix, since unmerged
     BEDs can otherwise distort peak-level metrics.

## Implementation Notes

- Keep current defaults stable for backwards compatibility.
- Avoid introducing p-values unless they are explicitly empirical.
- Prefer deterministic behavior with seed-controlled null generation.
- Use tests with synthetic score profiles:
  - adjacent high bins merge,
  - short gaps merge only when allowed,
  - nearby local maxima split only with a deep enough valley,
  - broad noisy plateaus do not explode into many tiny peaks,
  - nested/unmerged BED intervals are handled correctly in benchmarks.

## Status (2026-07-17)

The staged plan above is implemented: `call_profile_peaks` (candidate
seeding, valley splitting, boundary trimming), empirical calibration
(`--calibrate` on both `peaks` and `pipeline`, `candidate`/`genome` null
scopes, `summit`/`smoothed_summit`/`max`/`mean` statistics), and
strand-aware `--min_support_signal` gating are all in place. What the staged
plan did not anticipate is that self-referential calibration would itself be
hard to tune given how TROGDOR's score distribution actually looks in
practice — see `docs/trogdor_dreg_peak_calling_findings.md`'s
"Post-Implementation Calibration Findings" for the concrete trade-off
(candidate-null either fails to discriminate or over-punishes, genome-null
under-discriminates, depending on `min_score`/`seed_score`). The next steps
below are motivated by that finding, not by the original geometric concerns.

## Next Steps (Post-Calibration-Experiment)

1. **DONE, and closed as a dead end — candidate-null self-referential
   contamination.** `--null_exclusion_margin` was implemented (subtracts
   each called peak's own footprint, plus a margin, from the candidate-null
   allowed region) and confirmed mechanically correct: re-run on G7/GM12878
   at `margin=80` placed 100% of requested null draws with zero chromosomes
   running short, and null draws no longer land adjacent to real peaks.
   *But* this exposed a deeper, non-tunable problem:
   `--null_scope candidate`'s "elsewhere" (candidate territory that never
   produced a called peak) is mechanically capped at `min_score` — any
   territory that ever cleared `min_score` would have become a peak and been
   excluded. Real peaks are by definition `>= min_score`. So candidate-null
   will let ~100% of real peaks through no matter what `min_score`,
   `seed_score`, or margin is chosen — see
   `docs/trogdor_dreg_peak_calling_findings.md`'s "Margin-Exclusion Fix
   Confirmed, But Exposed a Deeper, Non-Tunable Problem" for the exact
   numbers. Do not keep tuning this path; it cannot produce a graded FDR by
   construction. The two ways out (only worth revisiting if independent
   validation, below, turns out to be insufficient on its own):
   - Exclude only the peak being tested from its own null territory (not
     every peak globally), so null draws land on *other* real peaks. Breaks
     the ceiling tautology, but becomes a relative-rank/triage measure, not
     a strict FDR.
   - A richer multi-feature split/merge decision (dREG's random-forest role
     — verified against dREG's own R source: 10 local score-profile geometry
     features, `dist`/`r1`/`r2`/`y1`/`y2`/`maxy`/`d1`/`d2`/`d3`/`dr`; no
     coverage or candidate-density features involved). This bullet originally
     tied it to null-comparison/calibration, which is now closed for good —
     but the split/merge decision itself is pure peak geometry (replacing
     `_segments_from_valleys`'s single `valley_fraction` threshold), not
     inherently a calibration concept, so it doesn't need to wait on anything
     above. Promoted to its own active item — see 2c below.

2. **DONE for G7/K562 groHMM+DNase — independent-ground-truth validation.**
   Ran on the `peak-geometry` branch using the surviving local assets in
   `tmp/trogdor/` (`G7.trogdor.prob.bw`, `K562.positive.bed.gz`): `trogdor
   fdr` produces a genuinely graded FDR curve (threshold 0.522 at FDR=0.05,
   86.7% of real truth peaks passing) — confirming it is not subject to the
   candidate-null tautology above. `compare_peaks.py --mode profile` vs.
   `simple` at the same `min_score=0.95` surfaced a real, actionable finding
   — see item 2a below — rather than a clean "profile wins" result. GM12878
   and ENCODE SCREEN/dREG/PINTS comparisons are still outstanding: those
   assets aren't present locally and need re-downloading via
   `scripts/data/download_peaks.sh`/`download_test_data.sh`/
   `download_genome.sh` (or re-scoring a fresh bigWig with `trogdor score`).
   Treat that as optional follow-on work, not blocking — full details and
   the metrics table are in `docs/trogdor_dreg_peak_calling_findings.md`'s
   "Independent-Ground-Truth Validation Results" section.

2a. **FIXED — the `boundary_fraction` default footgun.** The validation run
    above found `profile` mode, at its CLI defaults (`seed_score=0.5`,
    `boundary_fraction=0.0`), calls peaks 3.3x wider than `simple` mode's and
    trails it on bin-level precision/F1/Jaccard by roughly half, with
    *worse* center-window summit-localization sensitivity (0.594 vs. 0.680)
    — the opposite of what profile mode's smoothing/valley splitting is
    supposed to buy. Root cause: `_trim_segment` (`peaks.py`) computed its
    keep-threshold as `summit_score * boundary_fraction`, but every bin in a
    segment already cleared the `seed_score` floor during seeding — so
    trimming was a no-op until `boundary_fraction > seed_score /
    summit_score`. With the default `seed_score=0.5` and summit scores near
    1.0, `boundary_fraction` values up to ~`0.5` (confirmed a byte-for-byte
    no-op) did nothing.

    Fix: `_trim_segment` now interpolates the threshold between `seed_score`
    (kept as the true, principled no-op at `boundary_fraction=0.0`, since
    every bin already clears it) and `summit_score`
    (`boundary_fraction=1.0` keeps only the summit bin(s)):
    `threshold = seed_score + boundary_fraction * (summit_score -
    seed_score)`. This makes `boundary_fraction` meaningful across its whole
    `[0, 1]` range for any `seed_score`/summit-score combination, not just
    the specific default that happened to break. Verified on the same
    G7/K562 truth pair: `boundary_fraction=0.5` now actually trims (median
    width 704bp → 512bp, down from doing nothing) and improves precision
    0.1121→0.1419, F1 0.1936→0.2360, Jaccard 0.1072→0.1338 relative to the
    old no-op behavior — sitting between the untrimmed and
    `boundary_fraction=0.9` numbers as expected on a properly linear scale.
    Regression tests added in `tests/test_peaks.py`
    (`test_profile_boundary_fraction_zero_is_true_noop`,
    `test_profile_boundary_fraction_one_keeps_only_summit`) pin the new
    semantics at both ends of the range. `profile` mode still trails
    `simple` mode's raw bin-level precision/F1/Jaccard on this dataset even
    with trimming fixed — that's a real width/precision trade, not a bug,
    and users who want `simple`-mode-level bin precision should raise
    `--boundary_fraction` well above `0.5` (or lower `--seed_score`, since it
    now directly sets the trim floor). A follow-up sweep pinned the crossover:
    `boundary_fraction=0.95` matches `simple`'s median width and comes within
    ~3% of its precision/F1/Jaccard; `boundary_fraction=0.99` overtakes
    `simple` on precision/F1/Jaccard outright, at the cost of recall dropping
    below `simple`'s for the first time — see the full sweep table in
    `docs/trogdor_dreg_peak_calling_findings.md`. `boundary_fraction` spans
    the same precision/recall trade-off `simple`'s fixed threshold sits one
    point on, so "best" depends on what's being optimized for.

    **Decision: the CLI default for `--boundary_fraction` on both `peaks`
    and `pipeline` is now `0.95`** (changed from `0.0`), chosen as the best
    overall compromise on bin- and peak-level quality metrics together —
    matches `simple` mode's width/precision closely while keeping `profile`
    mode's peak PPV/center-window-specificity edge. The `_peak_params`/
    `_run_with_bigwig` `getattr(args, "boundary_fraction", ...)` fallbacks in
    `cli/commands.py` were updated to the same value for consistency. The
    library-level default in `call_profile_peaks` (`peaks.py`) deliberately
    stays at `0.0` — several unit tests in `tests/test_peaks.py` call it
    without specifying `boundary_fraction` to isolate seeding/splitting
    behavior from trimming, so the function itself keeps the conservative,
    easy-to-reason-about no-op default; only the CLI's product-facing
    default changed.

    **Decision: `--mode` on `trogdor peaks` now also defaults to `profile`**
    (changed from `simple`), matching `trogdor pipeline`'s `--peak_mode`,
    which already defaulted to `profile`. That asymmetry meant the two
    subcommands silently disagreed on out-of-the-box behavior; with this
    change plus the `boundary_fraction=0.95` default above, an unqualified
    `trogdor peaks -i scores.bw -o peaks.bed.gz --min_score 0.95` now runs
    the validated, tuned profile caller rather than the legacy
    threshold-and-merge one. Two `getattr(args, "mode", "simple")` fallbacks
    in `cli/commands.py` (`_peak_params`, and the `Namespace` `cmd_pipeline`
    builds to re-invoke `cmd_peaks` when streaming from a fresh model run)
    had the same stale-default problem `boundary_fraction`'s fallbacks did
    and were updated to `"profile"` for the same reason — they only matter
    if `args` is ever missing the attribute, but should still match what the
    CLI itself defaults to.

2b. **DONE — GM12878 `seed_score` sweep: `0.3` is the best single-threshold
    trade found, but a single threshold can't close the rest of the gap to
    dREG.** Swept `seed_score` ∈ {0.3, 0.1, 0.0} (`min_score`/
    `boundary_fraction` held at default). `seed_score=0.3` gives a real,
    favorably-priced recall gain (peak sensitivity 0.776→0.797, bin recall
    0.671→0.710) for a modest precision cost — clearly better-behaved than
    the earlier `min_score=0.9` attempt. `seed_score=0.0` is pathological
    (bin precision collapses to 0.0147 while candidate peak *count* stays
    identical to `0.3`/`0.1` — a handful of calls likely blew up to
    chromosome-scale width; do not use it). Even at the best-behaved
    aggressive point tested (`0.1`), recall/sensitivity remain well short of
    dREG's (0.738/0.814 vs. 0.852/0.870), while F1/Jaccard have already
    fallen to parity with dREG — diminishing returns setting in. Conclusion:
    recommend `seed_score=0.3` as the practical default-candidate for
    GM12878-like data, but reaching dREG's recall needs a richer decision
    rule than any single global threshold, not further threshold tuning.
    Full sweep table in `docs/trogdor_dreg_peak_calling_findings.md`'s
    "GM12878 Recall Gap vs. dREG" section.

2c. **CLOSED (for now) — richer multi-feature split/merge decision via a
    fitted classifier; distance-confounded labels made it untrustworthy,
    split into two follow-ups (2c-i done, 2c-ii open).** dREG uses a random
    forest over 10 local score-profile geometry features (`dist`, `r1`, `r2`,
    `y1`, `y2`, `maxy`, `d1`, `d2`, `d3`, `dr` — see item 1 above) to decide
    whether adjacent local maxima split or merge; TROGDOR's
    `_segments_from_valleys` (`peaks.py`) used a single fixed
    `valley_fraction` threshold on valley depth alone. Implemented the same
    10 features (`_pairwise_features`) and fit LogisticRegression/
    DecisionTree/RandomForest on K562 pairs
    (`scripts/train/fit_split_merge_model.py`), labeling each pair by
    borrowing `K562.positive.bed.gz` (groHMM+DNase) truth-interval
    membership: both summits in one truth interval → merge, spanning two
    distinct truth intervals → split.

    **Finding: this label source is structurally distance-confounded, not
    fixable by relabeling rules.** `dist`/`r2` alone perfectly separate
    merge/split on real G1-G6 K562 data with zero value-range overlap (merge
    `dist` ∈ [32,144]bp, split `dist` ∈ [224,1760]bp) — because a
    region-interval ground truth's own boundaries already encode *its*
    merge-radius convention, any pair-labeling rule built from interval
    membership inherits that as a distance signal. A distance-matched-band
    control (restricting to the `dist` range where classes overlap) found no
    overlap at all — there's no data left once distance is controlled for.
    A shape-only ablation (dropping `dist`/`r1`/`r2` entirely) still hit
    near-perfect held-out log-loss (0.008-0.03 for tree/RF/LR vs. the
    full-feature winner's 0.0000), showing real valley-shape signal exists
    independent of distance — but the actual winner-selection procedure
    (highest F1, log-loss tie-break) picked a full-feature decision tree
    that degenerated to a single raw-distance threshold (`r2 >= 128bp →
    split`), not a shape-aware rule — no more useful than the existing
    `valley_fraction` threshold, and with no reason to generalize to the 96%+
    of pairs excluded from training as ambiguous. (Also surfaced, separately:
    comparing log-loss between a single decision tree and a random forest is
    unreliable on small held-out sets — a tree with pure leaves reports hard
    0/1 probabilities and can look artificially better than a forest's
    honestly-averaged, less overconfident ones.)

    Full 10-feature runtime inference (`_predict_split`, `_pairwise_features`,
    `split_merge_rule="learned"`/`--split_merge_cutoff` on `peaks`/`pipeline`)
    is implemented and tested in `peaks.py`/`cli/`, opt-in and off by default
    — but the currently-hardcoded `_SPLIT_MERGE_WEIGHTS`/`_SPLIT_MERGE_BIAS`
    are from the confounded fit and are **not recommended for production**.
    Two follow-ups, not mutually exclusive:

    - **2c-i, DONE — option B, hand-tunable distance cap.** Added
      `max_merge_distance` (`peaks.py`, `cli/`): forces a split when two
      adjacent summits are at least this many bp apart, regardless of valley
      depth or `split_merge_rule` — one interpretable knob, no proxy-label
      fitting, sweepable the same proven way `seed_score`/`boundary_fraction`
      were (item 2b). **Benchmarked 2026-10-15 — not worth a default.**
      `scripts/benchmark/sweep_max_merge_distance.sh` on top of
      `seed_score=0.3`, against GM12878 groHMM+DNase plus G7 vs. both K562
      PRO-cap and K562 groHMM+DNase:

      | sample / truth | cap | n peaks | bin P | bin R | bin F1 | peak sens. | peak PPV |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | GM12878 / groHMM | none | 71,770 | 0.1985 | 0.7110 | 0.3103 | 0.7986 | 0.3465 |
      | GM12878 / groHMM | 200 | 88,125 | 0.2090 | 0.6751 | 0.3191 | 0.7927 | 0.3219 |
      | GM12878 / groHMM | 500 | 76,964 | 0.2076 | 0.6875 | 0.3189 | 0.7878 | 0.3358 |
      | G7 / groHMM | none | 35,864 | 0.2471 | 0.6406 | 0.3566 | 0.6711 | 0.5228 |
      | G7 / groHMM | 200 | 42,273 | 0.2687 | 0.6384 | 0.3782 | 0.6719 | 0.4489 |
      | G7 / PRO-cap | none | 35,864 | 0.5178 | 0.5784 | 0.5464 | 0.5493 | 0.6680 |
      | G7 / PRO-cap | 200 | 42,273 | 0.5370 | 0.5496 | 0.5432 | 0.5434 | 0.6725 |

      1000/2000bp are effectively no-ops (within a few peaks of `none`).
      200/500bp split more, trading recall/peak sensitivity/PPV for a little
      bin precision — away from dREG's recall (0.852), not toward it. And
      the confound concern was real: G7's F1 *gain* at 200bp against
      groHMM+DNase region truth (0.357→0.378) vanishes against PRO-cap
      (0.546→0.543), i.e. the apparent improvement was matching the region
      truth's own merge radius. Merge distance is not the recall bottleneck;
      keep the option, don't change defaults. (Tracks: pre-Ampere bf16 —
      see the bf16 note under item 5; conclusions are within-track
      comparisons so unaffected.)
    - **2c-ii, READY (promising G7 smoke test, awaiting a real G1-G6 run) —
      option A, fix the label source.** Turned out to need no new labeling
      *logic* — `_label_pair`'s existing rule (both summits in one truth
      interval → merge, spanning two distinct intervals → split) already
      does the right thing, since the goal is to keep bidirectional
      promoters as one span, not split their two internal summits. The fix
      is entirely about *which* truth-interval file: swapped
      `K562.positive.bed.gz` (groHMM+DNase — gene-body/accessibility-level,
      not initiation-precise) for `data/ENCSR220XSM_peaks.hg19.bed` (K562
      PRO-cap, ENCODE ENCSR220XSM, bidirectional + unidirectional peaks,
      already downloaded/lifted-over by `scripts/data/download_peaks.sh` —
      no new data-sourcing code needed). `fit_split_merge_model.py`'s
      `--tss_bed` default now points here; the old file still works via an
      explicit `--tss_bed` override.

      G7-only smoke test (not the real validation, but a strong signal):
      excluded-ambiguous dropped from 95.6% to 40.7%; `dist` now shows real
      overlap between merge/split (merge ∈ [32,672]bp, split ∈ [32,1696]bp —
      no manufactured gap); the distance-matched-band control actually ran
      this time (4,644 pairs) with held-out F1 staying at 0.90–0.92 inside
      it; the shape-only ablation (no `dist`/`r1`/`r2`) scored comparably
      (0.84–0.93); the winner was a 100-tree RandomForest (F1=0.93,
      log_loss=0.24) — no degenerate single-threshold collapse, no
      suspicious 0.0 log-loss artifact. Next step: re-run on the full G1-G6
      set and re-check the same signals before refitting and reconsidering
      `split_merge_rule="learned"` for production.

      **Full G1-G6 run done (2026-10-15; G1/G2/G3/G5/G6 — no G4 — dense
      native-bf16 Ampere tracks): the smoke-test signals held.** 60,603
      pairs, 56.1% excluded as ambiguous (was 96.2% with groHMM labels);
      26,598 kept (25,723 train / 875 held-out on chr21/22; ~71% split).
      `dist` overlaps between classes (merge 32-768bp, split 32-3840bp). The
      distance-matched band has 23,998 pairs, held-out F1 0.89-0.90. Shape-only
      ablation: F1 0.84 (LR) / 0.91 (tree, RF). Held-out, all features:

      | Model | P | R | F1 | log-loss |
      | --- | --- | --- | --- | --- |
      | LogisticRegression | 0.919 | 0.886 | 0.902 | 0.277 |
      | DecisionTree(depth=3) | 0.918 | 0.906 | 0.912 | 0.271 |
      | RandomForest(100, depth 6) — winner | 0.917 | 0.911 | 0.914 | 0.268 |

      The depth-3 tree scores the same F1 with or without `dist`/`r1`/`r2`
      (0.9118 both), so it likely uses only valley-shape features
      (`d2`/`d3`/`dr`) — possibly close to a re-tuned `valley_fraction`.
      Inspect it before assuming it's new information. **Gap:** runtime
      `_predict_split` only evaluates logistic weights; the RF export
      (`_SPLIT_MERGE_FOREST`, ~100 trees) has no runtime evaluator. These
      are proxy-label scores; the real test is peak-level benchmarking on
      GM12878, not yet run. G7 vs K562 PRO-cap can't evaluate the learned
      rule fairly (same truth file as the labels).

      **TROGDOR vs. pydreg, same harness (2026-10-15, `tmp/vs_dreg/`).** At
      `min_score=0.95` the two callers make nearly matched call counts
      (GM12878 70,339 vs 71,448; G7 35,054 vs 35,048):

      | sample / truth | caller | bin P | bin R | bin F1 | peak sens. | peak PPV | center sens. | center spec. |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | GM12878 / groHMM | TROGDOR | 0.2225 | 0.6701 | 0.3341 | 0.7759 | 0.3502 | 0.7223 | 0.3405 |
      | GM12878 / groHMM | dREG | 0.1788 | 0.8516 | 0.2955 | 0.8701 | 0.4337 | 0.8501 | 0.4481 |
      | G7 / K562 PRO-cap | TROGDOR | 0.5583 | 0.5340 | 0.5459 | 0.5246 | 0.6751 | 0.5086 | 0.6766 |
      | G7 / K562 PRO-cap | dREG | 0.4702 | 0.7646 | 0.5823 | 0.6050 | 0.8255 | 0.5743 | 0.8410 |
      | G7 / K562 groHMM | TROGDOR | 0.2769 | 0.6148 | 0.3818 | 0.6580 | 0.5247 | 0.6317 | 0.5050 |
      | G7 / K562 groHMM | dREG | 0.1593 | 0.6011 | 0.2519 | 0.5981 | 0.4872 | 0.5937 | 0.4900 |

      Caveats: (1) Training splits, per the 2019 dREG paper (Wang et al.,
      Genome Res 29:293, doi:10.1101/gr.238279.118). dREG trained on five
      K562 PRO/GRO-seq sets, with G7 (K562) and GM12878 held out. Its
      positives are K562 transcribed DHSs, i.e. Duke ∩ UW DHSs ∩ GRO-cap HMM
      calls. Note: the paper's positives come from *GRO-cap* HMM calls, not
      groHMM, so "groHMM+DNase" in these docs is likely a misnomer for the
      `*.positive.bed.gz` sets. TROGDOR follows the same split: trained on
      `K562.positive.bed.gz` from G1/G2/G3/G5/G6, with G7 and GM12878 held out.
      So GM12878 is a held-out cell type for *both* callers, and the
      GM12878/positive rows are a fair head-to-head. G7/K562-positive is a
      held-out sample scored against both callers' training-label
      definition, which is also level. G7/K562 PRO-cap is independent of the
      labels entirely. (2) dREG peaks are ~2x wider (median 430-450bp vs.
      224-240bp, 1.6x total bp), which inflates ≥1bp-overlap sensitivity/PPV
      and bin recall and deflates bin precision. On the independent truth
      (G7/PRO-cap), dREG still leads on every metric except bin precision,
      including the center-window ones. Next: a width-controlled comparison
      (fixed ±W windows around TROGDOR summits and dREG centers, at matched
      counts by rank), scored against independent truth. That settles
      whether the gap is peak geometry or the scores themselves.

      **Done — the gap is not peak geometry (2026-10-15, `tmp/width_ctrl/`,
      `tmp/fp32/`).** Two checks:

      - *Width-controlled peaks:* top-N calls by each caller's own score, with
        identical ±100/±200bp windows around TROGDOR summits and dREG
        centres. dREG wins every metric on GM12878 and on G7/PRO-cap at
        every N and W, e.g. GM12878 N=70k W=200: peak sens. 0.835 vs 0.707,
        PPV 0.476 vs 0.346; G7/PRO-cap N=35k W=200: F1 0.570 vs 0.476, PPV
        0.852 vs 0.681. TROGDOR wins only on G7/K562.positive.
      - *Like-for-like score separation:* both callers scored on dREG's own
        informative positions (`*.dREG.infp.bw`), with the same labels and
        the same shuffled-null placements:

        | truth | site AUPRC TROGDOR / dREG | site AUROC | peak AUROC |
        | --- | --- | --- | --- |
        | G7 / K562.positive | 0.517 / 0.322 | 0.952 / 0.920 | 0.954 / 0.934 |
        | GM12878 / positive | 0.414 / 0.396 | 0.901 / 0.931 | 0.953 / 0.966 |
        | G7 / K562 PRO-cap | 0.681 / 0.751 | 0.916 / 0.954 | 0.915 / 0.924 |

      So TROGDOR's large separation advantage is specific to K562.positive,
      the label definition it was trained on (scored on held-out K562 sample
      G7). On the held-out cell type it's roughly a tie, and on independent
      PRO-cap truth dREG is ahead, though dREG trained on the same K562
      labels. TROGDOR appears to fit its training label definition more
      tightly but generalize less well. Caveats: sites are chosen by dREG's
      heuristic (densified within dREG candidates); TROGDOR's summit may sit
      on one TSS of a divergent pair, not the NFR centre.

      **bf16 output quantization checked, not the cause.** Under bf16
      autocast the 1x1 head emits bf16 logits, so only 206 distinct
      probabilities occur among G7 chr1 bins ≥0.95 (fp32: 59,687). Re-running
      both checks on dense fp32 tracks changed TROGDOR's numbers by ≤0.005
      and no conclusion. The coarse grid still makes ties (ranking, FDR
      steps). A prototype that runs only the 1x1 head in fp32 under bf16
      autocast (G7 chr1, L40S) gives continuous scores at bf16 speed: 7.5s
      and 58,730 distinct values among bins ≥0.95, vs bf16 8.7s / 206 and
      fp32 14.8s / 59,684. Values still differ from fp32 because the body
      stays bf16. Candidate small follow-up PR to main.

      Implication: further peak-geometry tuning can't close this gap. It's in
      how well TROGDOR's scores generalize beyond the K562 label definition,
      which points to `docs/training_improvement_plan.md` (e.g.
      label sources, cross-cell-type validation) rather than the caller.

3. **Informative-site pre-filtering at candidate-seeding time — still
   demoted, likely low-value**, for the reason already established: the
   prior `scripts/benchmark/infp_filter.py` experiment (masking a dense
   `prob.bw` before the legacy external-truth `trogdor fdr`; see
   `scripts/benchmark/_results/fdr.txt`, output
   `GM12878.trogdor.infp.groHMM.fdr.pdf`) flattened the null to zero without
   fixing anything that was actually broken. Coverage was never the axis of
   ambiguity in any of the self-referential-calibration findings above.

4. **Centroid reporting + two-pass sparse-then-dense scoring** — lower
   priority, unrelated to calibration. Output richness
   (probability-weighted centroid alongside the summit) and
   summit-localization precision (dREG scores informative sites sparsely
   first, then densifies inside promising regions).

5. **Decoy-input empirical FDR — proposed (2026-10-15), deferred until the
   2c-i sweep and 2c-ii G1-G6 refit are done.** A third way out of item 1's
   dead end, not on that list: both failed nulls resampled the *same score
   track* the peaks came from. Instead, run the full `score` → `peaks`
   pipeline on input with initiation structure destroyed, and estimate
   `FDR(t) ≈ N_decoy_calls(t) / N_real_calls(t)` (target-decoy, as in
   proteomics; cf. MACS's treatment/control swap). Decoy calls must clear
   the same `min_score` through the same caller, so the item-1 tautology
   can't arise, and the result is an FDR on TROGDOR's *own* call set — which
   `trogdor fdr` against external truth doesn't provide.

   The crux is the decoy: it must remove TIR structure while preserving the
   coverage distribution. Strand-swapping/mirroring is too gentle (still
   valid TIR signal; the model trains with strand-flip augmentation); a
   genome-wide read shuffle is too harsh (reproduces the trivially-easy
   genome null). Candidates in between: local read shuffling within
   ~1-10kb windows (preserves local depth), or offsetting plus vs. minus
   strand by several kb (breaks divergent pairing). The resulting FDR is
   only "relative to that decoy", so it must be validated: decoy calls
   should be depleted at PRO-cap TSSs, and decoy FDR should roughly agree
   with external-truth `trogdor fdr` on K562. Plan: offline prototype in
   `scripts/benchmark/` on G7 + GM12878 with 2 decoy types (one extra
   scoring pass per sample per decoy); promote to a CLI flag only if it
   agrees with the external-truth FDR.

**Benchmarking note — bf16 scores depend on the GPU (2026-10-15).**
`trogdor score` defaults to bf16 autocast (`predict.py` `dtype="auto"`,
`torch.cuda.is_bf16_supported()`, which counts emulation, so pre-Ampere GPUs
run emulated bf16). On G7 chr1, vs. native bf16 on an L40S: fp32 moves
1,607/64,457 (2.5%) of `>=0.95` bins across 0.95 and changes ~15% of profile
peak coordinates; emulated bf16 on a TITAN Xp moves 1,859 (2.9%). fp32 is
hardware-consistent (TITAN Xp vs. L40S: max |Δp| 4e-4, 6 bins crossing).
Speed: native bf16 is 1.8x faster than fp32 on Ampere+ (8.2 vs 14.8s), while
emulation costs nothing on Pascal (108 vs 109s — the TITAN Xp is just ~13x
slower). **Decision: keep bf16 as the default for the speed.** Consequence
for benchmarks: generate every track that gets compared or fit on together
with the same hardware class — Ampere+ (`-C "GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"`).

## Primary References Checked

- Danko-Lab dREG GitHub repository and `dREG/R/peak_calling.R`.
- dREG README, including recommended peak calling and legacy thresholded
  bedGraph workflow.
- PINTS GitHub repository, README, and `pints/calling_engine.py`.
- PINTS Nature Biotechnology paper page:
  https://www.nature.com/articles/s41587-022-01211-7
