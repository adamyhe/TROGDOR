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
   - A richer multi-feature split/merge/null-comparison decision (dREG's
     random-forest role — valley depth, width, local candidate density,
     coverage support) — but only meaningful once null draws represent
     genuine alternatives, which requires the point above first.

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

## Primary References Checked

- Danko-Lab dREG GitHub repository and `dREG/R/peak_calling.R`.
- dREG README, including recommended peak calling and legacy thresholded
  bedGraph workflow.
- PINTS GitHub repository, README, and `pints/calling_engine.py`.
- PINTS Nature Biotechnology paper page:
  https://www.nature.com/articles/s41587-022-01211-7
