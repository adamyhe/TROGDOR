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

1. **Fix candidate-null placement to exclude self/near territory — new top
   priority.** `--calibration_null_log` plus a distance-to-nearest-real-summit
   join (see `docs/trogdor_dreg_peak_calling_findings.md`'s "Null-Log
   Diagnostic" section) showed that 99%+ of null draws that clear the FDR=0.05
   threshold sit within ~1 output bin (25-28bp) of an actual real summit —
   not "a comparably strong independent candidate," effectively the same
   feature. `--null_scope candidate`'s allowed placement region (the union of
   all `>= seed_score` blocks) is a scattered archipelago of tiny islands,
   each *is* a real peak's own footprint, with no unclaimed non-peak
   territory between them — so a shuffled draw can't land anywhere except on
   or beside some real peak, often the one it was drawn from. This needs a
   real fix at the null-construction level, not a scoring-scheme tweak:
   e.g. a `shuffle_peaks_within_intervals` variant (or a new `null_scope`)
   that subtracts a margin around every real peak's own footprint from the
   allowed region before placing null draws, so a draw represents a genuine
   "elsewhere" rather than a few bp from thyself. Needs scoping: how wide a
   margin, whether margin should scale with peak width or smoothing window,
   and what happens when a chromosome's candidate footprint is dominated by
   peaks (little "elsewhere" left at all).

2. **A richer, multi-feature split/merge and/or null-comparison decision —
   demoted, revisit only after #1.** Borrowing the role dREG's random
   forest plays (valley depth, distance between maxima, width, local
   candidate density, coverage support) assumed the null candidates were
   genuine independent alternatives that a scalar score just couldn't rank.
   The null-log diagnostic says that assumption doesn't hold — the null
   isn't independent in the first place under the current construction, so
   there's nothing yet to usefully rank against. Worth returning to once #1
   is fixed and null draws represent genuine alternatives.

3. **Informative-site pre-filtering at candidate-seeding time — still
   demoted, likely low-value**, for the separate reason already established:
   the prior `scripts/benchmark/infp_filter.py` experiment (masking a dense
   `prob.bw` before the legacy external-truth `trogdor fdr`; see
   `scripts/benchmark/_results/fdr.txt`, output
   `GM12878.trogdor.infp.groHMM.fdr.pdf`) flattened the null to zero without
   fixing anything that was actually broken (the null there was already
   well-separated from real without the mask — see
   `GM12878.trogdor.groHMM.fdr.png`). That means the model's higher-scoring
   output is already concentrated on genuinely covered positions — coverage
   isn't the axis of ambiguity, self-referential null geometry is (#1).

4. **Centroid reporting + two-pass sparse-then-dense scoring** — lower
   priority. These are mostly about output richness (probability-weighted
   centroid alongside the summit) and summit-localization precision
   (dREG scores informative sites sparsely first, then densifies inside
   promising regions), not directly aimed at the calibration difficulty
   above.

## Primary References Checked

- Danko-Lab dREG GitHub repository and `dREG/R/peak_calling.R`.
- dREG README, including recommended peak calling and legacy thresholded
  bedGraph workflow.
- PINTS GitHub repository, README, and `pints/calling_engine.py`.
- PINTS Nature Biotechnology paper page:
  https://www.nature.com/articles/s41587-022-01211-7
