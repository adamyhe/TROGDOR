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

1. **A richer, multi-feature split/merge and/or null-comparison decision**
   (promoted to top priority — see the informative-sites finding below for
   why), borrowing the role dREG's random forest plays rather than deciding
   on `smoothed_summit` score alone. Candidate features: valley depth,
   distance between adjacent maxima, peak width, local density of other
   candidates nearby, and raw plus/minus coverage support. This is real
   work (feature engineering plus some kind of classifier or multi-feature
   scoring rule, not just a CLI flag), but the evidence below points at it
   as the actual axis of the problem: the candidate pool is not
   contaminated by background, it's genuinely full of comparably-confident
   real local maxima, and a single scalar score can't rank among them.

2. **Informative-site pre-filtering at candidate-seeding time — demoted,
   likely low-value.** The idea was to restrict candidate seeding and null
   placement to bins with real read support (dREG's heuristic: >3 reads in
   a 100bp window on either strand, or >1 read in a 1kbp window on both
   strands), on the theory that some fraction of the `>= seed_score` pool
   is model noise in low/no-coverage regions.

   *Prior art rules this out.* `scripts/benchmark/infp_filter.py` already
   ports this exact dREG heuristic and was previously tried in the legacy
   pipeline: mask a dense `prob.bw`, then run the *external-truth* `trogdor
   fdr` (genome-wide uniform null, `--stat max`/`mean`, groHMM/SCREEN as
   truth) against the masked track (see `scripts/benchmark/_results/fdr.txt`,
   output `GM12878.trogdor.infp.groHMM.fdr.pdf`). The observed effect was
   that masking flattened the null distribution to a spike at 0 — but that
   wasn't fixing a real problem: the null was already well-separated from
   real *without* the mask (see `GM12878.trogdor.groHMM.fdr.png`). That
   means essentially none of the model's higher-scoring output sits on
   genuinely uninformative (no-coverage) positions in the first place — the
   model isn't confidently wrong about background. That's a property of the
   model's own calibration, not of the genome-wide-null pipeline stage it
   was tested in, so it should transfer: the `>= seed_score` candidate pool
   used by `--null_scope candidate` is almost certainly *also* already
   concentrated on covered positions. Filtering it by informative sites
   would likely just re-confirm that without shrinking the pool — coverage
   isn't the axis separating "real summit" from "other candidate region"
   here; both are covered. Not worth implementing unless #1 stalls and this
   gets revisited with direct evidence from the new pipeline.

3. **Centroid reporting + two-pass sparse-then-dense scoring** — lower
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
