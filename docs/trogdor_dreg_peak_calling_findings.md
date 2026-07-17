# TROGDOR Peak-Calling Findings and dREG Comparison

## Summary

TROGDOR appears strong as a scoring model but weaker as a peak caller. The main
failure mode is not simply that TROGDOR assigns low scores to true initiation
regions. Rather, TROGDOR emits highly spiky probability profiles, and the
original caller converted those profiles into BED intervals with a simple
threshold-and-merge rule. That rule is too blunt for peak-level evaluation:
small score spikes can become many narrow calls, while permissive thresholds can
produce noisy candidate sets whose geometry is poorly matched to annotations.

dREG provides a useful contrast. The current recommended dREG workflow is not a
plain thresholded score track. It uses a multi-stage peak-calling procedure that
builds broad candidates, densifies promising regions, smooths local score
profiles, detects local maxima, decides whether adjacent maxima should split or
merge, and then filters called peaks with a probability-like statistic plus
multiple-testing correction. TROGDOR should borrow the geometric ideas, but not
dREG's parametric significance model, because TROGDOR logits/probabilities do
not follow a known parametric null distribution.

## Observed Benchmark Pattern

The confusing result was that TROGDOR looked much better than dREG in some FDR
or ranking-style plots, but worse in peak-level benchmark summaries. The
available peak benchmarks show why that can happen:

| Comparison | TROGDOR pattern | dREG pattern |
| --- | --- | --- |
| K562 vs groHMM + DNase | Higher recall/sensitivity than dREG, but lower PPV | Lower recall, slightly higher PPV |
| GM12878 | Many more candidate peaks than dREG and lower PPV | Fewer candidates, higher peak-level sensitivity and PPV |
| SCREEN-style broad truth | dREG outperforms the PINTS/TROGDOR-style call set in both recall and PPV | Better-shaped/filtered calls |

Representative values from `scripts/benchmark/_results/peak_benchmarks.txt`:

- K562, groHMM + DNase: TROGDOR sensitivity 0.6691 and PPV 0.4514; dREG
  sensitivity 0.5980 and PPV 0.4871.
- GM12878: TROGDOR calls 86,938 candidates with PPV 0.3129; dREG calls 71,411
  candidates with PPV 0.4339.
- K562, ENCODE SCREEN peaks: dREG sensitivity 0.5050 and PPV 0.8959.

This is consistent with a model/caller mismatch. TROGDOR can rank likely
regulatory positions well, but a simple thresholded BED can inflate call count,
fragment true sites, or merge/shape intervals in ways that hurt peak-level
metrics.

## What dREG Does Differently

The public dREG README says the recommended solution is the integrated peak
caller, which directly generates narrow peaks with peak position, max score,
probability, and center. The older legacy path thresholded dREG score tracks
into broad peaks and then required dREG-HD-style refinement.

From `Danko-Lab/dREG/dREG/R/peak_calling.R`, the recommended caller does several
things that TROGDOR's historical threshold caller did not:

- Scores informative positions, then fills gaps between nearby high-scoring
  informative sites.
- Builds broad candidate regions by expanding scored sites and merging nearby
  blocks.
- Densifies scoring inside promising broad candidates.
- Smooths local score profiles and detects local maxima/valleys.
- Uses a random forest helper to decide how to split or merge adjacent local
  maxima.
- Reports raw peaks with score, probability-like value, smoothed/original mode,
  and centroid.
- Applies multiple-testing correction before emitting significant peak calls.

Important implementation details in dREG that motivated TROGDOR changes:

- dREG's broad candidate stage uses a much looser geometry than "only abutting
  bins merge."
- dREG treats summit/center information as a first-class peak output.
- dREG's peak caller explicitly reasons about local maxima and valleys.
- dREG separates candidate geometry from statistical filtering.

## Why TROGDOR Should Not Copy dREG Exactly

dREG's statistical layer is model-specific. Its source estimates a Laplace-like
null from dREG predictions and computes probability-like values over local
score vectors. That is not justified for TROGDOR:

- TROGDOR emits neural model probabilities/logits, not dREG SVR scores.
- TROGDOR score distributions are highly spiky and do not have a known
  parametric null.
- Applying Gaussian, Laplace, Poisson, ZIP, or related parametric p-values to
  TROGDOR logits would create false precision.

The safe TROGDOR analogue is empirical calibration: compare real candidate
summit scores against a shuffled/null set generated with the same calling and
scoring rules.

## Consequences of Spiky TROGDOR Probability Profiles

The example plots show sparse, sharp score spikes rather than smooth peak
bodies. That changes how the caller should behave:

- Peak calling should be summit-first, not broad-body-first.
- `seed_score` should collect candidate context around possible summits, but
  should not be treated as calibrated confidence.
- `max_gap` should be small and interpreted in output bins; large gaps can
  stitch together unrelated spikes.
- Smoothing should be light and used only for local-maxima stability.
- Valley-based splitting is important because adjacent spikes often represent
  separate local candidates.
- Boundary trimming should be optional and benchmarked; it can help remove low
  shoulders but can also over-shrink calls.
- A high `min_width` is risky because true predictions may be one-to-few output
  bins wide.
- `max_width` is more defensible as a guardrail against accidental broad merged
  candidates.

## Resulting TROGDOR Design Direction

The implemented direction is a profile/summit-aware caller plus streamed
empirical calibration:

1. Stream per-chromosome TROGDOR probabilities from the model.
2. Form candidate intervals from bins above a candidate threshold:
   - `seed_score` for profile mode when set.
   - otherwise `min_score`.
3. Call raw peaks from the candidate profile:
   - preserve summit start/end/score internally for all modes;
   - report summit columns for refined/profile modes;
   - keep historical four-column BED output for simple mode.
4. Write raw peaks and calibrated peaks separately from the one-step pipeline.
5. Calibrate empirically without writing dense score bigWigs:
   - default statistic: `smoothed_summit`;
   - default null scope: thresholded candidate intervals;
   - optional null scope: whole chromosome/genome;
   - optional statistics: raw `summit`, interval `max`, or interval `mean`.
6. Select calibrated peaks by empirical FDR threshold, not by a parametric
   p-value.

The default calibration statistic should be `smoothed_summit`, because the
new example tracks show that raw TROGDOR probabilities can be dominated by
single-bin spikes. A raw `summit` max is useful for diagnostics, but it rewards
isolated spikes in both real and null placements. `smoothed_summit` instead
averages a small summit-centered window, so calibrated calls require local
support around the maximum while still preserving summit-focused peak identity.
For `summit` and `smoothed_summit` calibration, the null should shuffle
summit-sized windows, not full peak bodies. When users choose `max` or `mean`,
the null should instead shuffle full peak intervals and score those interval
bodies.

## Candidate-Region Null vs Whole-Genome Null

The default empirical null should be candidate-region based:

- `candidate` null asks whether a summit is strong relative to other
  thresholded TROGDOR-like candidate regions.
- `genome` null asks whether the score is strong relative to the whole genome,
  which is often dominated by low-score background.

Because the caller has already conditioned on candidate bins, the candidate
null is the more conservative and relevant default for calibrated peak calling.
The genome null remains useful as a broader diagnostic.

## Practical Recommendations

For benchmarking and production use:

- Prefer the one-step streamed pipeline for profile calling and calibration, so
  low candidate thresholds do not require dense score bigWigs.
- Treat `max_gap`, `smooth_bins`, `valley_fraction`, and `boundary_fraction` as
  tuning knobs, not canonical constants.
- Evaluate raw and calibrated BEDs separately.
- Report summit-centered metrics in addition to interval-overlap metrics.
- Tune hyperparameters on held-out annotations using peak count, summit-window
  recall/precision, interval PPV, and empirical FDR curves.
- Avoid any TROGDOR p-value unless it is explicitly empirical/non-parametric.

## Implementation Status (2026-07-17)

The design direction below has been implemented, not just proposed:

- `src/chiaroscuro/peaks.py` (`call_profile_peaks`, `resolve_seed_score`) —
  seed-threshold candidate blocks, light smoothing, local-maxima/valley
  splitting, optional boundary trimming, `min_width`/`max_width` guardrails.
- `src/chiaroscuro/calibration.py` + `src/chiaroscuro/stats.py` — empirical,
  non-parametric calibration: `summit`/`smoothed_summit`/`max`/`mean`
  statistics, `candidate`/`genome` null scopes, and a `quantile` (default,
  tail-enriched)/`linear`/`logit`/`unique` threshold grid for `compute_fdr`.
- `--calibrate` is wired into both `trogdor pipeline` (streamed from the
  model) and `trogdor peaks` (reads directly from a saved bigWig, so
  self-calibration sweeps don't require re-scoring).
- The nested-overlap bug in `scripts/benchmark/compare_peaks.py` /
  `truth_panel.py` (unmerged subject intervals inflating coverage fractions)
  is fixed, with regression tests in `tests/test_benchmark_interval_helpers.py`.

What has **not** been re-established: the peak-level benchmark numbers in
`scripts/benchmark/_results/peak_benchmarks.txt` predate all of the above —
they still reflect the old threshold-and-merge caller. Re-running that
benchmark with `--mode profile` (+ a calibrated FDR target) against
dREG/PINTS/groHMM truth remains the outstanding validation step.

## Post-Implementation Calibration Findings (2026-07-17)

Running `--calibrate` end-to-end on real data (G7, GM12878 GRO-seq) surfaced
a second-order problem beyond peak geometry: **self-referential empirical
calibration is itself hard to tune, because TROGDOR's own probability output
is not smoothly graded.**

`scripts/benchmark/logit_dist.py` on dense (`--min_score 0`) probability
tracks for both samples shows the same shape: a dominant, very narrow
background mode around logit ≈ -3 to -4 (p ≈ 0.02–0.05) holding roughly
55–60% of all mass, plus a long, thin tail toward saturation — only the top
~5% of bins (q95) exceed p = 0.5. The distribution is closer to a couple of
sharp point-masses than a continuum, which is the mechanism behind the
"spiky and uniform/well-calibrated" difficulty: once you condition on
"candidate" (anything above a permissive threshold), most of what remains
looks comparably extreme to the model, leaving little smooth gradient for a
same-population empirical null to rank against.

This produces a three-way trade-off, none of whose corners give a usable
default (all runs: `--mode profile --seed_score 0.5 --calibration_stat
smoothed_summit --calibration_smooth_bins 5 --calibration_fdr_target 0.05`):

| `min_score` | `null_scope` | raw peaks (G7 / GM12878) | calibrated | failure mode |
| --- | --- | --- | --- | --- |
| 0.5 (= seed_score) | `candidate` | 229,037 / 395,688 | ~0 | null and real are drawn from the same overly permissive population; empirical FDR floors around 13-16% and never reaches 0.05 |
| 0.95 | `genome` | 35,058 / 70,341 | 100% / 100% | genome background is such an easy bar that literally every raw candidate passes trivially — no discrimination within the candidate set |
| 0.95 | `candidate` | 35,058 / 70,341 | 26% / 8.5% | now discriminates, but the FDR-vs-threshold curve is a near step-function (e.g. G7: 8,985 peaks at FDR 0.05 vs. 35,058 — the full raw set — at FDR 0.15); a strict 5% target lands right on the cliff |

Practical takeaways:

- `min_score` and `seed_score` are not interchangeable: `min_score` gates
  which local maxima are ever *emitted* as raw peaks in `call_profile_peaks`
  (any summit that never clears it is dropped before calibration sees it at
  all), while `seed_score` only shapes candidate-block geometry for
  splitting. Collapsing them to the same value (both at 0.5) reproduces the
  degenerate first row above.
- `--null_scope candidate` only becomes informative once `min_score` is
  strictly higher than `seed_score` (so "real" is a genuine strict subset of
  the broader seeded population the null draws from) — but even then, the
  FDR curve is steep enough that `--calibration_fdr_target 0.05` may be an
  unreasonably strict choice for this scoring scheme. Inspect the
  `--calibration_curve` TSV's `n_real` vs. `fdr` columns directly rather
  than assuming 0.05 is the right target; 5-15% is a defensible range to
  scan before concluding calibration has "failed."
- `--null_scope genome` is a weak negative control once conditioned on a
  strict `min_score` — useful as a sanity check ("are candidates enriched
  over naive background at all?") but not for pruning within the candidate
  set.
- Self-referential calibration should be treated as a ranking/triage tool,
  not a substitute for validating the final call set against independent
  ground truth (`trogdor fdr` against ENCODE cCREs, dREG, or groHMM calls).

## Null-Log Diagnostic Confirms the Candidate Null Is Self-Referential
(2026-07-17)

To test *why* `--null_scope candidate`'s FDR curve is a near step-function
(previous section), `--calibration_null_log` was added to log every null
draw's position and score, then joined against the raw peaks' summit
positions to measure distance-to-nearest-real-summit. Same runs as above
(`min_score=0.95`, `seed_score=0.5`, `null_scope=candidate`,
`calibration_stat=smoothed_summit`, `n_shuffle=20`):

| | median dist. to nearest real summit | frac. ≤100bp |
| --- | --- | --- |
| null draws that clear the FDR=0.05 threshold (G7 / GM12878) | 28bp / 25bp | 99.0% / 99.5% |
| background nulls (5x random sample of all draws) | 2,494bp / 1,246bp | 9.2% / 10.5% |
| real peaks' own nearest-neighbor spacing (for scale) | 4,112bp / 2,688bp | 0.0% / 0.0% |

Every null draw that survives the FDR filter is sitting within about one
output bin of an actual real summit — not "near" it in a loose sense, close
enough that it is effectively the same feature. This rules out the
"comparably-confident independent local maxima" framing from the previous
section's takeaways. The real mechanism: `--null_scope candidate`'s allowed
placement region is the union of all `>= seed_score` blocks on the
chromosome, and because background essentially never clears `seed_score`
(only ~5% of the genome does, concentrated at real spikes — see the logit
histograms above), that region isn't a broad gray zone with real peaks
scattered inside it. It's a scattered archipelago of tiny islands, each
*is* a real peak's own immediate footprint, with no unclaimed non-peak
territory between them. Shuffling a summit-sized window within that space
cannot land anywhere except on or immediately beside some real peak (often
the one it was drawn from) — there is no other kind of territory to draw
from. The null is not failing to discriminate a real peak from a genuine
alternative; under this construction it isn't an independent negative
control at all.

This changes the priority order in `docs/peak_calling_handoff.md`: a richer
multi-feature ranking (dREG's random-forest role) assumed the null
candidates were genuine, independent alternatives that a scalar score just
couldn't rank — this diagnostic says that assumption doesn't hold, so fixing
null *construction* (excluding a peak's own footprint, plus a margin, from
serving as its own null territory) now comes first. Informative-site
pre-filtering remains demoted for the separate reason given in the previous
section (coverage isn't the axis of ambiguity either way).

## Sources Checked

- Danko-Lab dREG README:
  https://github.com/Danko-Lab/dREG/blob/master/README.md
- Danko-Lab dREG peak caller:
  https://github.com/Danko-Lab/dREG/blob/master/dREG/R/peak_calling.R
- Local benchmark summary:
  `scripts/benchmark/_results/peak_benchmarks.txt`
- Local implementation/handoff context:
  `docs/peak_calling_handoff.md`
