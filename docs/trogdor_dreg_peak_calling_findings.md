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
benchmark with `--mode profile` against dREG/PINTS/groHMM truth remains the
outstanding validation step. **Update:** this has now been done for
G7/K562 groHMM+DNase truth — see "Independent-Ground-Truth Validation
Results" below. GM12878 and ENCODE SCREEN/dREG/PINTS comparisons are still
outstanding (data not present locally). Note also that this document
predates the `peak-geometry` branch split: `src/chiaroscuro/calibration.py`
and the `--calibrate` flag referenced above were subsequently stripped out
of `peak-geometry` (they remain on `codex/peak-calling`) once self-referential
calibration was found to be a structural dead end — see the "Margin-Exclusion
Fix" section below.

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

This motivated implementing exactly that fix (excluding a peak's own
footprint, plus a margin, from serving as its own null territory) — see the
next section, "Margin-Exclusion Fix Confirmed, But Exposed a Deeper,
Non-Tunable Problem," for the result: the fix worked, but revealed that
`--null_scope candidate` can't produce a graded FDR at all regardless of
tuning, which supersedes the "multi-feature ranking comes next" framing
below. Informative-site pre-filtering remains demoted for the separate
reason given in the previous
section (coverage isn't the axis of ambiguity either way).

## Margin-Exclusion Fix Confirmed, But Exposed a Deeper, Non-Tunable Problem
(2026-07-17)

`--null_exclusion_margin` (see `docs/peak_calling_handoff.md`'s next steps)
was implemented and re-run on G7/GM12878 at `margin=80`,
`min_score=0.95`/`seed_score=0.5`/`null_scope=candidate`/
`calibration_stat=smoothed_summit`. It worked mechanically: 701,160/701,160
(G7) and 1,406,820/1,406,820 (GM12878) null draws were placed with zero
chromosomes running short of "elsewhere" territory, and none of them land
adjacent to a real peak anymore (the earlier ~25-28bp self-referential
contamination is gone).

But the FDR result was 100% pass at both a moderate threshold (0.889/0.917)
and, on inspection, for a structural reason rather than a genuinely
discriminating one:

| | max null score | null quantile 0.999 | real peaks' min `summit_score` |
| --- | --- | --- | --- |
| G7 | 0.9507 | 0.9439 | 0.9504 |
| GM12878 | 0.9513 | 0.9448 | 0.9504 |

The null distribution's ceiling sits at essentially exactly `min_score`
(0.95) in both samples — 99.999% of 700K-1.4M null draws score below it.
This is not a coincidence and not fixable by choosing a different
`min_score`, `seed_score`, or margin: `--null_scope candidate`'s "elsewhere"
(the allowed region after margin-excluding called peaks) is defined as
candidate territory (`>= seed_score`) that *never produced a called peak*.
Producing a peak requires some bin in that seed block to clear `min_score`.
So any seed block that ever reaches `min_score` becomes a peak and is
removed from "elsewhere" by construction — "elsewhere" can only consist of
blocks that never reached `min_score` anywhere, capping its ceiling at
`min_score` for *any* choice of `min_score`. Real peaks are, by the same
definition, always `>= min_score`. Comparing "things defined as clearing X"
against "things defined as not clearing X" isn't an empirical question — the
answer is baked into the definitions regardless of how the null is
otherwise constructed.

Conclusion: the margin fix correctly solved the self-referential-contamination
problem it targeted, but `--null_scope candidate`'s "candidate minus called
peaks" construction cannot produce a graded, informative FDR curve no matter
how it's tuned from here — this is a structural dead end, not a parameter to
keep searching over. Two ways to get a genuinely informative comparison from
here:

1. Exclude only the peak being tested from its own null territory (not all
   peaks globally), so null draws can land on *other* real peaks. This
   breaks the ceiling tautology (other peaks legitimately clear
   `min_score`), but it stops being a strict FDR against "no signal" and
   becomes a relative-rank/triage measure ("is this peak stronger than
   typical peaks") — a different, and honestly weaker, guarantee than FDR.
2. Stop investing further in self-referential candidate-null calibration and
   rely on `trogdor fdr` against independent ground truth (ENCODE SCREEN
   cCREs, dREG, groHMM calls) as the actual quality signal instead — this
   has been available throughout and is not subject to the tautology above,
   since the comparison isn't defined in terms of the same threshold being
   tested.

`docs/peak_calling_handoff.md`'s next-steps list is updated to reflect this;
the working assumption going forward is direction 2 unless there's a
specific reason to pursue 1.

## Independent-Ground-Truth Validation Results (2026-07-17)

Following direction 2 above, ran the promoted validation path on the `peak-geometry`
branch: `trogdor peaks --mode profile` against the existing G7 (K562 celastrol
PRO-seq) probability bigWig, `trogdor fdr` against the groHMM+DNase K562
truth set (`K562.positive.bed.gz`), and `scripts/benchmark/compare_peaks.py`
against the same truth set. All runs used the surviving local assets in
`tmp/trogdor/` (`G7.trogdor.prob.bw`, `K562.positive.bed.gz`); a chrom.sizes
file was derived directly from the bigWig header rather than re-downloading
the genome. GM12878 and ENCODE SCREEN/dREG/PINTS comparisons are not
included here — those assets aren't present locally and require
re-downloading via `scripts/data/download_peaks.sh` etc. (separate, optional
follow-on work).

**`trogdor fdr` against independent truth is graded, unlike the candidate-null
tautology above.** Shuffling `K562.positive.bed.gz` genome-wide and scoring
real vs. shuffled truth peaks against `G7.trogdor.prob.bw` gives a real
FDR-vs-threshold curve: at FDR target 0.05, the threshold is 0.522 and
24,283/28,008 (86.7%) real truth peaks pass. This is qualitatively different
from the candidate-null case, where the "FDR curve" saturated at exactly
`min_score` — confirming `trogdor fdr` against independent truth is not
subject to the tautology.

**`compare_peaks.py` results reveal profile mode trades bin-level precision
for peak-level recall, and the trade is worse than it needs to be because of
a parameter footgun.** Controlled comparison, same input bigWig, same
`min_score=0.95`, same truth set, mode as the only variable:

| Config | N peaks | median width (bp) | bin P | bin R | bin F1 | bin Jaccard | peak sens. | peak PPV | mean candidate coverage | center-window sens. | center-window spec. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `simple` (`min_score=0.95`) | 41,549 | 240 | 0.2813 | 0.6292 | 0.3888 | 0.2413 | 0.6691 | 0.4514 | 0.2236 | 0.6804 | 0.4634 |
| `profile` (`seed_score=0.5`, `boundary_fraction=0.0`, the CLI default) | 35,964 | 704 | 0.1121 | 0.7086 | 0.1936 | 0.1072 | 0.7030 | 0.5443 | 0.1130 | 0.5940 | 0.4624 |
| `profile` (`boundary_fraction=0.5`) | 35,964 | 704 | *identical to `0.0` row — see below* | | | | | | | | |
| `profile` (`boundary_fraction=0.9`) | 35,964 | 352 | 0.1858 | 0.6829 | 0.2921 | 0.1710 | 0.6898 | 0.5348 | 0.1844 | 0.6211 | 0.4833 |

The `simple` row exactly reproduces the pre-existing (stale) entry in
`scripts/benchmark/_results/peak_benchmarks.txt` (41,549 peaks, identical
metrics to 4 decimal places) — a useful sanity check that nothing about the
input data or comparison harness changed.

At its CLI default (`boundary_fraction=0.0`), `profile` mode calls fewer but
much wider peaks (median 704bp vs. `simple`'s 240bp — 3.3x wider, 28.3Mb vs.
10.0Mb total footprint) than the legacy caller on identical input. This
buys real peak-level gains (higher sensitivity, higher mean GT coverage,
higher PPV — broader peaks are more likely to touch *some* truth territory)
but at a real bin-level cost: precision, F1, and Jaccard all drop by roughly
half, mean candidate coverage fraction drops from 0.224 to 0.113 (each
`profile` peak is on average half as "pure" as each `simple` peak), and
center-window sensitivity — whether the *summit* lands within 200bp of a
true feature, arguably the most direct test of profile mode's localization
claim — actually drops (0.680 → 0.594) rather than improving.

**Root cause: `boundary_fraction`'s default of `0.0` is a silent no-op, and
`0.5` is *also* a no-op for typical summit scores.** `_trim_segment` (
`peaks.py:137-144`) computes `threshold = summit_score * boundary_fraction`
and keeps every bin with raw score `>= threshold`. But every bin in a
segment already cleared the block-level `seed_score` floor during seeding
(`peaks.py:58-71`) — so trimming only removes anything once
`boundary_fraction > seed_score / summit_score`. With the default
`seed_score = min(min_score, 0.5) = 0.5` and summit scores close to 1.0 (as
they usually are for real TIRs), that threshold is close to `0.5` —
`boundary_fraction=0.5` computes `threshold ≈ 0.495`, just barely under the
`0.5` seed floor, so *nothing* fails the filter and the trimmed BED is
byte-for-byte identical to the untrimmed one. `boundary_fraction` has to be
pushed close to `1.0` (tested: `0.9`) before it does anything meaningful —
at which point it recovers about half the lost precision/F1/Jaccard and
actually improves center-window sensitivity/specificity above the
`simple`-mode baseline's untrimmed profile run, at a small cost to
peak-level sensitivity/PPV. Even at `boundary_fraction=0.9`, `profile` mode
still trails `simple` mode on bin-level precision/F1/Jaccard for this
dataset — the width/precision trade is real, not fully a tuning artifact,
but the CLI defaults are needlessly on the worst point of that trade-off
curve.

**Practical takeaway:** `--boundary_fraction 0.0` should not be treated as
"trimming disabled, safe default" — it's disabled in the same practical
sense whether it's `0.0` or `~0.5`, given `seed_score`'s default. Anyone
using `profile` mode who cares about bin-level precision or summit
localization (as opposed to broad-region recall) should raise
`--boundary_fraction` well above `0.5` — this is the first concrete,
actionable knob this validation surfaced, as distinct from the abandoned
calibration-tuning direction.

**Fixed.** `_trim_segment` now anchors its threshold at `seed_score` instead
of `0`: `threshold = seed_score + boundary_fraction * (summit_score -
seed_score)`. `boundary_fraction=0.0` remains a true no-op (every segment
bin already clears `seed_score`), `boundary_fraction=1.0` keeps only the
summit bin(s), and every value in between now does proportional trimming
regardless of the `seed_score`/summit-score gap. Regression tests:
`tests/test_peaks.py::test_profile_boundary_fraction_zero_is_true_noop`,
`::test_profile_boundary_fraction_one_keeps_only_summit`.

**Full post-fix sweep (same G7/K562 comparison), extended to find where
`profile` crosses `simple`'s numbers:**

| Config | median width (bp) | bin P | bin R | bin F1 | bin Jaccard | peak sens. | peak PPV | center sens. | center spec. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `simple` (`min_score=0.95`) | 240 | 0.2813 | 0.6292 | 0.3888 | 0.2413 | 0.6691 | 0.4514 | 0.6804 | 0.4634 |
| `profile` b=0.0 | 704 | 0.1121 | 0.7086 | 0.1936 | 0.1072 | 0.7030 | 0.5443 | 0.5940 | 0.4624 |
| `profile` b=0.5 | 512 | 0.1419 | 0.7016 | 0.2360 | 0.1338 | 0.6991 | 0.5415 | 0.6053 | 0.4712 |
| `profile` b=0.9 | 288 | 0.2269 | 0.6576 | 0.3373 | 0.2029 | 0.6780 | 0.5259 | 0.6315 | 0.4915 |
| `profile` b=0.95 | 240 | 0.2731 | 0.6211 | 0.3794 | 0.2341 | 0.6644 | 0.5157 | 0.6380 | 0.4965 |
| `profile` b=0.99 | 128 | 0.4020 | 0.4820 | 0.4384 | 0.2807 | 0.6238 | 0.4852 | 0.6434 | 0.5008 |

At `b=0.95`, `profile` matches `simple`'s median width almost exactly (240bp
vs. 240bp) and comes within ~3% of its precision/F1/Jaccard, while still
keeping a higher peak PPV and center-window specificity. At `b=0.99`,
`profile` overtakes `simple` on precision (0.402 vs. 0.281), F1 (0.438 vs.
0.389), and Jaccard (0.281 vs. 0.241) — but pays for it with recall (0.482
vs. 0.629) and mean GT coverage (0.486 vs. 0.624), and its peak PPV drops
back below `simple`'s for the first time in the sweep. So `boundary_fraction`
spans the same precision/recall trade-off space that `simple`'s single fixed
threshold occupies one point on — `profile` can be tuned to sit anywhere
along that curve, including points that strictly dominate `simple` if some
recall is an acceptable cost. There is no single "best" `boundary_fraction`
independent of what the caller is optimizing for; `min_score`/`seed_score`
retuning was not explored in this sweep and would shift the curve further.

## GM12878 Recall Gap vs. dREG: Not a Raw-Model Ceiling — `seed_score` Was Never Actually Lowered (2026-07-18)

Running `profile` mode (defaults: `seed_score=0.5`, `boundary_fraction=0.95`)
against GM12878/`GM12878.positive` (groHMM+DNase) landed at bin precision
0.2223/recall 0.6711/F1 0.3339/Jaccard 0.2004 — ahead of dREG on
precision/F1/Jaccard (dREG: 0.1784/0.2951/0.1731) but behind on recall (dREG:
0.8523) and peak-level sensitivity (0.7759 vs. dREG's 0.8700).

**First attempt to close the gap — lowering `--min_score` to `0.9`, everything
else default — was a bad trade, not a step toward dREG:** candidate peaks
grew 44% (70,341 → 101,583) for only ~1-2 points of recall/sensitivity
(0.6711→0.6818 bin recall; 0.7759→0.7980 peak sensitivity), while precision
fell hard (0.2223→0.1877, PPV 0.3502→0.2503, center-window specificity
0.3405→0.2454) — enough to fall back *below* dREG on precision/F1/Jaccard,
giving up the one advantage the default had, for essentially no recall gain.

**This looked at first like a raw-model sensitivity ceiling — it isn't.**
`scripts/benchmark/_results/benchmarks.txt`'s per-bin ROC for `TROGDOR.torch`
on this exact GM12878/`GM12878.positive` pair:

| FPR | TPR | Threshold |
| --- | --- | --- |
| 0.1% | 40.6% | 0.9919 |
| 1.0% | 78.2% | 0.8888 |
| 5.0% | 94.5% | 0.4608 |
| 10.0% | 97.0% | 0.2736 |

The `min_score=0.9` row (threshold≈0.9005) shows TPR=76.8%/FPR=0.91% in the
same table — consistent with the 0.798 peak-level sensitivity measured above,
confirming the numbers line up. But TPR keeps climbing sharply below that
threshold (94.5% at threshold≈0.46, FPR=5%) — the model has plenty of real
signal on true positive bins still to give; it is not saturated at the
thresholds `profile` mode's default is operating at.

**Root cause of why lowering `min_score` alone did almost nothing: `seed_score`
never changed.** `seed_score` defaults to `min(min_score, 0.5)`. At
`min_score=0.95` *and* `min_score=0.9`, that resolves to the same `0.5` —
candidate-block *seeding* (what region is even considered before
valley-splitting/trimming) was identical in both runs. Lowering `min_score`
only moved the final summit-acceptance gate; it cannot recover signal that
seeding never included in a candidate block to begin with. This is the same
`resolve_seed_score` behavior documented in `peaks.py`, working exactly as
designed — the mistake was tuning `min_score` while expecting it to also
move `seed_score`.

**Next thing to actually test**: explicitly lower `--seed_score` below `0.5`
(not `--min_score`) on GM12878, since the ROC table shows real signal is
available at least down to threshold≈0.46. This widens candidate blocks into
weaker-but-real territory that seeding currently excludes outright;
`boundary_fraction`/valley-splitting then decide how much of that survives as
a final call — the mechanism actually capable of moving peak-level
recall/sensitivity toward dREG's, unlike `min_score` alone. Not yet run.

**Ran it — `seed_score` sweep at 0.3/0.1/0.0, `min_score`/`boundary_fraction`
held at default:**

| seed_score | candidates | bin P | bin R | bin F1 | bin Jaccard | peak sens. | peak PPV | mean GT cov. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 (default) | 70,341 | 0.2223 | 0.6711 | 0.3339 | 0.2004 | 0.7759 | 0.3502 | 0.6643 |
| 0.3 | 70,015 | 0.2023 | 0.7102 | 0.3149 | 0.1868 | 0.7971 | 0.3547 | 0.7027 |
| 0.1 | 70,015 | 0.1842 | 0.7384 | 0.2948 | 0.1729 | 0.8143 | 0.3580 | 0.7308 |
| 0.0 | 70,015 | 0.0147 | 0.7590 | 0.0288 | 0.0146 | 0.8307 | 0.3625 | 0.7518 |
| dREG | 71,411 | 0.1784 | 0.8523 | 0.2951 | 0.1731 | 0.8700 | 0.4339 | 0.8352 |

`seed_score=0.0` is pathological, not just a further point on the curve:
precision collapses an order of magnitude (0.2223→0.0147) while the
*candidate peak count is identical* (70,015) at 0.3, 0.1, and 0.0 — same
number of discrete intervals, wildly different bin-level footprint. Since
sigmoid outputs are always `>0`, `seed_score=0.0` means every scored bin
passes seeding, which likely collapses whole chromosomes into a handful of
giant candidate blocks that valley-splitting/boundary-trimming can't cleanly
carve individual peaks back out of. Do not use `seed_score=0.0`; not
investigated further (would need to inspect the resulting BED's width
distribution to confirm the mechanism, but the direction — avoid it — is
already clear from the metrics alone).

Returns diminish, then go negative, as `seed_score` drops: 0.5→0.3 bought
+0.039 recall for −0.020 precision; 0.3→0.1 bought +0.028 recall for −0.018
precision; 0.1→0.0 bought +0.021 recall for a −0.170 precision collapse. Even
at the best-behaved aggressive point (`0.1`), peak sensitivity (0.8143) and
recall (0.7384) remain well short of dREG's (0.8700/0.8523), while F1/Jaccard
have already fallen to roughly tied with dREG (0.2948 vs. 0.2951; 0.1729 vs.
0.1731). **Conclusion: `seed_score=0.3` is the best point found on this
curve** — a real recall/sensitivity gain over default at a modest, favorable
precision cost, clearly better-behaved than the earlier `min_score=0.9`
attempt. But `seed_score` alone cannot reach dREG's recall without giving up
precision-parity entirely; a single global threshold is the wrong tool to
close the rest of this gap. dREG's extra recall most likely comes from its
per-candidate, multi-feature (valley depth, width, local density) split/merge
decision — a fundamentally richer rule than any single scalar threshold —
which is the direction to pursue next (see
`docs/peak_calling_handoff.md`'s multi-feature split/merge item).

## Sources Checked

- Danko-Lab dREG README:
  https://github.com/Danko-Lab/dREG/blob/master/README.md
- Danko-Lab dREG peak caller:
  https://github.com/Danko-Lab/dREG/blob/master/dREG/R/peak_calling.R
- Local benchmark summary:
  `scripts/benchmark/_results/peak_benchmarks.txt`
- Local implementation/handoff context:
  `docs/peak_calling_handoff.md`
