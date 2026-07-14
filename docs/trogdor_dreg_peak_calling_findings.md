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
   - default statistic: `summit`;
   - default null scope: thresholded candidate intervals;
   - optional null scope: whole chromosome/genome;
   - optional interval statistics: `max` or `mean`.
6. Select calibrated peaks by empirical FDR threshold, not by a parametric
   p-value.

The default calibration statistic should be `summit`, because TROGDOR's signal
is spike-like and peak identity is dominated by local maxima. For summit
calibration, the null should shuffle summit-sized windows, not full peak bodies.
When users choose `max` or `mean`, the null should instead shuffle full peak
intervals and score those interval bodies.

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

## Sources Checked

- Danko-Lab dREG README:
  https://github.com/Danko-Lab/dREG/blob/master/README.md
- Danko-Lab dREG peak caller:
  https://github.com/Danko-Lab/dREG/blob/master/dREG/R/peak_calling.R
- Local benchmark summary:
  `scripts/benchmark/_results/peak_benchmarks.txt`
- Local implementation/handoff context:
  `docs/peak_calling_handoff.md`
