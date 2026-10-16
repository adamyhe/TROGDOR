# TROGDOR Architecture & Training Improvement Plan

## Status (2026-10-16)

**Active; the lead experiment has changed.** The peak-geometry work
(`docs/peak_calling_handoff.md` item 2c-ii) found that TROGDOR's gap to dREG
is in how its scores generalize, not in the peak caller. Like-for-like score
comparisons and width-controlled peak comparisons put TROGDOR ahead only
against K562.positive, the label definition it was trained on. On held-out
GM12878 it ties, and against independent K562 PRO-cap dREG is ahead. The
earlier PINTS-label checkpoint (item 2 below) showed the mirror image. So
the lead experiment is now **training labels and data**, laid out in the
next section. The older findings below stay valid but come after it.

## Lead plan: PRO-cap labels on reprocessed DBNascent (2026-10-16)

### Data

- **Inputs:** the DBNascent reprocessing (`$SCRATCH/dbnascent-reprocess`,
  Nextflow pipeline, validated 2026-09-30). It produces strand-split,
  point-mode, raw-count bigWigs in **hg38** with UCSC-style `chr` contigs,
  each with an empirically determined strand convention. As of 2026-10-16:
  1,354 samples in the samplesheet across ~10 organisms; 2,673 bigWigs done
  (~78 K562 files, ~8 GM12878). Keep them in `$SCRATCH` during development.
  Copy what a published checkpoint depends on to Oak once that is settled.
- **The whole project moves to hg38.** The shipped model reads coverage
  only (no sequence, no coordinates), so it scores hg38 bigWigs unchanged.
  G1-G7 and GM12878 should come from the same pipeline as everything else,
  so preprocessing differences aren't mistaken for generalization.
- **Labels:** ENCODE PRO-cap (224 released experiments, mostly Yu lab,
  natively GRCh38). Pair each input with PRO-cap from the same biosample,
  in control conditions only. Where a biosample has several PRO-cap
  experiments, pool their reads into one consensus target per biosample,
  but keep at least one K562 experiment out of the pool as evaluation
  truth. ENCODE has no GM12878 PRO-cap; for GM12878, use primary **B-cell
  PRO-cap** (ENCSR083AMN). It's the closest point-resolution truth and is
  independent of every training label. EBV transformation means some TSSs
  differ, so recall against it has a ceiling below 1.
- **Supervised training set:** QC-passing human samples with a
  matched control-condition PRO-cap biosample. Not the whole of DBNascent:
  treated or perturbed samples (initiation changes, so control PRO-cap is
  the wrong target), non-human samples (no ENCODE PRO-cap), and low-depth or
  failed-QC samples are kept for evaluation (response to perturbation,
  cross-species transfer, depth robustness). Unmatched control samples
  could later add diversity through consistency training or pseudo-labels.

### Labels: two arms, same loss family

- **Hard:** 0/1 bins inside PRO-cap bidirectional + unidirectional peaks,
  i.e. a drop-in replacement for K562.positive. This tests the label
  *source*, but still learns one peak caller's boundary and width
  conventions, and its depth-dependent choices about which weak TSSs are
  positive.
- **Soft:** per-16bp-bin PRO-cap 5′-end density from both strands, lightly
  smoothed (±1-2 bins, to absorb small offsets between labs), scaled per
  experiment to [0, 1] against a high quantile and clipped. Trained with the same
  `BCEWithLogitsLoss` (fractional targets are fine), so the output stays a
  probability and the caller, FDR tools and benchmarks are unchanged. Fix
  the smoothing width and quantile up front, not against evaluation truth.
  `pos_weight=500` was tuned for sparse hard labels, so give each arm a
  small re-tune.
- Baseline: the **shipped checkpoint as-is** on the new hg38 inputs. A
  K562.positive retrain can't be built for multi-biosample data, since that
  annotation exists only for K562. Optional control to separate the effect
  of labels from the effect of more data: a K562-only PRO-cap-label run at
  the shipped model's budget and samples.
- If soft wins, a full profile model (strand-specific TSS density with a
  shape and count loss, cherimoya-style) is the natural follow-up. It
  changes what the output means, so the caller and FDR tools would need
  rework. Defer.

### Model

- **One shared, input-conditioned model**, trained on pooled
  (input, target) pairs. No per-dataset heads: TROGDOR reads a sample and
  predicts that sample, so a new sample would have no head to use.
- **Several output-type heads are fine:** soft density (main), hard peak
  membership (auxiliary, which also gives the hard-vs-soft comparison in
  one run), and optionally strand-resolved initiation, i.e. two channels,
  for directionality.
- **No assay/biosample conditioning at first.** It would need metadata at
  inference and could let the model memorize dataset identity. Add it later
  only as an ablation, if one assay clearly lags.
- **Multiple inputs:** don't build set-pooling over replicates. Merge
  replicate bigWigs instead, and randomly downsample/sub-select
  replicates during training to cover a wide range of depths. **DNA sequence
  as a second input** is a separate, later variant. It could sharpen summit
  placement, but it makes chromosome holdouts mandatory (the model could
  then memorize loci, which voids "Confirmed/settled" #2), and it risks
  predicting sequence-plausible but inactive TSSs. Don't require other
  assays (ATAC/DNase/ChIP) as inputs; needing only one nascent-RNA run is
  TROGDOR's point.

### Handling heterogeneous datasets

- **Balanced sampling:** sample a biosample (uniformly or with a
  temperature), then an input within it, then a window, so K562 and a few
  common lines don't dominate.
- **Normalization:** targets per experiment (above). Inputs keep the
  current per-window logistic normalization at first, matching the shipped
  model. With mixed depths and assays, item 9 below (per-window vs.
  per-sample) becomes an early ablation.
- **Pair quality:** input and PRO-cap usually come from different labs.
  Start by filtering pairs (input depth, PRO-cap depth, agreement at
  annotated promoters), before trying per-pair loss weights.
- **Loader:** `NascentDataset` loads a few bigWigs whole. Hundreds need a
  window index across samples, a per-sample sampler, and cached per-sample
  normalization references. This is the first engineering blocker.

### Splits and evaluation

- Hold out **whole biosamples** (all their inputs and their PRO-cap) and at
  least one **assay class** (ChRO-seq or mNET-seq). Group replicates and
  same-study samples so a held-out biosample is fully held out. Never let a
  PRO-cap experiment be both a training target and evaluation truth.
- **Evaluation panel (hg38):**
  - held-out PRO-cap: B-cell for GM12878, the held-out K562 experiment, and
    held-out DBNascent/ENCODE biosample pairs. This is the generalization
    signal that drives decisions.
  - K562.positive and GM12878.positive, lifted to hg38 once, as standing
    benchmarks and a regression check. A model trained on new labels should
    hold these while improving on PRO-cap (the PINTS-label checkpoint
    collapsed on them). They are the shipped model's and dREG's
    training-label definition, so a win there partly means "fits that
    definition".
  - dREG (pydreg) on every row; it is fast enough.
  - Peak metrics (`compare_peaks.py`, plus the width-controlled and
    like-for-like comparisons from `peak_calling_handoff.md`) and point
    metrics (correlation with PRO-cap density, summit distance to nearest
    PRO-cap TSS).

### Order of work

1. Inventory the reprocessed samples (biosample, assay, condition, depth,
   from sample IDs/DBNascent metadata) and pair them with ENCODE PRO-cap
   biosamples. Confirm G1-G7 and GM12878 are in the reprocessing.
2. Build targets: consensus soft density and hard peaks per biosample, and
   the hg38 truth panel (liftOver K562/GM12878.positive once).
3. Re-baseline on hg38: shipped checkpoint and pydreg on the evaluation panel.
4. Multi-sample loader with balanced sampling and depth augmentation.
5. Pilot: K562 plus a handful of biosamples, hard and soft heads, matched
   against the shipped checkpoint. Then scale up.
6. Then the older items below (residual re-run, architecture ablations),
   which would otherwise be tuned to the old data and labels.

## Status (2026-07-17, superseded)

**Deferred.** This plan comes out of an architecture+training audit done
in parallel with the peak-calling geometry work on `peak-geometry`. We are
not acting on it yet — priority right now is finishing the peak-calling
analysis and merging that into `main`. Revisit this doc once that's done.

## Context

The user asked for a critical audit of the model architecture and training
approach, noting that: BCE with 500x positive weight is the current main
checkpoint's loss (not `focal_tversky_loss`, which the docs previously
implied was default/best); there's prior work on a residual architecture on
another branch (`dev-residual`); and `tmp/infp_filtering/` plots show
informative-site filtering doesn't help much.

Two parallel research agents read the architecture (`trogdor.py`,
`modules.py`, `data_transforms.py`) and the training pipeline (`losses.py`,
`dataset.py`, training loop, training scripts). Direct investigation also
covered: the `dev-residual` branch's `TROGDORResidual` architecture and its
recorded benchmark numbers, three other checkpoints' benchmark numbers
(`scripts/benchmark/_results/*.txt`), and the `tmp/infp_filtering/` plots.

The doc-cleanup and training-script-consolidation work that came out of this
audit is already done (see `scripts/train/train.py`, `CLAUDE.md`,
`scripts/README.md`, commit `5bda9af`). This doc is the substantive
follow-up: what's confirmed/settled vs. what's still open, ranked.

## Confirmed / settled (no action needed)

1. **Production loss is `BCEWithLogitsLoss(pos_weight=500)`**, applied via
   `scripts/train/train.py --loss bce` (the default). `focal_tversky_loss`/
   `tversky_loss`/`focal_loss` were tried and have not outperformed it.
   Reconciled into `CLAUDE.md`/`scripts/README.md`.
2. **The train/val split (G1/G2/G3/G5 train, G6 val, no chromosome holdout,
   same label BED for both) is intentional, not a leakage bug.** G1–G6 are
   K562 from different labs/platforms (GRO-seq vs. PRO-seq)/read depths. A
   chromosome holdout would test cross-locus generalization, which isn't the
   risk here — the model's only input is stranded coverage (no DNA sequence,
   no genomic coordinate), so it has no way to memorize locus identity in the
   first place. Do not re-propose adding one.
2a. **Cross-*cell-type* validation (e.g. swapping in GM12878 as the
   training-time val set) was considered and rejected.** What TROGDOR is
   actually modeling — the geometric relationship between active Pol II
   elongation signal (PRO-seq/GRO-seq/ChRO-seq/mNET-seq) and TSS/TIR
   location — is believed to be largely cell-type-invariant in
   humans/mammals; it's not something that needs to be relearned per cell
   type. The axis that actually varies the input signal's *shape* is assay
   characteristics (PRO-seq's nucleotide-resolution run-on labeling vs.
   GRO-seq's lower resolution vs. ChRO-seq's chromatin-based prep vs.
   mNET-seq's different capture mechanism), not cell identity. Using
   GM12878 (or any other cell type) to drive checkpoint selection/early
   stopping would also contaminate it as a clean benchmark, for a
   generalization axis that isn't expected to be the hard part. Keep
   GM12878 and any future cell-type data (Jurkat, etc.) as post-hoc
   benchmark-only truth (`benchmark.py`/`compare_peaks.py`/`truth_panel.py`),
   never as training-time validation. See "Open findings" #3 below for the
   axis worth actually validating on instead.
3. **Informative-site (`infp`) pre-filtering is confirmed low-value.** Visual
   comparison of `tmp/infp_filtering/`'s two plots (GM12878 groHMM+DNase FDR,
   filtered vs. unfiltered) shows the null collapses to near-zero either way,
   and the practically relevant point (FDR=0.01 → threshold≈0.990, similar
   recall) barely moves. Already demoted in `docs/peak_calling_handoff.md`;
   this just re-confirms it from the training side too.
4. **Training scripts consolidated.** Five near-duplicate
   `train_bce*.py`/`train_focal*.py` scripts → one `scripts/train/train.py`
   with `--loss`/`--tss_bed`/etc. `lr_search.py`'s calls to a removed
   `TROGDOR(pos_weight=...)` arg and nonexistent `.loss()` method are fixed.

## Open findings, ranked

### High priority

**1. The residual-architecture comparison is real but confounded — needs a
matched-budget re-run before drawing conclusions.**

Three checkpoints, benchmarked identically (`scripts/benchmark/_results/`):

| Truth set                          | `TROGDOR.torch` (shipped) | `TROGDOR_BCE_500_0.001` (same arch, BCE+pw500 retrain) | `TROGDORResidual_BCE_pw500...` (`dev-residual`)                                                                                       |
| ---------------------------------- | ------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| G7 vs K562.positive (groHMM+DNase) | AUROC .9665 / AUPRC .4161 | AUROC .9656 / AUPRC .4307                              | AUROC .9489 / AUPRC .4206                                                                                                             |
| G7 vs PINTS (ENCSR220XSM)          | AUROC .9693 / AUPRC .5875 | AUROC .9683 / AUPRC .5598                              | AUROC .9683 / AUPRC .5598 *(byte-identical to previous column — smells like a copy/paste in the results log; re-run before trusting)* |
| GM12878 vs GM12878.positive        | AUROC .9837 / AUPRC .4084 | AUROC .9818 / AUPRC .3843                              | AUROC .9634 / AUPRC .3677                                                                                                             |

The residual model (BatchNorm, residual blocks throughout, learned stride-2
downsampling, dilated bottleneck, engineered strand-sum/difference channels)
comes out behind the shipped model on every truth set tested.

**But**: the script most likely to have produced `TROGDOR.torch`
(`train_focaltversky.py`, since deleted/consolidated) used `MAX_EPOCHS=25`,
`EARLY_STOPPING=5`; the BCE retrains used `MAX_EPOCHS=20`,
`EARLY_STOPPING=None`. That's a real confound — training budget changed
alongside architecture/loss. (One thing that did *not* differ, correcting an
earlier over-read during the audit: window size and stride both default to
`2**18`/`2**17` in `NascentDataset` and neither script overrode them, despite
a misleading comment in the old `train_bce.py` claiming otherwise.)

Also, only 1 of the 5 variants planned in `run_residual_ablations.sh` (silu +
dilated bottleneck + strand features) was actually benchmarked. Simpler
variants (no strand features, no dilation) are untested.

**Action when we revisit**: re-run the BCE baseline and the residual
architecture with identical `--max_epochs`/`--early_stopping`/window size via
`scripts/train/train.py` (baseline) and the residual training script (still
only on `dev-residual`), then re-benchmark against all three truth sets
before concluding anything about whether residual blocks/dilated
bottleneck/strand features help.

**2. Label-source sensitivity is large — always report which truth set a
result is validated against.**

A fourth checkpoint (`TROGDOR_BCE_PINTS_500_0.001`, trained on PINTS-derived
labels instead of dREG/groHMM labels) scored AUROC .9730/AUPRC .7035 against
PINTS truth (best of any model — of course, same label geometry) but AUROC
.8990/AUPRC .2282 against groHMM+DNase truth — dramatically worse than any
dREG/groHMM-trained model. Whichever label source a model trains on, it
looks artificially strong against that same source. "BCE+500 works best" is
a conclusion validated primarily against dREG/groHMM labels — treat it as
such, not as a universal claim, and when comparing future configs evaluate
against multiple truth sets (groHMM/dREG, PINTS, SCREEN) rather than one.

**3. Cross-assay validation is the concrete next experiment for
generalization — not cross-cell-type (see "Confirmed/settled" 2a).**

The current val set (G6) only ever validates across GRO-seq/PRO-seq — 2 of
the 4 assays actually named as test-time targets (PRO-seq, GRO-seq, ChRO-seq,
mNET-seq). Since assay characteristics, not cell identity, are the axis
expected to change the input signal's shape, checkpoint selection/early
stopping should be driven by held-out-*assay* performance, not held-out-cell-
type performance.

**Action when we revisit**: swap (or rotate) the training-time validation
sample to a ChRO-seq or mNET-seq run — `scripts/data/download_test_data.sh`
already fetches Jurkat ChRO-seq/PRO-seq and K562 mNET-seq bigWigs — using
`scripts/train/train.py`'s `--tss_bed`/val-sample plumbing. `K562_mnetseq`
is the cheaper first step (same K562 biology, existing `K562.positive.bed.gz`
truth, isolates the assay-generalization question from cell-type entirely).
Separately, fold the user's additional PRO-cap/H3K27ac/DNase/ENCODE data into
the post-hoc benchmark panel (`truth_panel.py`'s multi-reference manifest) —
this is eval-only, so there's no contamination risk in using it freely,
unlike training-time validation data.

### Medium priority — cheap architecture ablations, not yet run

**4. Receptive field (~3-3.5kb) is much larger than the TIR feature scale**
(described elsewhere in this codebase as one-to-few output bins wide, i.e.
16-100bp) — driven almost entirely by `context_depth=4` stacking 256× extra
downsampling beyond `output_stride`. Worth sweeping `context_depth` ∈
{1,2,3,4} and checking AUPRC/Dice/boundary precision.

**5. `max_channels=512` exactly equals `base_channels · 2**n_out`**, so
channel width caps the instant the "context-building" half of the network
(inner encoder + bottleneck) starts, and 5 blocks/10 conv layers run at flat
width with no capacity growth as resolution shrinks — the opposite of usual
U-Net practice (double channels as you halve resolution). Consider raising
`max_channels` or adjusting the `base_channels`/`output_stride` relationship.

**6. BatchNorm's running statistics are computed on training batches that
are 7/8 TSS-centered (positive-enriched)**, but applied genome-wide
(background-dominated) at inference — a plausible train/inference
distribution mismatch specific to BatchNorm. GroupNorm/LayerNorm would be
batch-composition-independent; worth an ablation if BN is suspected to
matter here.

**7. The "outer encoder" (4 levels, run before the real U-Net) discards its
skip connections and downsamples via lossy `MaxPool1d`, so the decoder has
no high-resolution shortcut at all** — any sub-`output_stride` (16bp)
positional precision is destroyed permanently at the very first stage.
Consider a direct high-res skip, or replacing outer-encoder pooling with
something less lossy (e.g. a strided conv).

### Lower priority — hygiene/safety, not urgent

**8. No gradient clipping anywhere in `fit()`.** Cheap to add; worth it given
`pos_weight=500` amplifies gradients on rare positive bins, especially
combined with bf16 mixed precision.

**9. Per-window (262kb) adaptive normalization (`data_transforms.py
normalization()`), not per-sample/global.** Could cause within-experiment
inconsistency (same raw count normalizes differently depending which window
it falls in) and small chunk-boundary artifacts at inference. The
`min_ref=20` floor also creates a sharp regime change rather than smooth
behavior. Only worth touching if this becomes a live problem (e.g. visible
chunk-seam artifacts in scored bigWigs).

**10. No dropout/stochastic depth** in an 18M-parameter, 26-conv-layer network
— regularization currently relies solely on weight decay + early stopping
(when enabled) + data augmentation (reverse-complement flip, window jitter).
Revisit only if overfitting is actually observed.

## Explicitly not doing

- **Chromosome-based train/val holdout.** Not the risk here — the model has
  no sequence/positional input to memorize loci with. See "Confirmed /
  settled" #2.
- **Cross-cell-type training-time validation** (e.g. swapping in GM12878 as
  the val set). The Pol II-elongation-to-TSS geometry is believed to be
  cell-type-invariant; the real risk axis is assay characteristics. Swapping
  in a different cell type would also contaminate it as a benchmark for no
  real generalization gain. See "Confirmed/settled" 2a and open finding 3.

## Next steps when we revisit

1. Finish the peak-calling geometry analysis and merge `peak-geometry` into
   `main` (current priority, tracked separately).
2. Re-run the residual-vs-baseline comparison with matched training budget
   (item 1) — this is the single most informative next experiment, since
   it's the one place we already have a concrete, surprising result that
   just needs to be de-confounded rather than a fresh idea to test.
3. Run the cross-assay validation experiment (item 3), starting with
   `K562_mnetseq` as the cheapest version. Fold additional PRO-cap/H3K27ac/
   DNase/ENCODE data into the `truth_panel.py` benchmark manifest regardless
   of training-time changes.
4. Depending on those outcomes, prioritize among the architecture ablations
   (items 4–7).
5. Re-benchmark any change against all available truth sets (K562
   groHMM+DNase, PINTS, GM12878, plus whatever the expanded truth panel
   adds), not just one — per the label-sensitivity finding (item 2).
