# Inference-Time Accuracy Ideas

## Status (2026-07-21)

**Not started.** Logged during a scoring-speed discussion so these don't get
lost; not evaluated yet. Both are inference-only (no retraining), which
makes them cheap to try relative to the architecture/training items in
`docs/training_improvement_plan.md`.

## Ideas

**1. Test-time strand-augmentation averaging.**

Training already uses reverse-complement/strand-flip as an augmentation
(`docs/training_improvement_plan.md` item 10), but `predict_chromosome`
(`src/chiaroscuro/predict.py`) only ever runs the forward orientation at
inference. Averaging the prediction from the original input with the
prediction from the strand-swapped input (swap plus/minus channels, flip
along the length axis, un-flip the output) is a standard TTA move and should
be close to free to try: ~2x inference compute, no retraining, no CLI
default changes needed to test it offline first.

Open question: whether the model's asymmetric architecture (per
`training_improvement_plan.md`'s outer-encoder/context-depth structure)
treats the two strand channels symmetrically enough for this to help rather
than average away real signal — needs a benchmark check
(`scripts/benchmark/benchmark.py` or `benchmark_bw.py`) before considering
it as a default.

**2. Checkpoint ensembling (baseline + residual, or multi-seed).**

`training_improvement_plan.md` item 1 found the residual architecture
(`dev-residual`) loses to the shipped baseline on every truth set tested,
individually. That doesn't rule out an ensemble: averaging two models that
make different mistakes can beat either alone even when one is
individually worse. Also inference-only cost (run both models, average
probs/logits). Worth a quick check once (or instead of waiting for) the
matched-budget residual re-run in that doc's item 1 — an ensemble result
would help interpret whatever that re-run shows.

## Next steps when revisited

1. Prototype strand-averaging TTA against `benchmark.py`'s AUROC/AUPRC on
   at least one truth set (G7/K562 groHMM+DNase, to match the numbers
   already in `training_improvement_plan.md`).
2. If checkpoints for both `TROGDOR.torch` and a residual variant are
   available locally, try a simple 50/50 logit-average ensemble against
   the same truth sets used in `training_improvement_plan.md`'s item 1
   table, before/independent of the matched-budget re-run.
3. Fold whichever of these (if either helps) into
   `docs/training_improvement_plan.md`'s ranked list rather than keeping
   them as a separate doc long-term.
