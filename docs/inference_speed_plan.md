# Scoring-Speed Plan

## Status (2026-07-22)

**Levers 1 and 2 as originally scoped are dead. A revised lever 1 is
proposed below.** Four levers were identified while auditing `predict.py`
and the CLI argument defaults for `trogdor score`/`trogdor pipeline`. All
are inference-time only (no retraining, no change to model weights).

**Platform note:** initial validation happened on macOS (CPU/MPS); the
deployment target is Linux + NVIDIA (confirmed: A100). Correctness checks
(does changing a knob change the output) are platform-independent.
Throughput conclusions are not, and macOS numbers turned out to be
misleading for lever 2 specifically — see below.

## Levers, ranked

**1. ~~Shrink `overlap`~~ — dead as scoped; revised to "grow `chunk_size`
instead" (same goal, doesn't touch the risky part).**

Original idea: `overlap` (32768bp default) is ~10x the model's receptive
field (~3-3.5kb, per `training_improvement_plan.md` item 4), and
`stride = chunk_size - 2*overlap` means only 75% of each chunk's compute is
retained — shrinking `overlap` to ~4096 should recover most of the other
25%.

**Why it's dead:** validated bit-identical in float32 on macOS
(`max_abs_diff=0`, `corr=1.0`), but the production path always runs bf16 on
CUDA (`dtype="auto"` in `predict_chromosome`/`predict_genome`, no `--dtype`
CLI flag exists to override it — confirmed by grepping `cli/commands.py`/
`cli/main.py`). On the real GM12878 A100 run, `overlap=4096` vs. `32768`
gave jaccard=0.968 on the resulting peak calls — and a same-config rerun of
the default came back at exactly jaccard=1, which rules out ordinary bf16/
cuDNN run-to-run nondeterminism as the explanation. The difference is a
real, reproducible function of `overlap` under bf16: bf16's coarser mantissa
means content near the nominal receptive-field edge carries more rounding
error than in float32, and with `pos_weight=500` training producing sharp
probability transitions (`training_improvement_plan.md` item 8), that
rounding is enough to flip discrete threshold decisions (`min_score`,
`seed_score`, `valley_fraction`) in the profile caller for a meaningful
fraction of peak territory. Forcing float32 to make this safe isn't a real
option — bf16 is "a massive gain" over float32, per direct A100
measurement, so trading it away to shrink `overlap` would very likely lose
more than it gains.

**Revised approach — grow `chunk_size` instead of shrinking `overlap`.**
The actual goal was reducing the 25% wasted-compute fraction
(`2*overlap/chunk_size`), not specifically shrinking `overlap`. Holding
`overlap=32768` fixed (unchanged margin → unchanged bf16 rounding behavior
near the boundary — nothing about the risk above applies) and doubling
`chunk_size` to 524288 drops the wasted fraction from 25% to 12.5%; doubling
again to 1048576 drops it to 6.25%. This should be safe from the
correctness angle since it doesn't touch the boundary-margin size that
caused the bf16 sensitivity — but it does increase per-chunk memory
(activations at the outer encoder's near-full-length layers scale with
chunk length), so it needs its own check: sweep `chunk_size` ∈ {262144,
524288, 1048576} at fixed `overlap=32768` on the A100, watching both
wall-clock and VRAM headroom, and probably re-check `batch_size` at each
`chunk_size` since the two interact (bigger chunks may need a smaller batch
to fit).

**2. ~~Raise `batch_size` CLI default from 8 to 64~~ — dead, confirmed on
three backends.**

Original idea: `predict_chromosome`/`predict_genome` default to
`batch_size=64` in their function signatures, but the CLI hardcodes `8`
(matching a stale docstring, not the real signature default) — hypothesized
this undersold GPU utilization.

**Confirmed wrong in the same direction on every backend tested:**

| device | batch_size=8 | batch_size=64 | ratio |
| --- | --- | --- | --- |
| CPU (macOS) | 6.55s | 219.80s | ~33x slower |
| MPS (macOS) | 4.06s | 4.83s | ~1.2x slower |
| A100 (target, full GM12878 pipeline) | 2m40s | 3m17s | ~1.23x slower |

Correctness is unaffected (eval-mode, BatchNorm uses running stats, output
is batch-composition-independent — confirmed on macOS). But there is no
remaining hypothesis under which raising the CLI default to 64 helps; don't
pursue this further as scoped. If there's appetite for it, the open
question left is whether something *below* 8 is even better on the A100
(untested direction), but this is a minor optimization at best, not the
"free win" it was expected to be.

**3. `torch.compile(model)` — still open, now relatively higher priority.**

The model is a static-shape conv U-Net with no data-dependent control flow —
compile-friendly. Not yet evaluated on any backend. With levers 1/2 mostly
closed out, this and the `chunk_size` sweep above are the main remaining
throughput candidates. Needs the A100 specifically (macOS can't meaningfully
exercise CUDA compile backends).

**4. `num_workers` CLI default (4) vs. `CLAUDE.md`'s documented guidance**
("default 0 ... leave at 0 on macOS for fork-safety") — still an open
doc/code mismatch, low priority. Target platform is Linux, where
`num_workers>0` is expected to be fine regardless, so this is a
documentation correctness issue more than a throughput lever.

## Validation results

**macOS (CPU + MPS), `dtype=float32` forced, `chr21` slice of
`tmp/trogdor/G7.{pl,mn}.bw`, cached HF checkpoint:**

- `overlap` 32768 vs. 4096: `max_abs_diff=0`, `corr=1.0` on both backends,
  ~1.45–1.48x speedup. (Later found not to transfer to the real bf16
  production path — see lever 1 above.)
- `batch_size` 8 vs. 64: `max_abs_diff=0`, `corr=1.0` on both backends, but
  64 was ~33x slower on CPU and ~1.2x slower on MPS.

**A100 (target hardware), real GM12878 groseq data, full `trogdor pipeline`
(default bf16, no dtype override available):**

- `overlap` 32768 (default) vs. 4096: jaccard=0.968 on called peaks — not
  bit-identical, and NOT run-to-run noise (a same-config rerun of the
  default overlap reproduced jaccard=1.0 exactly against itself). Real,
  reproducible effect of `overlap` under bf16. Do not ship this change.
- `batch_size` 8 vs. 64: 2m40s vs. 3m17s — 64 confirmed slower, consistent
  with both macOS backends.

## Next steps

1. Sweep `chunk_size` ∈ {262144, 524288, 1048576} at fixed `overlap=32768`
   on the A100 (the revised, safer version of lever 1) — check wall-clock,
   VRAM headroom, and re-verify peak-call jaccard against the current
   default stays at 1.0 (expected, since the margin itself is unchanged).
2. Evaluate `torch.compile(model)` on the A100.
3. Resolve the `num_workers` CLI-default-vs-`CLAUDE.md` mismatch (doc fix or
   code fix, whichever is actually correct).
4. Fix the stale "Default is 8" docstring text in `predict.py` regardless of
   what default is ultimately chosen — it already disagrees with the
   function's own signature default of 64, independent of this
   investigation.
5. Lower priority, if still curious: check whether `batch_size` below 8 has
   further gains on the A100 — direction untested, unlikely to be large.
