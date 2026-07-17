# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TROGDOR is a deep learning method for identifying transcription initiation regions (TIRs) from nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). It takes stranded bigWig files as input and outputs a bigWig of predicted TIR probabilities and/or BED peak calls.

## Installation

```bash
uv tool install trogdor
```

For use as a dependency in a uv-managed Python project:

```bash
uv add trogdor
```

For development from source, see `CONTRIBUTING.md`.

The package installs four CLI aliases that all invoke the same entry point: `TROGDOR`, `trogdor`, `dREG`, `dreg`.

## CLI Pipeline

The tool operates in four subcommands:

1. **score** (alias: **thatch**) – Score the whole genome using the pre-trained model; outputs a bigWig of raw sigmoid probabilities (no multiple-testing correction is applied at this stage)
2. **peaks** (alias: **consummate_vs**) – Call peaks from the scored bigWig. `--mode` defaults to `profile`: seeds candidate blocks at a permissive `--seed_score` (default `min(min_score, 0.5)`), splits/merges local maxima by valley depth, and trims low-confidence shoulders via `--boundary_fraction` (default `0.95`, interpolated between `seed_score` and the local summit score — `0.0` disables trimming, `1.0` keeps only the summit bin) and reports summit columns. `--mode simple` is the legacy threshold-and-merge caller (no seeding/splitting/trimming); `--mode refined` adds `max_gap`/`min_width`/summit columns without the seed/valley/boundary machinery.
3. **pipeline** (alias: **burninate**) – Run both steps in sequence given an output filename prefix; supports the same `--mode`/`--seed_score`/`--boundary_fraction` options as `peaks`, with the same `profile` default
4. **fdr** (alias: **fdr_bw**) – Estimate empirical FDR for an *externally supplied* candidate peak set (e.g. ENCODE cCREs, dREG, groHMM) against a probability bigWig; shuffles the peak set genome-wide to build a null distribution and reports the score threshold at a target FDR. This is the validation path against independent annotations — there is no self-calibration flag on `peaks`/`pipeline` (a self-referential empirical-FDR calibration effort was tried and abandoned as a structural dead end; see `docs/trogdor_dreg_peak_calling_findings.md`).

Example (individual steps — `--mode profile` shown explicitly for clarity, but it's the default so it can be omitted):
```bash
trogdor score -M model.torch -p plus.bw -m minus.bw -o scores.bw -d cuda
trogdor peaks -i scores.bw -o peaks.bed.gz --mode profile --min_score 0.95
```

Example (FDR estimation against independent ground truth):

```bash
trogdor fdr -b scores.bw -t candidate_peaks.bed.gz --fdr_target 0.05 --output fdr_table.tsv --figure fdr_curve.png
```

The `fdr` subcommand scores each candidate peak with the summary statistic (`--stat max` or `mean`), then shuffles those peaks within chromosome bounds to build a null distribution. FDR at threshold `t` is estimated as `min(1, N_null(t) / N_real(t))`, averaged over `--n_shuffle` independent shuffles (default 1). The score threshold at the target FDR is printed to stdout.

The `profile` mode default and its `--boundary_fraction=0.95` default were chosen from a real benchmark sweep against independent groHMM+DNase truth (not from theory) — see `docs/trogdor_dreg_peak_calling_findings.md`'s "Independent-Ground-Truth Validation Results" and `docs/peak_calling_handoff.md`'s "Next Steps" for the numbers and the width/precision/recall trade-off `boundary_fraction` controls.

Example (full pipeline):

```bash
trogdor pipeline -M model.torch -p plus.bw -m minus.bw -o sample.peaks.bed.gz -d cuda
# writes sample.peaks.bed.gz; intermediate bigWig is written to a temp file
# and deleted automatically

# optionally save the bigWig:
trogdor pipeline -M model.torch -p plus.bw -m minus.bw -o sample.peaks.bed.gz -b sample.prob.bw -d cuda
```

Short contigs shorter than `--chunk_size` (default 262144) are automatically skipped by `score` with a warning when `-v` is set.

Pass `--num_workers N` (default 0) to enable parallel DataLoader workers for chunk preprocessing within each chromosome. Values of 1–4 are useful on Linux/CUDA systems; leave at 0 on macOS (fork-safety) or when CPU is not the bottleneck.

## Benchmark scripts

Diagnostic and evaluation scripts live in `scripts/benchmark/`:

- `benchmark.py` – Genome-wide AUROC/AUPRC from a trained model and peak BED ground truth
- `benchmark_bw.py` – Same benchmarking from a pre-computed probability bigWig
- `benchmark_tile_position.py` – Compares auPRC for tile-centre vs tile-edge bins across overlapping chunks
- `compare_peaks.py` / `truth_panel.py` – Peak-level overlap benchmarking against ground-truth BEDs (bin/peak-level precision-recall, centre-window hits); both merge subject intervals before computing coverage fractions to avoid double-counting nested/overlapping calls
- `frip.py` – Calculates raw and normalized FRIP (Fraction of Reads In Peaks) from stranded bigWigs and a peak BED; normalized FRIP corrects for peak-set size (equivalent to fold-enrichment over uniform expectation)
- `logit_dist.py` – Logit score distribution diagnostic: reads a probability bigWig, converts to logits, and produces a histogram+KDE / empirical-CDF figure with quantile reference lines
- `infp_filter.py` – Applies dREG's informative-positions heuristic (read-count thresholds in 100bp/1kbp windows) to mask a probability bigWig down to positions with real coverage support. Tried against the external-truth `trogdor fdr` workflow and found to add little value: it flattens the null to near-zero without fixing anything that was actually broken (the model already suppresses background on its own) — demoted, not recommended as a next step; see `docs/peak_calling_handoff.md`
- `fdr_dreg.py` – Empirical FDR estimation analogous to `trogdor fdr`, but for dREG's own scored BED output (centre 1bp per 100bp window) instead of a TROGDOR probability bigWig

## Architecture

### Package layout

- `cli/main.py` – CLI entry point (`cli()` function); parses args and dispatches to subcommands
- `cli/commands.py` – Subcommand implementations: `cmd_score`, `cmd_peaks`, `cmd_pipeline`, `cmd_fdr`
- `src/chiaroscuro/utils.py` – Shared utilities: `load_model()`, `merge_intervals()`, `encode_labels()`
- `src/chiaroscuro/trogdor.py` – Core model (`TROGDOR` class) and training loop
- `src/chiaroscuro/data_transforms.py` – `normalization()`, `standardization()` (deprecated)
- `src/chiaroscuro/modules.py` – `DoubleConv1D`, `EncoderBlock`, `DecoderBlock`
- `src/chiaroscuro/losses.py` – `focal_tversky_loss`, `tversky_loss`, `focal_loss`; the `TROGDOR` class itself defaults to plain unweighted `BCEWithLogitsLoss` (`loss_fn=None`), but the actual production recipe is `BCEWithLogitsLoss(pos_weight=500)`, applied via `loss_fn=` in `scripts/train/train.py --loss bce` (the default) — this has outperformed all three `chiaroscuro.losses` alternatives in benchmarking so far
- `src/chiaroscuro/dataset.py` – Dataset classes for training; not used in deployment
- `src/chiaroscuro/predict.py` – `predict_chromosome()` (sliding-window chromosome scoring via DataLoader) and `predict_genome()` (genome-wide generator with background IO prefetch); yields raw `torch.sigmoid` probabilities, no correction applied
- `src/chiaroscuro/peaks.py` – Peak-calling logic: `call_peaks()` (legacy threshold-and-merge), `call_profile_peaks()` (seed/smooth/valley-split/boundary-trim caller), `resolve_seed_score()` (defaults unset `seed_score` to `min(min_score, 0.5)` so profile mode's local-maxima splitting has more than one bin to work with)
- `src/chiaroscuro/stats.py` – Empirical FDR primitives used by the `fdr` subcommand: `score_peaks()` (summarise bigWig scores over a peak BED), `shuffle_peaks()` (uniform genome-wide null), `compute_fdr()` (build an FDR curve; `threshold_grid` of `quantile`/`linear`/`logit`/`unique` — `quantile` is default and oversamples the high-score tail, since TROGDOR's scores concentrate there), `select_fdr_threshold()`
- `src/chiaroscuro/logger.py` – Training metrics logger (copied from bpnet-lite)

### Model architecture (`TROGDOR`)

An asymmetric 1D U-Net for per-bin TIR prediction.

Input: `(batch, 2, length)` tensor of logistically-normalized stranded nascent RNA coverage.

- **Stem**: Conv1d (2→`base_channels`, kernel=7) + ReLU
- **Outer encoder** (`log2(output_stride)` levels): MaxPool2× downsampling to output resolution; skip connections discarded
- **Inner encoder** (`context_depth` levels): further downsampling with retained skip connections
- **Bottleneck**: DoubleConv1D
- **Decoder** (`context_depth` levels): ConvTranspose1d upsampling back to output resolution with skip connections
- **Head**: Conv1d → 1 channel logit per output bin

Output: `(batch, 1, length // output_stride)` logits. Loss: `BCEWithLogitsLoss(pos_weight=500)` (the production recipe — see `src/chiaroscuro/losses.py` above for library-default/alternative losses). Validation metrics logged during training: loss, AUPRC, Dice (`TROGDOR._validate`); AUROC is only computed post-hoc by `scripts/benchmark/benchmark.py`, not during training.

### Data normalization

Raw coverage is squashed per-strand to (0, 1) using a logistic function (`normalization()` in `data_transforms.py`) before passing to the model, following Danko et al. 2015. The reference point is the 99th percentile of nonzero values, clamped to `min_ref=20` to prevent noise amplification on sparse strands.

`standardization()` is a deprecated alias for the original global-max-based version.

## Development

All Python commands, import checks, and CLI invocations must be run inside the
`torch` conda environment:

```bash
conda run -n torch <command>
```

Examples:

```bash
conda run -n torch python -c "from chiaroscuro.peaks import call_profile_peaks; print('OK')"
conda run -n torch trogdor score --help
```
