# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TROGDOR is a deep learning method for identifying transcription initiation regions (TIRs) from nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). It takes stranded bigWig files as input and outputs a bigWig of predicted TIR probabilities and/or BED peak calls.

## Installation

```bash
uv tool install trogdor
```

For development from source, see `CONTRIBUTING.md`.

The package installs two CLI aliases that invoke the same entry point: `TROGDOR`, `trogdor`.

## CLI Pipeline

The tool operates in four subcommands:

1. **score** (alias: **thatch**) – Score the whole genome using the pre-trained model; outputs a bigWig of raw sigmoid probabilities (no multiple-testing correction is applied)
2. **peaks** (alias: **consummate_vs**) – Call peaks from the scored bigWig by thresholding probabilities at `--min_score` and merging passing bins (`--mode simple`, the default; `--mode refined` adds `max_gap`/`min_width`/summit columns)
3. **pipeline** (alias: **burninate**) – Run both steps in sequence given an output filename prefix
4. **fdr** (alias: **fdr_bw**) – Estimate empirical FDR for candidate peaks from a probability bigWig; shuffles the peak set to build a null distribution and reports the score threshold at a target FDR

Example (individual steps):
```bash
trogdor score -M model.torch -p plus.bw -m minus.bw -o scores.bw -d cuda
trogdor peaks -i scores.bw -o peaks.bed.gz --fdr_threshold 0.05
```

Example (FDR estimation):

```bash
trogdor fdr -b scores.bw -t candidate_peaks.bed.gz --fdr_target 0.05 --output fdr_table.tsv --figure fdr_curve.png
```

The `fdr` subcommand scores each candidate peak with the summary statistic (`--stat max` or `mean`), then shuffles those peaks within chromosome bounds to build a null distribution. FDR at threshold `t` is estimated as `min(1, N_null(t) / N_real(t))`, averaged over `--n_shuffle` independent shuffles (default 1). The score threshold at the target FDR is printed to stdout.

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
- `frip.py` – Calculates raw and normalized FRIP (Fraction of Reads In Peaks) from stranded bigWigs and a peak BED; normalized FRIP corrects for peak-set size (equivalent to fold-enrichment over uniform expectation)
- `logit_dist.py` – Logit score distribution diagnostic: reads a probability bigWig, converts to logits, and produces a histogram+KDE / empirical-CDF figure with quantile reference lines

## Architecture

### Package layout

- `cli/main.py` – CLI entry point (`cli()` function); parses args and dispatches to subcommands
- `cli/commands.py` – Subcommand implementations: `cmd_score`, `cmd_peaks`, `cmd_pipeline`
- `src/chiaroscuro/bigwig.py` – figwig-backed bigWig I/O helpers: `open_bigwig()`, `chrom_sizes()`, `read_values()` (per-base reads, split into windows for multithreaded decompression), `read_runs()`/`split_runs()`/`infer_bin_size()` (figwig has no `records()`; these rebuild a score bigWig's fixed-width bins, since equal-valued neighbouring bins read back as one run), `write_intervals()`
- `src/chiaroscuro/utils.py` – Shared utilities: `load_model()`, `merge_intervals()`, `encode_labels()`
- `src/chiaroscuro/trogdor.py` – Core model (`TROGDOR` class) and training loop
- `src/chiaroscuro/data_transforms.py` – `normalization()`, `standardization()` (deprecated)
- `src/chiaroscuro/modules.py` – `DoubleConv1D`, `EncoderBlock`, `DecoderBlock`, `Conv1DBlock`
- `src/chiaroscuro/losses.py` – `focal_tversky_loss` (default), `tversky_loss`, `focal_loss`
- `src/chiaroscuro/dataset.py` – Dataset classes for training; not used in deployment
- `src/chiaroscuro/predict.py` – `predict_chromosome()` (sliding-window chromosome scoring via DataLoader) and `predict_genome()` (genome-wide generator with background IO prefetch)
- `src/chiaroscuro/stats.py` – Empirical FDR helpers: `score_peaks()` (summarise bigWig scores over a peak BED), `shuffle_peaks()` (randomise peak positions within chromosome bounds), `compute_fdr()` (build FDR curve from real and null scores)
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

Output: `(batch, 1, length // output_stride)` logits. Loss: BCEWithLogitsLoss. Metrics: AUROC, AUPRC, Dice.

### Data normalization

Raw coverage is squashed per-strand to (0, 1) using a logistic function (`normalization()` in `data_transforms.py`) before passing to the model, following Danko et al. 2015. The reference point is the 99th percentile of nonzero values, clamped to `min_ref=20` to prevent noise amplification on sparse strands.

`standardization()` is a deprecated alias for the original global-max-based version.

## Development

Development is managed with uv only (no pip/conda). Set up once with
`uv sync --group dev`, then run all Python commands, import checks, tests, and
CLI invocations through `uv run`:

```bash
uv run python -c "from chiaroscuro.bigwig import read_values; print('OK')"
uv run trogdor score --help
uv run pytest -m "not integration"
```

Add dependencies with `uv add` (`uv add --group dev` for dev-only) and commit
`uv.lock` alongside `pyproject.toml`. See `CONTRIBUTING.md`.
