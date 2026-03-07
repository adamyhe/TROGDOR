# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TROGDOR is a deep learning method for identifying transcription initiation regions (TIRs) from nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). It takes stranded bigWig files as input and outputs a bigWig of predicted TIR probabilities and/or BED peak calls.

## Installation

```bash
pip install -e .
```

The package installs four CLI aliases that all invoke the same entry point: `TROGDOR`, `trogdor`, `dREG`, `dreg`.

## CLI Pipeline

The tool operates in three subcommands:

1. **score** (alias: **thatch**) – Score the whole genome using the pre-trained model; outputs a bigWig of BH-adjusted probabilities
2. **peaks** (alias: **consummate_vs**) – Call peaks from the scored bigWig using a score threshold derived from a BH FDR threshold
3. **pipeline** (alias: **burninate**) – Run both steps in sequence given an output filename prefix

Example (individual steps):
```bash
trogdor score -M model.torch -p plus.bw -m minus.bw -o scores.bw -d cuda
trogdor peaks -i scores.bw -o peaks.bed.gz --fdr_threshold 0.05
```

Example (full pipeline):
```bash
trogdor pipeline -M model.torch -p plus.bw -m minus.bw -n sample -d cuda
# writes sample.prob.bw and sample.peaks.bed.gz
```

Short contigs shorter than `--chunk_size` (default 262144) are automatically skipped by `score` with a warning when `-v` is set.

## Architecture

### Package layout

- `src/chiaroscuro/cli.py` – CLI entry point (`cli()` function); parses args and dispatches to subcommands
- `src/chiaroscuro/commands.py` – Subcommand implementations: `cmd_score`, `cmd_peaks`, `cmd_pipeline`
- `src/chiaroscuro/trogdor.py` – Core model (`TROGDOR` class) and training loop
- `src/chiaroscuro/data_transforms.py` – `normalization()`, `standardization()` (deprecated)
- `src/chiaroscuro/modules.py` – `DoubleConv1D`, `EncoderBlock`, `DecoderBlock`, `Conv1DBlock`
- `src/chiaroscuro/losses.py` – `focal_tversky_loss` (default), `tversky_loss`, `focal_loss`
- `src/chiaroscuro/dataset.py` – Dataset classes for training; not used in deployment
- `src/chiaroscuro/predict.py` – `predict()` (batched inference, copied from tangermeme v1.0.2) and `predict_chromosome()` (sliding-window chromosome scoring)
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

