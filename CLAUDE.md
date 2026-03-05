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

The tool operates in two steps (`pipeline` is a planned wrapper for both):

1. **score** – Score the whole genome using the pre-trained model; outputs a bigWig of probabilities
2. **peaks** – Call peaks from the scored bigWig using BH FDR correction

Example:
```bash
trogdor score -p plus.bw -m minus.bw -o scores.bw -d cuda
trogdor peaks -t scores.bw -o peaks.bed --fdr_threshold 0.05
```

Short contigs shorter than `--chunk_size` (default 262144) are automatically skipped by `score` with a warning when `-v` is set.

## Architecture

### Package layout

- `src/burninate/cli.py` – CLI entry point (`chiaroscuro()` function); parses args and dispatches to subcommands
- `src/burninate/trogdor.py` – Core model (`TROGDOR` class), `normalization()`, `standardization()` (deprecated), `load_pretrained_model()`
- `src/burninate/dataset.py` – Dataset classes for training; not used in deployment
- `src/burninate/predict.py` – `predict()` (batched inference, copied from tangermeme v1.0.2) and `predict_chromosome()` (sliding-window chromosome scoring)
- `src/burninate/logger.py` – Training metrics logger (copied from bpnet-lite)

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

### Pre-trained model

Bundled as `src/burninate/TROGDOR_UNET.torch` (loaded via `importlib.resources`). `load_pretrained_model()` in `trogdor.py` handles loading (`strict=False` to ignore training-only keys).

### Data normalization

Raw coverage is squashed per-strand to (0, 1) using a logistic function (`normalization()` in `trogdor.py`) before passing to the model, following Danko et al. 2015. The reference point is the 99th percentile of nonzero values, clamped to `min_ref=20` to prevent noise amplification on sparse strands.

`standardization()` is a deprecated alias for the original global-max-based version.

## Known Issues / In-Progress

- The `peaks` and `pipeline` subcommands are not yet implemented (they `pass`).
- `NascentDataset` in `dataset.py` has stub `_load_chrom` and `__getitem__` methods.
