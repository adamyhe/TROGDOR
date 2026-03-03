# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TROGDOR is a deep learning method for identifying transcription initiation regions (TIRs) from nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). It takes stranded bigWig files as input and outputs predicted TIR peak calls.

## Installation

```bash
pip install -e .
```

The package installs four CLI aliases that all invoke the same entry point: `TROGDOR`, `trogdor`, `dREG`, `dreg`.

## CLI Pipeline

The tool operates in three steps (the `pipeline` subcommand is a planned wrapper for all three):

1. **infp** – Extract informative positions from bigWig files into a BED file
2. **score** – Score informative positions using the pre-trained model, output a scored BED
3. **peaks** – Call peaks from scored positions using BH FDR correction

Example:
```bash
trogdor infp -p plus.bw -m minus.bw -o infp.bed
trogdor score -i infp.bed -p plus.bw -m minus.bw -o scores.bed -d cuda
trogdor peaks -t scores.bed -o peaks.bed --fdr_threshold 0.05
```

## Architecture

### Package layout

- `src/cli.py` – CLI entry point (`chiaroscuro()` function); parses args and dispatches to subcommands
- `src/burninate/trogdor.py` – Core model (`TROGDOR` class), `standardization()`, `load_pretrained_model()`
- `src/burninate/dataset.py` – Dataset classes for training only; not used in deployment
- `src/burninate/predict.py` – Batched prediction utility (copied from tangermeme v1.0.2); handles memory-efficient GPU inference
- `src/burninate/logger.py` – Training metrics logger (copied from bpnet-lite)

### Model architecture (`TROGDOR`)

Input: `(batch, 2, length)` tensor of logistically-standardized stranded nascent RNA coverage.

1. Conv1d (2→64 channels, kernel=7) + ReLU
2. N × Inception1D blocks with additive skip connections (each Inception block outputs 480 channels, reduced back to 64 via 1×1 conv)
3. AdaptiveAvgPool1d → Linear → scalar logit
4. Loss: BCEWithLogitsLoss; Metrics: AUROC, AUPRC

`Inception1D` uses 5 parallel paths: 1×1 conv, and 3×/5×/7× conv with 1×1 reduction, plus a max-pool path.

### Pre-trained model

Bundled as `src/burninate/TROGDOR.torch` (loaded via `importlib.resources`). `load_pretrained_model()` in `trogdor.py` handles loading.

### Data standardization

Raw coverage is squashed to (0, 1) using a logistic function (`standardization()` in `trogdor.py`) before passing to the model, following Danko et al. 2015.

## Known Issues / In-Progress

- The `score`, `peaks`, and `pipeline` subcommands are not yet implemented (they `pass`).
- `NascentDataset` in `dataset.py` is incomplete (the `_load_chrom` and `__getitem__` methods are stubs).
- The `peaks` and `pipeline` subcommands are not yet implemented (they `pass`).
