# TROGDOR

Transcription Run-On Generates Detector Of Regulatory elements (TROGDOR).

https://www.youtube.com/watch?v=90X5NJleYJQ

TROGDOR identifies transcription initiation regions (TIRs) from stranded
nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). A pre-trained model
is bundled with the package — most users will only need to install and run the CLI.

## Usage

### Quick start

Run the full pipeline with a single command:

```bash
trogdor pipeline -p plus.bw -m minus.bw -n mysample 
```

This writes two output files:

| File                    | Description                                   |
| ----------------------- | --------------------------------------------- |
| `mysample.prob.bw`      | Per-bin TIR scores (bigWig, 16 bp resolution) |
| `mysample.peaks.bed.gz` | Called TIR peak regions (bgzipped BED)        |

**Inputs**: plus- and minus-strand bigWig files from a nascent RNA sequencing experiment (GRO-seq, PRO-seq, or ChRO-seq).

**GPU**: scoring uses a deep learning model and thus can be greatly accelerated by running on a CUDA-capable GPU. Apple Silicon MPS (`-d mps`) should also work but has not been tested. Pass `-d cpu` to run on CPU (much slower). If CUDA is unavailable, the tool automatically falls back to MPS (if detected) or CPU.

**Pretrained model**: downloaded automatically from HuggingFace Hub on first run and cached locally. To use a custom model, pass `-M /path/to/model.torch`.

### Key options

| Flag                   | Default | Description                                                      |
| ---------------------- | ------- | ---------------------------------------------------------------- |
| `-d / --device`        | `cuda`  | PyTorch device (`cuda`, `cpu`, `cuda:1`, …)                      |
| `-f / --fdr_threshold` | `0.1`   | BH FDR threshold for peak calls                                  |
| `-s / --min_score`     | `0.9`   | Pre-filter: only bins with raw score ≥ this enter FDR correction |
| `--chroms`             | all     | Score only specific chromosomes (e.g. `--chroms chr1 chr2`)      |
| `-v / --verbose`       | off     | Print progress messages                                          |

### Running steps separately

The pipeline can also be run as two separate steps — useful if you want to call peaks at multiple FDR thresholds without re-scoring:

```bash
# Step 1: score (GPU recommended)
trogdor score -p plus.bw -m minus.bw -o mysample.prob.bw -d cuda

# Step 2: call peaks (CPU, fast)
trogdor peaks -i mysample.prob.bw -o mysample.peaks.bed.gz --fdr_threshold 0.1
trogdor peaks -i mysample.prob.bw -o mysample.peaks.bed.gz --fdr_threshold 0.05
trogdor peaks -i mysample.prob.bw -o mysample.peaks.bed.gz --fdr_threshold 0.01
```

## Installation

We recommend installing inside an isolated Python environment (conda, venv, or uv):

```bash
pip install git@github.com:adamyhe/TROGDOR.git
```

## Development/Model retraining

Install development dependencies:

```bash
git clone git@github.com:adamyhe/TROGDOR.git
cd TROGDOR
pip install -e ".[dev]"
```

### Training

Most users do not need to retrain — a pre-trained model is bundled with the
package and used automatically by the CLI. See [`scripts/README.md`](scripts/README.md)
for data download, training, and benchmarking instructions of the original TROGDOR model.
I haven't included general scripts for retraining on custom datasets, but these should
be a useful starting point.
