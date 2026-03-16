# TROGDOR

Transcription Run-On Generates Detector Of Regulatory elements (TROGDOR).

https://www.youtube.com/watch?v=90X5NJleYJQ

TROGDOR identifies transcription initiation regions (TIRs) from stranded
nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). It uses a 1D U-Net model
and a tiled image segmentation approach to achieve SOTA performance at predicting
TIRs while maintaining computational efficiency.

## Installation

We recommend installing inside an isolated Python environment (conda, venv, or uv):

```bash
pip install git+https://github.com/adamyhe/TROGDOR.git
```

## Usage

### Quick start

Run the full pipeline with a single command:

```bash
trogdor pipeline -p plus.bw -m minus.bw -o mysample.peaks.bed.gz
```

This writes one output file:

| File                    | Description                            |
| ----------------------- | -------------------------------------- |
| `mysample.peaks.bed.gz` | Called TIR peak regions (bgzipped BED) |

The intermediate probability bigWig is written to a temporary file and deleted automatically.

**Inputs**: plus- and minus-strand bigWig files from a nascent RNA sequencing experiment (GRO-seq, PRO-seq, or ChRO-seq).

**GPU**: scoring uses a 1D U-Net model implemented in plain PyTorch and thus can be greatly accelerated by running on a CUDA-capable GPU (particularly Ampere or newer architectures that support bf16). Apple Silicon MPS (`-d mps`) should also work but has not been tested. Pass `-d cpu` to run on CPU (much slower). If CUDA is unavailable, the tool automatically falls back to MPS (if detected) or CPU.

**Pretrained model**: downloaded automatically from [HuggingFace Hub](https://huggingface.co/adamyhe/TROGDOR) on first run and cached locally. To use a custom model, pass `-M /path/to/model.torch`.

### Key options

| Flag                 | Default | Description                                                             |
| -------------------- | ------- | ----------------------------------------------------------------------- |
| `-d / --device`      | `cuda`  | PyTorch device (`cuda`, `cpu`, `cuda:1`, …)                             |
| `-s / --min_score`   | `0.95`  | Minimum score threshold; bins below this are not reported               |
| `-b / --save_bigwig` | off     | Save the intermediate probability bigWig to this path (`pipeline` only) |
| `--chroms`           | all     | Score only specific chromosomes (e.g. `--chroms chr1 chr2`)             |
| `-v / --verbose`     | off     | Print progress messages                                                 |

### Running steps separately

The pipeline can also be run as two separate steps — useful if you want to call peaks at multiple thresholds without re-scoring:

```bash
# Step 1: score (GPU recommended)
trogdor score -p plus.bw -m minus.bw -o mysample.prob0.9.bw -s 0.9

# Step 2: call peaks at different thresholds (CPU, fast)
trogdor peaks -i mysample.prob0.9.bw -o mysample.peaks0.9.bed.gz -s 0.9
trogdor peaks -i mysample.prob0.9.bw -o mysample.peaks0.95.bed.gz -s 0.95
trogdor peaks -i mysample.prob0.9.bw -o mysample.peaks0.99.bed.gz -s 0.99
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
