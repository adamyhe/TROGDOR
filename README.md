# TROGDOR

Transcription Run-On Grants Detection Of Regulatory elements (TROGDOR).

https://www.youtube.com/watch?v=90X5NJleYJQ

TROGDOR identifies transcription initiation regions (TIRs) from stranded
nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq, mNET-seq, etc.). It
uses a 1D U-Net model and a tiled image segmentation approach to achieve SOTA
performance at predicting TIRs while maintaining computational efficiency.

## Installation

We recommend installing inside an isolated Python environment (conda, venv, or uv):

```bash
pip install trogdor
```

Or install the latest development version directly from GitHub:

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

**Inputs**: plus- and minus-strand bigWig files from a nascent RNA sequencing experiment. TROGDOR was trained on PRO/GRO-seq data, and has been vetted on data from GRO/PRO/ChRO/mNET-seq experiments. These files should represent coverage tracks of the 3' ends of reads/fragments (that is, the most recent nucleotide added by the polymerase), ideally in raw counts. The minus strand data can be stored as either positive or negative.

**GPU**: scoring uses a 1D U-Net model implemented in plain PyTorch and thus can be greatly accelerated by running on a CUDA-capable GPU (particularly Ampere or newer architectures that support bf16). Apple Silicon MPS (`-d mps`) should also work but has not been tested. Pass `-d cpu` to run on CPU (much slower). If CUDA is unavailable, the tool automatically falls back to MPS (if detected) or CPU. Inference uses a streaming pipeline: bigWig IO for the next chromosome runs in a background thread while the GPU processes the current one, and chunks are fed to the GPU via a DataLoader with `pin_memory` for async CPU→GPU transfer.

**Pretrained model**: downloaded automatically from [HuggingFace Hub](https://huggingface.co/adamyhe/TROGDOR) on first run and cached locally. To use a custom model, pass `-M /path/to/model.torch`.

### Key options

| Flag                 | Default | Description                                                             |
| -------------------- | ------- | ----------------------------------------------------------------------- |
| `-d / --device`      | `cuda`  | PyTorch device (`cuda`, `cpu`, `cuda:1`, …)                             |
| `-s / --min_score`   | `0.95`  | Minimum score threshold; bins below this are not reported               |
| `-b / --save_bigwig` | off     | Save the intermediate probability bigWig to this path (`pipeline` only) |
| `--chroms`           | all     | Score only specific chromosomes (e.g. `--chroms chr1 chr2`)             |
| `--num_workers`      | `0`     | DataLoader workers for chunk preprocessing (set to 1–4 on Linux/CUDA)   |
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

### Empirical FDR estimation and `min_score` calibration

The `fdr` subcommand estimates the score threshold corresponding to a target empirical FDR from a probability bigWig and a ground truth peak set (e.g. ENCODE PLS/ELS or PRO-cap peaks for your cell type of interest). This can be useful for deciding what `min_score` threshold you should use (although the default `0.95` has worked well for me).

```bash
# Step 1: generate a dense score bigWig (report ALL values)
trogdor score -p plus.bw -m minus.bw -o mysample.prob.bw --min_score 0
# Step 2: calculate empirical FDR against a candidate set of "ground truth" peaks
trogdor fdr -b mysample.prob.bw -t candidate_peaks.bed.gz --fdr_target 0.05
```

**Strategy**: each candidate peak is summarised by its max (or mean) bigWig score. A null distribution is built by shuffling peak positions uniformly within chromosome bounds (preserving widths). FDR at threshold *t* is estimated as min(1, N\_null(*t*) / N\_real(*t*)), averaged over `--n_shuffle` independent shuffles. The score threshold at the target FDR is printed to stdout.

| Flag            | Default | Description                                              |
| --------------- | ------- | -------------------------------------------------------- |
| `-b / --bigwig` | —       | Probability bigWig (required)                            |
| `-t / --peaks`  | —       | Candidate peak BED (required)                            |
| `--stat`        | `max`   | Summary statistic per peak (`max` or `mean`)             |
| `--n_shuffle`   | `1`     | Independent genome shuffles to average the null over     |
| `--fdr_target`  | `0.05`  | Target FDR for reporting the score threshold             |
| `--output`      | off     | Write TSV table of threshold/FDR/N\_real/N\_null to path |
| `--figure`      | off     | Save FDR-vs-threshold plot to path                       |
| `--chroms`      | all     | Restrict to specific chromosomes                         |

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
