# TROGDOR

[![PyPI](https://img.shields.io/pypi/v/trogdor)](https://pypi.org/project/trogdor/) [![Tests](https://github.com/adamyhe/TROGDOR/actions/workflows/tests.yml/badge.svg)](https://github.com/adamyhe/TROGDOR/actions/workflows/tests.yml) [![Weights](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/adamyhe/TROGDOR) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/trogdor?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/trogdor)

Transcription Run-On Grants Detection Of Regulatory elements (TROGDOR).

https://www.youtube.com/watch?v=90X5NJleYJQ

TROGDOR identifies transcription initiation regions (TIRs) from stranded nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq, mNET-seq, etc.). It uses a 1D U-Net model and a tiled image segmentation approach to achieve SOTA performance at predicting TIRs while maintaining computational efficiency.

## Installation

We recommend installing inside an isolated Python environment. With **uv** (fastest):

```bash
uv tool install trogdor
```

Or with pip inside a conda/venv environment:

```bash
pip install trogdor
```

If you want to use TROGDOR as a dependency in a uv-managed Python project:

```bash
uv add trogdor
```

For development from source, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

By default, the intermediate probability scores are streamed directly into the
peak caller rather than materialized as a bigWig. Use `--save_bigwig` if you
want to keep the score track.

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

The default `peaks` command preserves the original threshold-and-merge caller.
An experimental refined caller is available for benchmarking post-processing
choices without changing the model or default behavior:

```bash
trogdor peaks -i mysample.prob.bw -o mysample.refined.bed \
  --mode refined --min_score 0.95 --max_gap 32 --min_width 32
```

Refined output includes BED columns for the merged peak and the max-score
summit bin: `chrom`, `start`, `end`, `score`, `summit_start`, `summit_end`,
`summit_score`. `--min_support_signal` can optionally require raw plus/minus
coverage support when `--support_plus_bigwig` and `--support_minus_bigwig` are
provided.

A profile-aware caller is also available for local peak-shape refinement:

```bash
trogdor peaks -i mysample.prob.bw -o mysample.profile.bed \
  --mode profile --min_score 0.95 --seed_score 0.5
```

The profile caller uses TROGDOR scores non-parametrically: it seeds candidate
blocks at `--seed_score`, splits nearby local summits only when the intervening
valley is sufficiently deep, trims optional low-scoring shoulders, and reports
summit columns. `--max_gap`, `--smooth_bins`, `--valley_fraction`, and
`--boundary_fraction` are tuning knobs for held-out benchmarking rather than
recommended constants. If `score` and `peaks` are run separately, the score
bigWig must have been written with a storage threshold no higher than the
intended `--seed_score`. The one-step `pipeline` command avoids writing an
intermediate score bigWig by default, so profile calling can use a lower
`--seed_score` without materializing that score track unless `--save_bigwig` is
explicitly set.

The one-step pipeline can also calibrate the caller empirically from streamed
per-chromosome probabilities, without writing a dense probability bigWig:

```bash
trogdor pipeline -p plus.bw -m minus.bw -o mysample.profile.fdr05.bed.gz \
  --peak_mode profile --min_score 0.95 --seed_score 0.5 \
  --calibrate --calibration_fdr_target 0.05 \
  --raw_output mysample.profile.raw.bed.gz \
  --calibration_curve mysample.profile.fdr.tsv \
  --calibration_figure mysample.profile.fdr.png
```

With `--calibrate`, TROGDOR first calls candidate peaks from the streamed
probabilities, then builds a non-parametric null within the thresholded
candidate intervals by default (`--null_scope candidate`). The raw BED is
written to `--raw_output` (or a `.raw` sibling of `--output` when omitted), and
the final `--output` BED contains peaks whose summit score reaches the empirical
FDR target. The default `--calibration_stat summit` shuffles summit-sized
windows for the null; `--calibration_stat max` or `mean` instead shuffles full
peak intervals and scores the interval body. Use `--null_scope genome` to
shuffle within whole chromosomes instead. `--calibrate` currently uses the
streaming pipeline path and should be run without `--save_bigwig`.
Empirical FDR is evaluated on a tail-enriched `--threshold_grid quantile` grid
by default, which avoids skipping the saturated high-score tail of TROGDOR
probabilities. `--calibration_figure` writes a PNG/PDF/SVG-style figure,
depending on the file extension accepted by matplotlib, showing real/null score
distributions and the empirical FDR curve on a logit x-axis by default
(`--calibration_plot_scale logit`).

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
| `--threshold_grid` | `quantile` | Threshold grid (`quantile`, `linear`, `logit`, or `unique`) |
| `--output`      | off     | Write TSV table of threshold/FDR/N\_real/N\_null to path |
| `--figure`      | off     | Save FDR-vs-threshold plot to path                       |
| `--chroms`      | all     | Restrict to specific chromosomes                         |

## Development/Model retraining

### Training

Most users do not need to retrain — a pre-trained model is bundled with the
package and used automatically by the CLI. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development setup, and [`scripts/README.md`](scripts/README.md) for data
download, training, and benchmarking instructions of the original TROGDOR model.
I haven't included general scripts for retraining on custom datasets, but these
should be a useful starting point.
