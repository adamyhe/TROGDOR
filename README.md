# TROGDOR

Transcription Run-On Generates Detector Of Regulatory elements (TROGDOR).

https://www.youtube.com/watch?v=90X5NJleYJQ

TROGDOR identifies transcription initiation regions (TIRs) from stranded
nascent RNA sequencing data (GRO-seq, PRO-seq, ChRO-seq). A pre-trained model
is bundled with the package — most users will only need to install and run the CLI.

## Usage

An end-to-end pipeline can be run to identify peak regions using TROGDOR.

```bash
trogdor pipeline # Finish writing this
```

This bundles together region scoring and peak identification, which can be run separately
(only the scoring step requires GPUs).

```bash
trogdor score -p plus.bw -m minus.bw -o scores.bed -d cuda
trogdor peaks -t scores.bed -o peaks.bed --fdr_threshold 0.05
```

Input: stranded bigWig files. Output: BED file of called TIR peaks.

## Installation

```bash
pip install -e .
```

## Development/Model retraining

Install training and test dependencies:

```bash
pip install -e ".[dev]"
```

### Testing

Run the full unit test suite (70 tests, no data files required):

```bash
pytest tests/
```

Useful flags:

```bash
pytest tests/ -v              # verbose — print each test name
pytest tests/ -q              # quiet — dots only
pytest tests/ -x              # stop on first failure
pytest tests/ -k "predict"    # run only tests whose name contains "predict"
```

Run a single test file:

```bash
pytest tests/test_trogdor.py   # model architecture tests
pytest tests/test_predict.py   # predict() and predict_chromosome() tests
pytest tests/test_dataset.py   # NascentDataset_/NascentDataset tests
pytest tests/test_cli.py       # CLI argument-parsing tests
```

Integration tests (require real BigWig/BED files on disk) are skipped by default.
To run them explicitly:

```bash
pytest tests/ -m integration
```

### Training

Most users do not need to retrain — the pre-trained model is bundled with the
package and used automatically by the CLI. See [`scripts/README.md`](scripts/README.md)
for data download, training, and benchmarking instructions of the original TROGDOR model.
I haven't included general scripts for retraining on custom datasets, but these should
be a useful starting point.
