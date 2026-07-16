# Contributing

Thanks for helping improve TROGDOR.

## Development setup

Clone the repository:

```bash
git clone git@github.com:adamyhe/TROGDOR.git
cd TROGDOR
```

Install TROGDOR from source with development dependencies:

```bash
uv sync --group dev
```

## Tests

Run the non-integration test suite with:

```bash
uv run pytest -m "not integration"
```

Integration tests require real genomic input files and are skipped by default.

## Optional model retraining tools

Some training and benchmarking workflows require UCSC command-line tools, such
as `liftOver`. One way to install them is:

```bash
mamba install -c bioconda ucsc-liftover
```

Most users do not need to retrain TROGDOR. See [`scripts/README.md`](scripts/README.md)
for data download, training, and benchmarking instructions for the original
model.
