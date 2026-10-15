# Contributing

Thanks for helping improve TROGDOR.

Development is managed entirely with [uv](https://docs.astral.sh/uv/): it
creates the virtual environment, installs dependencies from `uv.lock`, and runs
every command. You don't need pip, conda or a manually activated environment.

## Development setup

Clone the repository and install TROGDOR from source with the development
dependencies:

```bash
git clone git@github.com:adamyhe/TROGDOR.git
cd TROGDOR
uv sync --group dev
```

This creates `.venv/` with an editable install of the package, so changes under
`cli/` and `src/chiaroscuro/` take effect immediately. Prefix commands with
`uv run` to run them inside that environment:

```bash
uv run trogdor --help
uv run python scripts/benchmark/benchmark_bw.py --help
```

## Dependencies

Add or remove dependencies with uv so that `pyproject.toml` and `uv.lock` stay
in sync, and commit both files:

```bash
uv add <package>                # runtime dependency
uv add --group dev <package>    # development-only dependency
uv remove <package>
uv lock --upgrade-package <package>
```

## Tests

Run the unit test suite (no data files required):

```bash
uv run pytest -m "not integration"
```

Integration tests require real bigWig/BED files on disk and are deselected by
default. See [`tests/README.md`](tests/README.md) for running single files and
the integration tests.

CI runs the unit tests on Python 3.10 through 3.14. To check another version
locally:

```bash
uv run --python 3.14 pytest -m "not integration"
```

## Building

```bash
uv build
```

writes the sdist and wheel to `dist/`. Releases are published to PyPI by the
`Publish to PyPI` workflow when a GitHub release is created.

## Model retraining

Most users do not need to retrain TROGDOR; the pretrained model is downloaded
automatically from [HuggingFace Hub](https://huggingface.co/adamyhe/TROGDOR).
See [`scripts/README.md`](scripts/README.md) for data download, training and
benchmarking instructions for the original model.

Some data-preparation steps need UCSC command-line tools such as `liftOver`,
which are not Python packages. One way to install them is from bioconda:

```bash
mamba install -c bioconda ucsc-liftover
```

Training logs to [Weights & Biases](https://wandb.ai), which is included in the
dev group. Log in once before training:

```bash
uv run wandb login
```
