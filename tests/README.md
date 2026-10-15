# Tests

Run the full unit test suite (no data files required) after `uv sync --group dev`
(see [`CONTRIBUTING.md`](../CONTRIBUTING.md)):

```bash
uv run pytest tests/
```

Useful flags:

```bash
uv run pytest tests/ -v              # verbose — print each test name
uv run pytest tests/ -q              # quiet — dots only
uv run pytest tests/ -x              # stop on first failure
uv run pytest tests/ -k "predict"    # run only tests whose name contains "predict"
```

Run a single test file:

```bash
uv run pytest tests/test_trogdor.py   # model architecture tests
uv run pytest tests/test_predict.py   # predict() and predict_chromosome() tests
uv run pytest tests/test_dataset.py   # NascentDataset_/NascentDataset tests
uv run pytest tests/test_cli.py       # CLI argument-parsing tests
```

Integration tests (require real BigWig/BED files on disk) are skipped by default.
To run them explicitly:

```bash
uv run pytest tests/ -m integration
```
