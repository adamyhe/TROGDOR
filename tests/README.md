# Tests

Run the full unit test suite (no data files required):

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
