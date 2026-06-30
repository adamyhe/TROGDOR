"""Regression tests for benchmark interval overlap helpers."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_stubbed_modules = []
if "pybigtools" not in sys.modules:
    sys.modules["pybigtools"] = types.ModuleType("pybigtools")
    _stubbed_modules.append("pybigtools")

torcheval = types.ModuleType("torcheval")
torcheval_metrics = types.ModuleType("torcheval.metrics")
torcheval_functional = types.ModuleType("torcheval.metrics.functional")
torcheval_functional.binary_auprc = lambda *args, **kwargs: None
torcheval_functional.binary_auroc = lambda *args, **kwargs: None
for name, module in (
    ("torcheval", torcheval),
    ("torcheval.metrics", torcheval_metrics),
    ("torcheval.metrics.functional", torcheval_functional),
):
    if name not in sys.modules:
        sys.modules[name] = module
        _stubbed_modules.append(name)


def _load_script_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_peaks = _load_script_module(
    "compare_peaks", "scripts/benchmark/compare_peaks.py"
)
truth_panel = _load_script_module(
    "truth_panel", "scripts/benchmark/truth_panel.py"
)

for name in _stubbed_modules:
    sys.modules.pop(name, None)


def _bed(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


@pytest.mark.parametrize("module", [compare_peaks, truth_panel])
def test_coverage_fractions_handle_nested_subject_intervals(module):
    query = _bed([("chr1", 50, 60), ("chr1", 20, 30), ("chr1", 100, 110)])
    subject = _bed([("chr1", 0, 100), ("chr1", 20, 30)])

    fracs = module.coverage_fractions(query, subject, "chr1")

    assert fracs[0] == pytest.approx(1.0)
    assert fracs[1] == pytest.approx(1.0)
    assert fracs[2] == pytest.approx(0.0)


@pytest.mark.parametrize("module", [compare_peaks, truth_panel])
def test_center_window_hits_handle_nested_subject_intervals(module):
    query = _bed([("chr1", 50, 52), ("chr1", 24, 26), ("chr1", 100, 110)])
    subject = _bed([("chr1", 0, 100), ("chr1", 20, 30)])

    q_hits, s_hits = module.center_window_hits(query, subject, "chr1", window=1)

    assert np.array_equal(q_hits, np.array([True, True, False]))
    assert np.array_equal(s_hits, np.array([True, True]))


def test_truth_panel_overlaps_any_handles_nested_subject_intervals():
    query = _bed([("chr1", 50, 60), ("chr1", 100, 110)])
    subject = _bed([("chr1", 0, 100), ("chr1", 20, 30)])

    hits = truth_panel._overlaps_any(query, subject, "chr1")

    assert np.array_equal(hits, np.array([True, False]))
