"""Tests for the dense-track memory paths: NumPy-array input to the peak
callers, per-chromosome filtering in ``cmd_peaks``, streamed writes in
``cmd_score``, the bounded chromosome prefetch in ``predict_genome``, and
``--dtype`` passthrough."""

import argparse
import importlib.util
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

hf = types.ModuleType("huggingface_hub")
hf.hf_hub_download = lambda *args, **kwargs: None
hf.try_to_load_from_cache = lambda *args, **kwargs: None
sys.modules.setdefault("huggingface_hub", hf)

import chiaroscuro.predict as predict_mod
from chiaroscuro.bigwig import (
    IntervalWriter,
    chrom_sizes,
    open_bigwig,
    read_runs,
    split_runs,
    split_runs_arrays,
    write_intervals,
)
from chiaroscuro.peaks import _merge_seed_blocks, call_peaks, call_profile_peaks

STRIDE = 16


def _load_commands_module():
    path = Path(__file__).resolve().parents[1] / "cli" / "commands.py"
    spec = importlib.util.spec_from_file_location("trogdor_cli_commands_dense", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_commands = _load_commands_module()


def _spiky_profile(n_bins, seed):
    """A TROGDOR-like profile: low background with sharp, partly adjacent spikes."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.01, 0.1, n_bins)
    for center in rng.choice(n_bins, size=n_bins // 20, replace=False):
        width = rng.integers(1, 8)
        lo, hi = max(0, center - width), min(n_bins, center + width + 1)
        values[lo:hi] = np.maximum(values[lo:hi], rng.uniform(0.2, 1.0, hi - lo))
    starts = np.arange(n_bins, dtype=np.int64) * STRIDE
    return starts, starts + STRIDE, values.astype(np.float32)


def _as_tuples(starts, ends, values):
    return list(zip(starts.tolist(), ends.tolist(), values.tolist()))


# --- NumPy-array input to the peak callers ---------------------------------


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_score": 0.95},
        {"min_score": 0.7, "max_gap": 32},
        {"min_score": 0.5, "max_gap": 16, "min_width": 48},
    ],
)
def test_call_peaks_arrays_match_tuples(seed, kwargs):
    arrays = _spiky_profile(2000, seed)
    assert call_peaks(arrays, **kwargs) == call_peaks(_as_tuples(*arrays), **kwargs)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_score": 0.95},
        {"min_score": 0.95, "seed_score": 0.3, "smooth_bins": 3, "boundary_fraction": 0.95},
        {"min_score": 0.9, "seed_score": 0.2, "max_gap": 16, "max_merge_distance": 64},
        {"min_score": 0.8, "smooth_bins": 5, "max_width": 64, "min_width": 32},
        {"min_score": 0.95, "split_merge_rule": "learned"},
    ],
)
def test_call_profile_peaks_arrays_match_tuples(seed, kwargs):
    arrays = _spiky_profile(2000, seed)
    expected = call_profile_peaks(_as_tuples(*arrays), **kwargs)
    assert call_profile_peaks(arrays, **kwargs) == expected


def test_merge_seed_blocks_arrays_match_tuples():
    arrays = _spiky_profile(2000, 0)
    for max_gap in (0, 16, 48):
        assert _merge_seed_blocks(arrays, 0.3, max_gap) == _merge_seed_blocks(
            _as_tuples(*arrays), 0.3, max_gap
        )


def test_float32_threshold_compares_in_float64():
    """0.95f is 0.9499999881 and must not pass >= 0.95, as with Python floats.
    (A float32 array compared to a Python float would compare in float32.)"""
    arrays = (
        np.array([0, 16], dtype=np.int64),
        np.array([16, 32], dtype=np.int64),
        np.array([0.95, 0.99], dtype=np.float32),
    )
    peaks = call_peaks(arrays, min_score=0.95)
    assert [(p["start"], p["end"]) for p in peaks] == [(16, 32)]
    assert peaks == call_peaks(_as_tuples(*arrays), min_score=0.95)


def test_empty_array_input():
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    assert call_peaks(empty) == []
    assert call_profile_peaks(empty) == []


# --- bigWig helpers ---------------------------------------------------------


def test_split_runs_arrays_matches_split_runs():
    starts = np.array([0, 48, 64, 200], dtype=np.int64)
    ends = np.array([48, 64, 160, 216], dtype=np.int64)
    values = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
    arrays = split_runs_arrays(starts, ends, values, STRIDE)
    assert _as_tuples(*arrays) == split_runs(starts, ends, values, STRIDE)


def test_interval_writer_streams_per_chrom(tmp_path):
    sizes = {"chr1": 1024, "chr2": 512, "chr3": 256}
    path = str(tmp_path / "streamed.bw")
    with IntervalWriter(path, sizes) as writer:
        writer.write("chr1", [0, 32], [16, 48], [0.5, 0.25])
        writer.write("chr2", [], [], [])  # empty chromosome is skipped
        writer.write("chr3", [128], [144], [0.75])
    bw = open_bigwig(path)
    assert chrom_sizes(bw) == sizes
    s, e, v = read_runs(bw, "chr1", 1024)
    assert s.tolist() == [0, 32] and e.tolist() == [16, 48]
    assert v.tolist() == pytest.approx([0.5, 0.25])
    assert len(read_runs(bw, "chr2", 512)[0]) == 0
    assert read_runs(bw, "chr3", 256)[0].tolist() == [128]


# --- cmd_peaks: per-chromosome filtering ------------------------------------


def _write_dense_track(path, chroms, seed, threshold=None):
    """Write a dense score bigWig (optionally pre-sliced at ``threshold``)."""
    intervals, sizes = {}, {}
    for i, (chrom, n_bins) in enumerate(chroms.items()):
        s, e, v = _spiky_profile(n_bins, seed + i)
        if threshold is not None:
            keep = v.astype(np.float64) >= threshold
            s, e, v = s[keep], e[keep], v[keep]
        intervals[chrom] = (s, e, v)
        sizes[chrom] = n_bins * STRIDE
    write_intervals(str(path), sizes, intervals)


def _peaks_args(input_path, output_path, **overrides):
    args = dict(
        input=str(input_path),
        output=str(output_path),
        min_score=0.95,
        mode="profile",
        max_gap=0,
        min_width=0,
        max_width=None,
        seed_score=None,
        smooth_bins=3,
        valley_fraction=0.5,
        boundary_fraction=0.95,
        split_merge_rule="threshold",
        split_merge_cutoff=0.5,
        max_merge_distance=None,
        min_support_signal=0.0,
        support_plus_bigwig=None,
        support_minus_bigwig=None,
        verbose=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.mark.parametrize(
    "overrides,threshold",
    [
        ({"mode": "simple"}, 0.95),
        ({"mode": "refined", "max_gap": 32, "min_width": 32}, 0.95),
        ({"mode": "profile"}, 0.5),
        ({"mode": "profile", "seed_score": 0.3, "max_merge_distance": 64}, 0.3),
    ],
)
def test_cmd_peaks_dense_matches_presliced(tmp_path, overrides, threshold):
    """Dense input gives the same calls as input pre-sliced at the caller's
    candidate threshold -- the bins cmd_peaks now drops on read."""
    chroms = {"chr1": 3000, "chr2": 1500, "chrM": 40}
    dense, sliced = tmp_path / "dense.bw", tmp_path / "sliced.bw"
    _write_dense_track(dense, chroms, seed=7)
    _write_dense_track(sliced, chroms, seed=7, threshold=threshold)

    out_dense, out_sliced = tmp_path / "dense.bed", tmp_path / "sliced.bed"
    _commands.cmd_peaks(_peaks_args(dense, out_dense, **overrides))
    _commands.cmd_peaks(_peaks_args(sliced, out_sliced, **overrides))

    text = out_dense.read_text()
    assert text  # the profiles do produce calls
    assert text == out_sliced.read_text()


# --- cmd_score: streamed writes ---------------------------------------------


def _write_coverage(tmp_path, sizes):
    pl, mn = tmp_path / "pl.bw", tmp_path / "mn.bw"
    for path in (pl, mn):
        write_intervals(
            str(path), sizes, {c: ([0], [n], [1.0]) for c, n in sizes.items()}
        )
    return str(pl), str(mn)


def _score_args(pl, mn, output, **overrides):
    args = dict(
        model=None,
        pl_bigwig=pl,
        mn_bigwig=mn,
        output=str(output),
        device="cpu",
        chunk_size=256,
        overlap=32,
        output_stride=STRIDE,
        batch_size=2,
        chroms=None,
        min_score=0.0,
        verbose=False,
        num_workers=0,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _fake_predict_genome(fail_on=None, seen=None):
    def fake(model, pl_bigwig, mn_bigwig, chroms=None, output_stride=16, **kwargs):
        if seen is not None:
            seen.update(kwargs)
        sizes = chrom_sizes(open_bigwig(pl_bigwig))
        for i, chrom in enumerate(chroms):
            if chrom == fail_on:
                raise RuntimeError("boom")
            n = sizes[chrom] // output_stride
            yield chrom, sizes[chrom], np.full(n, 0.1 * (i + 1), dtype=np.float32)

    return fake


def test_cmd_score_streams_in_scoring_order(tmp_path, monkeypatch):
    sizes = {"chr1": 512, "chr2": 1024, "chrShort": 128, "chr3": 768}
    pl, mn = _write_coverage(tmp_path, sizes)
    monkeypatch.setattr(_commands, "_load_trogdor_model", lambda *a: (None, "cpu"))
    monkeypatch.setattr(_commands, "predict_genome", _fake_predict_genome())
    out = tmp_path / "scores.bw"
    with pytest.warns(UserWarning, match="shorter than chunk_size"):
        _commands.cmd_score(_score_args(pl, mn, out, chroms=["chr3", "chr1", "chrShort"]))

    bw = open_bigwig(str(out))
    # Header: only scored chromosomes (short contig dropped), in scoring order.
    assert list(chrom_sizes(bw)) == ["chr3", "chr1"]
    s, e, v = read_runs(bw, "chr3", 768)
    assert (s[0], e[-1]) == (0, 768) and v[0] == pytest.approx(0.1)


def test_cmd_score_removes_partial_output_on_failure(tmp_path, monkeypatch):
    sizes = {"chr1": 512, "chr2": 512}
    pl, mn = _write_coverage(tmp_path, sizes)
    monkeypatch.setattr(_commands, "_load_trogdor_model", lambda *a: (None, "cpu"))
    monkeypatch.setattr(_commands, "predict_genome", _fake_predict_genome(fail_on="chr2"))
    out = tmp_path / "scores.bw"
    with pytest.raises(RuntimeError, match="boom"):
        _commands.cmd_score(_score_args(pl, mn, out))
    assert not out.exists()


@pytest.mark.parametrize(
    "dtype,expected",
    [(None, "auto"), ("auto", "auto"), ("bf16", torch.bfloat16), ("fp32", torch.float32)],
)
def test_cmd_score_passes_dtype(tmp_path, monkeypatch, dtype, expected):
    pl, mn = _write_coverage(tmp_path, {"chr1": 512})
    seen = {}
    monkeypatch.setattr(_commands, "_load_trogdor_model", lambda *a: (None, "cpu"))
    monkeypatch.setattr(_commands, "predict_genome", _fake_predict_genome(seen=seen))
    overrides = {} if dtype is None else {"dtype": dtype}  # None: attribute absent
    _commands.cmd_score(_score_args(pl, mn, tmp_path / "s.bw", **overrides))
    assert seen["dtype"] == expected


@pytest.mark.parametrize("subcommand", ["score", "pipeline"])
def test_dtype_cli_option(monkeypatch, subcommand):
    import cli.main as cli_main

    captured = {}
    target = "cmd_score" if subcommand == "score" else "cmd_pipeline"
    monkeypatch.setattr(cli_main, target, lambda args: captured.update(vars(args)))
    base = [subcommand, "-p", "pl.bw", "-m", "mn.bw", "-o", "out"]

    monkeypatch.setattr(sys, "argv", ["trogdor", *base])
    cli_main.cli()
    assert captured["dtype"] == "auto"

    monkeypatch.setattr(sys, "argv", ["trogdor", *base, "--dtype", "fp32"])
    cli_main.cli()
    assert captured["dtype"] == "fp32"

    monkeypatch.setattr(sys, "argv", ["trogdor", *base, "--dtype", "fp16"])
    with pytest.raises(SystemExit):
        cli_main.cli()


# --- predict_genome: bounded prefetch ---------------------------------------


class _SlowZeros(nn.Module):
    """Zero logits, slowly, so the prefetch thread gets to run ahead."""

    def forward(self, x):
        time.sleep(0.02)
        return torch.zeros(x.shape[0], 1, x.shape[-1] // STRIDE)


def test_predict_genome_holds_at_most_two_chromosomes(tmp_path, monkeypatch):
    sizes = {f"chr{i}": 512 for i in range(1, 7)}
    pl, mn = _write_coverage(tmp_path, sizes)

    live, max_live, lock = [0], [0], threading.Lock()
    real_read = predict_mod._read_chrom

    def tracked_read(*args):
        signal = real_read(*args)
        with lock:
            live[0] += 1
            max_live[0] = max(max_live[0], live[0])

        def _freed():
            with lock:
                live[0] -= 1

        weakref.finalize(signal, _freed)
        return signal

    monkeypatch.setattr(predict_mod, "_read_chrom", tracked_read)
    results = list(
        predict_mod.predict_genome(
            _SlowZeros(), pl, mn, chunk_size=256, overlap=32, batch_size=1,
            device="cpu", dtype=torch.float32,
        )
    )
    assert [chrom for chrom, _, _ in results] == list(sizes)
    assert max_live[0] <= 2
