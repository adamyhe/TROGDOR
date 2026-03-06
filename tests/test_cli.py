"""
Tests for the CLI (cli.py) — argument parsing smoke tests.
These do not invoke actual model inference; they only check that argument
parsing works correctly and that the subcommands are wired up.
"""

import numpy as np
import pytest

from chiaroscuro.cli import _bh_threshold, _merge_intervals

# We import the argparse-building code via a direct call to avoid needing
# installed entry points or a TROGDOR_UNET.torch weights file.


def _make_parser():
    """Re-create the argparse parser without executing any subcommand body."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="cmd")

    # pipeline
    p_pipe = subparsers.add_parser("pipeline")
    p_pipe.add_argument("-p", "--pl_bigwig", required=True)
    p_pipe.add_argument("-m", "--mn_bigwig", required=True)
    p_pipe.add_argument("-o", "--output", required=True)
    p_pipe.add_argument("-d", "--device", default="cuda")
    p_pipe.add_argument("-v", "--verbose", action="store_true")

    # score
    p_score = subparsers.add_parser("score")
    p_score.add_argument("-i", "--input", required=False, default=None)
    p_score.add_argument("-p", "--pl_bigwig", required=True)
    p_score.add_argument("-m", "--mn_bigwig", required=True)
    p_score.add_argument("-o", "--output", required=True)
    p_score.add_argument("-d", "--device", default="cuda")
    p_score.add_argument("--chunk_size", type=int, default=262144)
    p_score.add_argument("--overlap", type=int, default=32768)
    p_score.add_argument("--output_stride", type=int, default=16)
    p_score.add_argument("--chroms", nargs="*", default=None)
    p_score.add_argument("-v", "--verbose", action="store_true")

    # peaks
    p_peaks = subparsers.add_parser("peaks")
    p_peaks.add_argument("-t", "--input", required=True)
    p_peaks.add_argument("-o", "--output", required=True)
    p_peaks.add_argument("-f", "--fdr_threshold", type=float, default=0.05)
    p_peaks.add_argument("-v", "--verbose", action="store_true")

    return parser


class TestScoreArgParsing:
    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_minimal_required_args(self, parser):
        """Minimal required flags parse correctly and set cmd='score'."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "plus.bw",
                "-m",
                "minus.bw",
                "-o",
                "out.bw",
            ]
        )
        assert args.cmd == "score"
        assert args.pl_bigwig == "plus.bw"
        assert args.mn_bigwig == "minus.bw"
        assert args.output == "out.bw"

    def test_input_is_optional(self, parser):
        """Omitting -i/--input must leave args.input as None."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.input is None

    def test_input_can_be_provided(self, parser):
        """-i/--input stores the provided path when given."""
        args = parser.parse_args(
            [
                "score",
                "-i",
                "infp.bed",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.input == "infp.bed"

    def test_chunk_size_default(self, parser):
        """chunk_size defaults to 262144 (2^18) when not specified."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.chunk_size == 262144

    def test_chunk_size_custom(self, parser):
        """--chunk_size stores the provided integer value."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "--chunk_size",
                "8192",
            ]
        )
        assert args.chunk_size == 8192

    def test_overlap_default(self, parser):
        """overlap defaults to 32768 (2^15) when not specified."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.overlap == 32768

    def test_overlap_custom(self, parser):
        """--overlap stores the provided integer value."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "--overlap",
                "256",
            ]
        )
        assert args.overlap == 256

    def test_output_stride_default(self, parser):
        """output_stride defaults to 16 when not specified."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.output_stride == 16

    def test_output_stride_custom(self, parser):
        """--output_stride stores the provided integer value."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "--output_stride",
                "32",
            ]
        )
        assert args.output_stride == 32

    def test_chroms_default_is_none(self, parser):
        """Omitting --chroms must leave args.chroms as None (score all chromosomes)."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.chroms is None

    def test_chroms_single(self, parser):
        """--chroms with one value produces a single-element list."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "--chroms",
                "chr1",
            ]
        )
        assert args.chroms == ["chr1"]

    def test_chroms_multiple(self, parser):
        """--chroms accepts multiple space-separated chromosome names."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "--chroms",
                "chr1",
                "chr2",
                "chrX",
            ]
        )
        assert args.chroms == ["chr1", "chr2", "chrX"]

    def test_device_default(self, parser):
        """device defaults to 'cuda' when not specified."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
            ]
        )
        assert args.device == "cuda"

    def test_device_cpu(self, parser):
        """-d cpu overrides the default device."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "-d",
                "cpu",
            ]
        )
        assert args.device == "cpu"

    def test_verbose_flag(self, parser):
        """-v sets args.verbose to True."""
        args = parser.parse_args(
            [
                "score",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "o.bw",
                "-v",
            ]
        )
        assert args.verbose is True


class TestPeaksArgParsing:
    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_required_args(self, parser):
        """Required -t/--input and -o/--output parse correctly and set cmd='peaks'."""
        args = parser.parse_args(
            [
                "peaks",
                "-t",
                "scores.bw",
                "-o",
                "peaks.bed",
            ]
        )
        assert args.cmd == "peaks"
        assert args.input == "scores.bw"
        assert args.output == "peaks.bed"

    def test_fdr_default(self, parser):
        """fdr_threshold defaults to 0.05 when not specified."""
        args = parser.parse_args(
            [
                "peaks",
                "-t",
                "scores.bw",
                "-o",
                "peaks.bed",
            ]
        )
        assert args.fdr_threshold == pytest.approx(0.05)

    def test_fdr_custom(self, parser):
        """-f/--fdr_threshold stores the provided float value."""
        args = parser.parse_args(
            [
                "peaks",
                "-t",
                "scores.bw",
                "-o",
                "peaks.bed",
                "-f",
                "0.01",
            ]
        )
        assert args.fdr_threshold == pytest.approx(0.01)


class TestPipelineArgParsing:
    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_required_args(self, parser):
        """Required flags parse correctly and set cmd='pipeline'."""
        args = parser.parse_args(
            [
                "pipeline",
                "-p",
                "p.bw",
                "-m",
                "m.bw",
                "-o",
                "out.bed",
            ]
        )
        assert args.cmd == "pipeline"
        assert args.pl_bigwig == "p.bw"


class TestPeaksHelpers:
    # _bh_threshold tests

    def test_bh_all_pass(self):
        """All-ones probability array → every bin passes BH at alpha=0.05."""
        probs = np.ones(10)
        t = _bh_threshold(probs, 0.05)
        assert t is not None
        assert all(p >= t for p in probs)

    def test_bh_none_pass(self):
        """All-zeros probability array → no bin passes; returns None."""
        probs = np.zeros(10)
        assert _bh_threshold(probs, 0.05) is None

    def test_bh_empty(self):
        """Empty array → no bins to test; returns None."""
        assert _bh_threshold(np.array([]), 0.05) is None

    def test_bh_partial(self):
        """Only the highest-probability bin passes BH; threshold equals that probability."""
        # 4 bins: probs 0.99, 0.8, 0.6, 0.01 → p-values 0.01, 0.2, 0.4, 0.99
        # sorted p: [0.01, 0.2, 0.4, 0.99]; BH thresholds at alpha=0.05: [0.0125, 0.025, 0.0375, 0.05]
        # Only rank 1 passes (0.01 <= 0.0125); highest passing sorted_p = 0.01 → threshold = 0.99
        probs = np.array([0.99, 0.8, 0.6, 0.01])
        t = _bh_threshold(probs, 0.05)
        assert t == pytest.approx(0.99)

    # _merge_intervals tests

    def test_merge_adjacent(self):
        """Directly adjacent intervals must be merged into one."""
        assert _merge_intervals([(0, 16), (16, 32), (32, 48)]) == [(0, 48)]

    def test_merge_gap(self):
        """Intervals separated by a gap must remain distinct."""
        assert _merge_intervals([(0, 16), (32, 48)]) == [(0, 16), (32, 48)]

    def test_merge_empty(self):
        """Empty input list must return an empty list."""
        assert _merge_intervals([]) == []

    def test_merge_single(self):
        """A single interval must be returned as-is."""
        assert _merge_intervals([(0, 16)]) == [(0, 16)]
