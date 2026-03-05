"""
Tests for predict.py — both the existing predict() and new predict_chromosome().
"""

import pytest
import torch
import torch.nn as nn

from chiaroscuro.predict import predict, predict_chromosome

# ---------------------------------------------------------------------------
# Minimal mock models for testing
# ---------------------------------------------------------------------------


class IdentityUNet(nn.Module):
    """Returns a (B, 1, L//output_stride) tensor of zeros."""

    def __init__(self, output_stride=16):
        super().__init__()
        self.output_stride = output_stride

    def forward(self, x):
        B, _, L = x.shape
        return torch.zeros(B, 1, L // self.output_stride)


class ConstantUNet(nn.Module):
    """Returns a (B, 1, L//output_stride) tensor filled with a constant value."""

    def __init__(self, value=0.5, output_stride=16):
        super().__init__()
        self.value = value
        self.output_stride = output_stride
        # Dummy parameter so predict() can read dtype
        self._p = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, _, L = x.shape
        return torch.full((B, 1, L // self.output_stride), self.value)


class EchoFirstChannel(nn.Module):
    """Returns every output_stride-th position of the first input channel."""

    def __init__(self, output_stride=16):
        super().__init__()
        self._p = nn.Parameter(torch.tensor(0.0))
        self.output_stride = output_stride

    def forward(self, x):
        return x[:, :1, :: self.output_stride]  # (B, 1, L//output_stride)


# ---------------------------------------------------------------------------
# predict() (existing utility) — uses output_stride=1 models
# ---------------------------------------------------------------------------


class TestPredict:
    def test_basic_shape(self):
        model = ConstantUNet(value=1.0, output_stride=1)
        X = torch.randn(10, 2, 128)
        y = predict(model, X, batch_size=4, device="cpu")
        assert y.shape == (10, 1, 128)

    def test_constant_value(self):
        model = ConstantUNet(value=0.7, output_stride=1)
        X = torch.randn(6, 2, 64)
        y = predict(model, X, batch_size=3, device="cpu")
        assert torch.allclose(y, torch.full_like(y, 0.7))

    def test_single_example(self):
        model = IdentityUNet(output_stride=1)
        X = torch.randn(1, 2, 256)
        y = predict(model, X, batch_size=1, device="cpu")
        assert y.shape == (1, 1, 256)

    def test_batch_size_larger_than_dataset(self):
        model = IdentityUNet(output_stride=1)
        X = torch.randn(3, 2, 64)
        y = predict(model, X, batch_size=100, device="cpu")
        assert y.shape == (3, 1, 64)


# ---------------------------------------------------------------------------
# predict_chromosome()
# ---------------------------------------------------------------------------


class TestPredictChromosome:
    def _make_signal(self, length):
        return torch.randn(2, length)

    def test_output_shape(self):
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(8192)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=4096,
            overlap=512,
            output_stride=16,
            batch_size=2,
            device="cpu",
        )
        assert out.shape == (1, 512), f"Expected (1,512), got {out.shape}"

    def test_short_chrom_exact_one_chunk(self):
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(4096)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=4096,
            overlap=512,
            output_stride=16,
            batch_size=1,
            device="cpu",
        )
        assert out.shape == (1, 256)

    def test_chrom_shorter_than_stride(self):
        """Chromosome just barely larger than chunk_size — two starts needed."""
        model = IdentityUNet(output_stride=16)
        chunk_size = 256
        overlap = 32
        total = 300  # > chunk_size but < 2*stride
        signal = self._make_signal(total)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=chunk_size,
            overlap=overlap,
            output_stride=16,
            batch_size=2,
            device="cpu",
        )
        assert out.shape == (1, total // 16)

    def test_constant_model_fills_output(self):
        """A model returning a constant should produce that constant everywhere."""
        value = 0.42
        model = ConstantUNet(value=value, output_stride=16)
        total = 6144  # divisible by 16
        signal = self._make_signal(total)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=2048,
            overlap=256,
            output_stride=16,
            batch_size=4,
            device="cpu",
        )
        assert out.shape == (1, total // 16)
        assert torch.allclose(out, torch.full_like(out, value), atol=1e-5), (
            f"Not all values equal {value}: min={out.min()}, max={out.max()}"
        )

    def test_first_chunk_no_left_crop(self):
        """Position 0 (first output bin) must be written (first chunk left edge not cropped)."""
        model = ConstantUNet(value=1.0, output_stride=16)
        signal = self._make_signal(5120)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=2048,
            overlap=256,
            output_stride=16,
            batch_size=2,
            device="cpu",
        )
        assert out[0, 0].item() == pytest.approx(1.0)

    def test_last_chunk_no_right_crop(self):
        """Last output bin must be written (last chunk right edge not cropped)."""
        model = ConstantUNet(value=1.0, output_stride=16)
        total = 5120
        signal = self._make_signal(total)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=2048,
            overlap=256,
            output_stride=16,
            batch_size=2,
            device="cpu",
        )
        assert out[0, total // 16 - 1].item() == pytest.approx(1.0)

    def test_batch_size_one(self):
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(4096)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=1024,
            overlap=128,
            output_stride=16,
            batch_size=1,
            device="cpu",
        )
        assert out.shape == (1, 4096 // 16)

    def test_large_batch_size(self):
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(4096)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=1024,
            overlap=128,
            output_stride=16,
            batch_size=100,
            device="cpu",
        )
        assert out.shape == (1, 4096 // 16)

    def test_echo_model_passthrough(self):
        """EchoFirstChannel lets us verify signal values are passed correctly."""
        output_stride = 16
        model = EchoFirstChannel(output_stride=output_stride)
        total = 2048
        signal = torch.zeros(2, total)
        # Mark position 0 on the plus strand (first output bin, first chunk, no left crop)
        signal[0, 0] = 99.0
        out = predict_chromosome(
            model,
            signal,
            chunk_size=1024,
            overlap=128,
            output_stride=output_stride,
            batch_size=2,
            device="cpu",
        )
        assert out.shape == (1, total // output_stride)
        assert out[0, 0].item() == pytest.approx(99.0)

    def test_chunk_size_not_divisible_raises(self):
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(4096)
        with pytest.raises(ValueError, match="chunk_size"):
            predict_chromosome(
                model,
                signal,
                chunk_size=4097,
                overlap=512,
                output_stride=16,
                device="cpu",
            )

    def test_chrom_shorter_than_chunk_size_raises(self):
        """Chromosome smaller than chunk_size must raise a clear ValueError."""
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(2000)  # < chunk_size=4096
        with pytest.raises(ValueError, match="chunk_size"):
            predict_chromosome(
                model,
                signal,
                chunk_size=4096,
                overlap=512,
                output_stride=16,
                batch_size=1,
                device="cpu",
            )

    def test_overlap_not_divisible_raises(self):
        model = IdentityUNet(output_stride=16)
        signal = self._make_signal(4096)
        with pytest.raises(ValueError, match="overlap"):
            predict_chromosome(
                model,
                signal,
                chunk_size=4096,
                overlap=513,
                output_stride=16,
                device="cpu",
            )

    def test_output_stride_1_backward_compat(self):
        """output_stride=1 restores old per-position behaviour."""
        model = IdentityUNet(output_stride=1)
        total = 8192
        signal = self._make_signal(total)
        out = predict_chromosome(
            model,
            signal,
            chunk_size=4096,
            overlap=512,
            output_stride=1,
            batch_size=2,
            device="cpu",
        )
        assert out.shape == (1, total)
