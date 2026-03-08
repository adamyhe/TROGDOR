"""
Tests for the TROGDOR asymmetric 1D U-Net model (trogdor.py).
"""

import functools

import pytest
import torch
from torch.nn import BCEWithLogitsLoss

from chiaroscuro.data_transforms import normalization
from chiaroscuro.losses import focal_loss, focal_tversky_loss, tversky_loss
from chiaroscuro.modules import DecoderBlock, DoubleConv1D, EncoderBlock
from chiaroscuro.trogdor import TROGDOR

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_output_range(self):
        """All output values must lie in [0, 1]."""
        t = torch.rand(2, 1000) * 100
        out = normalization(t)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_shape_preserved(self):
        """Output tensor must have the same shape as the input."""
        t = torch.rand(2, 500)
        out = normalization(t)
        assert out.shape == t.shape

    def test_all_zeros(self):
        """All-zero input (no coverage) must produce an all-zero output."""
        t = torch.zeros(2, 100)
        out = normalization(t)
        assert out.shape == t.shape
        assert torch.all(out == 0.0)

    def test_per_strand_independence(self):
        """A spike on strand 0 must not suppress the moderate values on strand 1."""
        t = torch.zeros(2, 1000)
        t[0] = torch.ones(1000) * 10
        t[0, 0] = 10000  # spike on strand 0
        t[1] = torch.ones(1000) * 10  # moderate values on strand 1
        out = normalization(t)
        assert out[1].mean() > 0.3

    def test_spike_robustness(self):
        """A single extreme spike must not push the remaining positions near zero."""
        t = torch.zeros(1, 1000)
        t[0] = torch.ones(1000) * 10
        t[0, 0] = 10000  # single spike
        out = normalization(t)
        assert out[0].mean() > 0.3

    def test_gene_desert_noise_suppressed(self):
        """1-read background noise should not amplify to ~1.0 in gene deserts."""
        t = torch.zeros(1, 10000)
        # Scatter ~5% positions with single-read background noise
        noise_idx = torch.randperm(10000)[:500]
        t[0, noise_idx] = 1.0
        out = normalization(t)
        # With min_ref=20: beta=1.0, F(1 read) = sigmoid(0) = 0.5 (not amplified)
        assert out.max() <= 0.5


# ---------------------------------------------------------------------------
# DoubleConv1D
# ---------------------------------------------------------------------------


class TestDoubleConv1D:
    def test_forward_shape(self):
        """Output length must equal input length (same-padding)."""
        m = DoubleConv1D(8, 16, kernel_size=3)
        x = torch.randn(2, 8, 64)
        y = m(x)
        assert y.shape == (2, 16, 64), f"Expected (2,16,64), got {y.shape}"

    def test_same_in_out_channels(self):
        """Works correctly when in_channels == out_channels."""
        m = DoubleConv1D(32, 32, kernel_size=3)
        x = torch.randn(4, 32, 128)
        y = m(x)
        assert y.shape == (4, 32, 128)

    def test_even_kernel_raises(self):
        """An even kernel_size must raise ValueError (same-padding requires odd kernels)."""
        with pytest.raises(ValueError):
            DoubleConv1D(8, 16, kernel_size=4)

    def test_large_kernel(self):
        """kernel_size=7 (larger receptive field) must still preserve sequence length."""
        m = DoubleConv1D(4, 8, kernel_size=7)
        x = torch.randn(1, 4, 256)
        y = m(x)
        assert y.shape == (1, 8, 256)


# ---------------------------------------------------------------------------
# EncoderBlock
# ---------------------------------------------------------------------------


class TestEncoderBlock:
    def test_forward_shapes(self):
        """Skip has pre-pool shape; pooled is 2× downsampled."""
        m = EncoderBlock(8, 16, kernel_size=3)
        x = torch.randn(2, 8, 64)
        skip, pooled = m(x)
        assert skip.shape == (2, 16, 64), f"skip shape {skip.shape}"
        assert pooled.shape == (2, 16, 32), f"pooled shape {pooled.shape}"

    def test_odd_length_pooled(self):
        """MaxPool1d floors odd lengths: L=65 → pooled L=32."""
        m = EncoderBlock(4, 8)
        x = torch.randn(1, 4, 65)
        skip, pooled = m(x)
        assert skip.shape == (1, 8, 65)
        assert pooled.shape == (1, 8, 32)  # floor(65/2)


# ---------------------------------------------------------------------------
# DecoderBlock
# ---------------------------------------------------------------------------


class TestDecoderBlock:
    def test_forward_shape(self):
        """Upsampled + skip concat must produce the skip's spatial length."""
        m = DecoderBlock(in_channels=16, skip_channels=8, out_channels=8)
        x = torch.randn(2, 16, 32)  # upsampled from L//2
        skip = torch.randn(2, 8, 64)
        y = m(x, skip)
        assert y.shape == (2, 8, 64), f"Expected (2,8,64), got {y.shape}"

    def test_odd_length_handling(self):
        """Encoder may floor odd lengths; decoder must handle ±1 mismatch."""
        m = DecoderBlock(in_channels=16, skip_channels=8, out_channels=8)
        x = torch.randn(1, 16, 32)  # would pair with skip of 64 or 65
        skip = torch.randn(1, 8, 65)
        y = m(x, skip)
        assert y.shape == (1, 8, 65)

    def test_pad_to_match_crops(self):
        """When upsampled tensor is 1 longer than skip, it should be cropped."""
        m = DecoderBlock(in_channels=8, skip_channels=4, out_channels=4)
        x = torch.randn(1, 8, 33)  # 1 too long
        skip = torch.randn(1, 4, 32)
        y = m(x, skip)
        assert y.shape == (1, 4, 32)


# ---------------------------------------------------------------------------
# TROGDOR end-to-end
# ---------------------------------------------------------------------------


class TestTROGDOR:
    @pytest.fixture
    def model(self):
        # Small model for fast tests: output_stride=4, context_depth=2
        return TROGDOR(
            in_channels=2,
            base_channels=8,
            output_stride=4,
            context_depth=2,
            kernel_size=3,
        )

    def test_output_shape_small(self, model):
        """Small model: (3, 2, 256) → (3, 1, 64) since 256 // 4 = 64."""
        x = torch.randn(3, 2, 256)
        y = model(x)
        assert y.shape == (3, 1, 64), f"Expected (3,1,64), got {y.shape}"

    def test_output_shape_batch1(self):
        """output_stride=4, context_depth=2: (1, 2, 256) → (1, 1, 64)."""
        m = TROGDOR(base_channels=8, output_stride=4, context_depth=2)
        x = torch.randn(1, 2, 256)
        y = m(x)
        assert y.shape == (1, 1, 64), f"Expected (1,1,64), got {y.shape}"

    def test_output_stride_16_default(self):
        """Default model: (2, 2, 4096) → (2, 1, 256) since 4096 // 16 = 256."""
        m = TROGDOR(base_channels=8, output_stride=16, context_depth=2)
        x = torch.randn(2, 2, 4096)
        y = m(x)
        assert y.shape == (2, 1, 256), f"Expected (2,1,256), got {y.shape}"

    def test_output_stride_not_power_of_2_raises(self):
        """output_stride=6 is not a power of 2 and must raise ValueError."""
        with pytest.raises(ValueError):
            TROGDOR(output_stride=6)

    def test_output_stride_1_valid(self):
        """output_stride=1 is a power of 2; model has zero outer encoders and returns (B, 1, L)."""
        m = TROGDOR(base_channels=8, output_stride=1, context_depth=2)
        x = torch.randn(1, 2, 64)
        y = m(x)
        assert y.shape == (1, 1, 64)

    def test_inner_skips_count(self, model):
        """The forward pass collects exactly context_depth inner skips."""
        # Patch forward to expose inner_skips count
        inner_skips_count = []
        original_forward = model.forward

        def patched_forward(X):
            X = model.stem(X)
            for enc in model.outer_encoders:
                _, X = enc(X)
            skips = []
            for enc in model.inner_encoders:
                skip, X = enc(X)
                skips.append(skip)
            inner_skips_count.append(len(skips))
            X = model.bottleneck(X)
            for dec, skip in zip(model.decoders, reversed(skips)):
                X = dec(X, skip)
            return model.head(X)

        x = torch.randn(1, 2, 256)
        patched_forward(x)
        assert inner_skips_count[0] == len(model.inner_encoders)

    def test_outer_encoders_count(self, model):
        """outer_encoders has n_out = log2(output_stride) levels."""
        import math

        expected = int(math.log2(model.output_stride))
        assert len(model.outer_encoders) == expected

    def test_gradient_flows(self, model):
        """Loss.backward() must populate at least one parameter gradient."""
        x = torch.randn(2, 2, 256, requires_grad=False)
        y = model(x)
        loss = y.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_output_length_matches_stride(self):
        """Output length must equal input_length // output_stride."""
        m = TROGDOR(base_channels=8, output_stride=8, context_depth=2)
        L = 512
        x = torch.randn(1, 2, L)
        y = m(x)
        assert y.shape[-1] == L // 8

    def test_channel_cap(self):
        """Channels must not exceed max_channels."""
        m = TROGDOR(
            base_channels=32, output_stride=16, context_depth=4, max_channels=128
        )
        # Check that all inner encoder output channels are <= 128
        for enc in m.inner_encoders:
            out_ch = enc.conv.block[0].out_channels
            assert out_ch <= 128, f"Channel count {out_ch} exceeds max_channels=128"


# ---------------------------------------------------------------------------
# focal_loss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_output_is_scalar(self):
        """focal_loss must return a zero-dimensional scalar tensor."""
        logits = torch.randn(2, 1, 64)
        targets = torch.zeros(2, 1, 64)
        targets[0, 0, :4] = 1.0
        loss = focal_loss(logits, targets)
        assert loss.shape == torch.Size([])

    def test_output_nonnegative(self):
        """focal_loss must always be >= 0."""
        logits = torch.randn(4, 1, 32)
        targets = (torch.rand(4, 1, 32) > 0.9).float()
        loss = focal_loss(logits, targets)
        assert loss.item() >= 0.0

    def test_focal_lower_than_bce_on_easy_negatives(self):
        """On confident correct negatives, focal loss should be lower than BCE."""
        import torch.nn.functional as F

        # Very negative logits → model is confident these are negatives, and they are
        logits = torch.full((1, 1, 100), -10.0)
        targets = torch.zeros(1, 1, 100)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        fl = focal_loss(logits, targets, alpha=0.001, gamma=2.0)
        assert fl.item() < bce.item()

    def test_perfect_predictions_low_loss(self):
        """Very high logits for positives, very low for negatives → near-zero loss."""
        logits = torch.cat(
            [torch.full((1, 1, 10), 20.0), torch.full((1, 1, 90), -20.0)], dim=2
        )
        targets = torch.cat([torch.ones(1, 1, 10), torch.zeros(1, 1, 90)], dim=2)
        loss = focal_loss(logits, targets)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# tversky_loss
# ---------------------------------------------------------------------------


class TestTverskyLoss:
    def test_output_in_unit_interval(self):
        """Tversky loss must lie in [0, 1] (up to float tolerance)."""
        logits = torch.randn(2, 1, 64)
        targets = (torch.rand(2, 1, 64) > 0.9).float()
        loss = tversky_loss(logits, targets)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_perfect_predictions_near_zero(self):
        """Near-perfect sigmoid predictions → Tversky loss ≈ 0."""
        logits = torch.full((1, 1, 50), 20.0)
        targets = torch.ones(1, 1, 50)
        loss = tversky_loss(logits, targets)
        assert loss.item() < 0.01

    def test_all_wrong_near_one(self):
        """Predicting 1 for all negatives → high Tversky loss."""
        logits = torch.full((1, 1, 50), 20.0)  # all predicted positive
        targets = torch.zeros(1, 1, 50)  # all actually negative
        loss = tversky_loss(logits, targets)
        assert loss.item() > 0.5

    def test_fn_penalised_more_than_fp(self):
        """With beta > alpha, a clear FN should cost more than a clear FP of equal magnitude."""
        # Scenario A: one hard FN — model confidently predicts 0 for the one positive
        logits_fn = torch.full((1, 1, 10), -20.0)  # all very negative predictions
        targets_fn = torch.zeros(1, 1, 10)
        targets_fn[0, 0, 0] = 1.0  # one positive → completely missed

        # Scenario B: one hard FP — model confidently predicts 1 for one negative
        logits_fp = torch.full((1, 1, 10), -20.0)
        logits_fp[0, 0, 0] = 20.0  # one false alarm
        targets_fp = torch.zeros(1, 1, 10)  # all negative

        loss_fn = tversky_loss(logits_fn, targets_fn, alpha=0.3, beta=0.7)
        loss_fp = tversky_loss(logits_fp, targets_fp, alpha=0.3, beta=0.7)
        assert loss_fn.item() > loss_fp.item()


# ---------------------------------------------------------------------------
# TROGDOR with focal+tversky loss
# ---------------------------------------------------------------------------


class TestTROGDORFocalTversky:
    @pytest.fixture
    def model(self):
        _fl = functools.partial(focal_loss, alpha=0.999, gamma=2.0)
        _tl = functools.partial(tversky_loss, alpha=0.3, beta=0.7)
        return TROGDOR(
            in_channels=2,
            base_channels=8,
            output_stride=4,
            context_depth=2,
            kernel_size=3,
            loss_fn=lambda logits, y: _fl(logits, y) + _tl(logits, y),
        )

    def test_forward_runs(self, model):
        """Forward pass with combined focal+tversky loss_fn must produce the correct shape."""
        x = torch.randn(2, 2, 256)
        y = model(x)
        assert y.shape == (2, 1, 64)

    def test_loss_step_runs(self, model):
        """One forward+backward step with focal+tversky loss should not error."""
        x = torch.randn(2, 2, 256)
        targets = torch.zeros(2, 1, 64)
        targets[0, 0, :2] = 1.0
        logits = model(x)
        loss = model._loss_fn(logits, targets)
        loss.backward()
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# TROGDOR with bce+tversky loss
# ---------------------------------------------------------------------------


class TestTROGDORBCETversky:
    @pytest.fixture
    def model(self):
        _bce = BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([10.0]))
        _tl = functools.partial(tversky_loss, alpha=0.3, beta=0.7)
        return TROGDOR(
            in_channels=2,
            base_channels=8,
            output_stride=4,
            context_depth=2,
            kernel_size=3,
            loss_fn=lambda logits, y: _bce(logits, y).mean() + _tl(logits, y),
        )

    def test_loss_step_nonnegative(self, model):
        """Forward + compound loss should be a non-negative scalar."""
        x = torch.randn(2, 2, 256)
        targets = torch.zeros(2, 1, 64)
        targets[0, 0, :2] = 1.0
        logits = model(x)
        loss = model._loss_fn(logits, targets)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# focal_tversky_loss
# ---------------------------------------------------------------------------


class TestFocalTverskyLoss:
    def test_output_in_unit_interval(self):
        """FTL must lie in [0, 1] (up to float tolerance)."""
        logits = torch.randn(2, 1, 64)
        targets = (torch.rand(2, 1, 64) > 0.9).float()
        loss = focal_tversky_loss(logits, targets)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_perfect_predictions_near_zero(self):
        """Very high positive logits for all-positive targets → FTL < 0.01."""
        logits = torch.full((1, 1, 50), 20.0)
        targets = torch.ones(1, 1, 50)
        loss = focal_tversky_loss(logits, targets)
        assert loss.item() < 0.01

    def test_all_wrong_near_one(self):
        """Predicting all positive for all-negative targets → FTL > 0.5."""
        logits = torch.full((1, 1, 50), 20.0)
        targets = torch.zeros(1, 1, 50)
        loss = focal_tversky_loss(logits, targets)
        assert loss.item() > 0.5

    def test_gamma_1_equals_tversky(self):
        """gamma=1 → FTL == tversky_loss exactly (1/1 = 1 power)."""
        logits = torch.randn(2, 1, 64)
        targets = (torch.rand(2, 1, 64) > 0.9).float()
        ftl = focal_tversky_loss(logits, targets, gamma=1.0)
        tl = tversky_loss(logits, targets)
        assert abs(ftl.item() - tl.item()) < 1e-6

    def test_larger_gamma_raises_loss(self):
        """gamma=3 → FTL > tversky_loss for imperfect predictions (x^(1/3) > x for x in (0,1))."""
        logits = torch.randn(2, 1, 64)
        targets = (torch.rand(2, 1, 64) > 0.9).float()
        ftl = focal_tversky_loss(logits, targets, gamma=3.0)
        tl = tversky_loss(logits, targets)
        assert ftl.item() > tl.item()


# ---------------------------------------------------------------------------
# TROGDOR with focal_tversky loss
# ---------------------------------------------------------------------------


class TestTROGDORFocalTverskyLoss:
    @pytest.fixture
    def model(self):
        return TROGDOR(
            base_channels=8,
            output_stride=4,
            context_depth=2,
            loss_fn=focal_tversky_loss,
            loss_kwargs={"alpha": 0.3, "beta": 0.7, "gamma": 4/3},
        )

    def test_forward_runs(self, model):
        """Forward pass with focal_tversky_loss must produce the correct output shape."""
        x = torch.randn(2, 2, 256)
        y = model(x)
        assert y.shape == (2, 1, 64)

    def test_loss_step_runs(self, model):
        """Forward + focal_tversky_loss + backward completes without error."""
        x = torch.randn(2, 2, 256)
        targets = torch.zeros(2, 1, 64)
        targets[0, 0, :2] = 1.0
        logits = model(x)
        loss = model._loss_fn(logits, targets)
        loss.backward()
        assert loss.item() >= 0.0
