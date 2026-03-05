# trogdor.py
# Author: Adam He <adamyhe@gmail.com>

"""
This module defines a neural network predicts TSS from nascent RNA
sequencing data
"""

import importlib.resources
import math
import time

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torcheval.metrics.functional import binary_auprc

from .logger import Logger
from .predict import predict

torch.backends.cudnn.benchmark = True


def load_pretrained_model(name="TROGDOR_UNET.torch", model_params={}):
    """
    Small utility function to load a pre-trained TROGDOR model placed inside
    of the deployed package.
    """
    model = TROGDOR(**model_params)
    with importlib.resources.path("burninate", name) as path:
        model.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return model


def normalization(t, x=0.05, y=0.01, min_ref=20):
    """
    Normalize each strand of nascent RNA coverage to (0, 1) using a
    per-strand logistic function, following Danko et al. 2015.

    For each strand independently:
      1. Compute ``ref`` as the 99th percentile of nonzero values,
         clamped to at least ``min_ref``.
      2. Set the inflection point ``beta = x * ref``.
      3. Derive slope ``alpha`` so that the logistic equals ``y`` at zero
         coverage: ``alpha = log(1/y - 1) / beta``.
      4. Apply ``F(v) = 1 / (1 + exp(-alpha * (v - beta)))``.

    Strands with no nonzero coverage are left as zeros.

    Parameters
    ----------
    t : torch.Tensor, shape (C, L)
        Stranded nascent RNA sequencing coverage; one row per strand.
    x : float, default 0.05
        Inflection point as a fraction of ``ref``; controls where the
        logistic transitions from low to high.
    y : float, default 0.01
        Logistic value at zero coverage; sets the baseline suppression
        of background signal.
    min_ref : float, default 20
        Minimum value for ``ref``, preventing extreme slopes on very
        low-coverage strands.

    Returns
    -------
    result : torch.Tensor, shape (C, L)
        Coverage values mapped to (0, 1) per strand.
    """
    result = torch.zeros_like(t)
    for i in range(t.shape[0]):
        strand = t[i]
        nonzero = strand[strand > 0]
        if nonzero.numel() == 0:
            continue
        ref = (
            torch.quantile(nonzero.float(), 0.99)
            if nonzero.numel() >= 2
            else nonzero.max()
        )
        ref = max(ref.item(), min_ref)
        beta = x * ref
        alpha = (1 / beta) * np.log(1 / y - 1)
        result[i] = 1 / (1 + torch.exp(-alpha * (strand - beta)))
    return result


def standardization(t, x=0.05, y=0.01):
    """Backwards-compatible alias for the original single-pass normalization.

    .. deprecated::
        Use :func:`normalization` instead. This function scales by the global
        maximum of the whole tensor rather than per-strand 99th percentile,
        making it sensitive to outlier spikes and treating both strands with
        the same reference point.
    """
    if torch.max(t) == 0:
        return torch.zeros_like(t)
    beta = x * torch.max(t)
    alpha = (1 / beta) * np.log(1 / y - 1)
    return 1 / (1 + torch.exp(-alpha * (t - beta)))


class Conv1DBlock(torch.nn.Module):
    """
    A small class that applies the conv + reduction operation at the core of the
    Inception module.
    """

    def __init__(self, in_channels, reduction_channels, out_channels, kernel_size):
        super(Conv1DBlock, self).__init__()
        self.reduce = torch.nn.Conv1d(in_channels, reduction_channels, kernel_size=1)
        self.conv = torch.nn.Conv1d(
            reduction_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relus = [torch.nn.ReLU(), torch.nn.ReLU()]

    def forward(self, X):
        return self.relus[1](self.bn(self.conv(self.relus[0](self.reduce(X)))))


class DoubleConv1D(torch.nn.Module):
    """Two rounds of Conv1d (same-pad) -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv1D, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padding.")
        pad = kernel_size // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(torch.nn.Module):
    """DoubleConv1D then MaxPool1d(2). Returns (skip, pooled)."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv1D(in_channels, out_channels, kernel_size)
        self.pool = torch.nn.MaxPool1d(2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(torch.nn.Module):
    """Upsample via ConvTranspose1d, concat skip, then DoubleConv1D."""

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3):
        super(DecoderBlock, self).__init__()
        self.up = torch.nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv1D(in_channels + skip_channels, out_channels, kernel_size)

    @staticmethod
    def _pad_to_match(x, skip):
        """Right-pad or right-crop x by ≤1 to match skip's length."""
        diff = skip.shape[2] - x.shape[2]
        if diff > 0:
            x = torch.nn.functional.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, : skip.shape[2]]
        return x

    def forward(self, x, skip):
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TROGDOR(torch.nn.Module):
    """
    An asymmetric 1D U-Net for TSS prediction from stranded nascent RNA
    sequencing data.

    The encoder downsamples in two stages: ``n_out = log2(output_stride)``
    outer levels reduce the signal from full resolution to output resolution
    (their skip connections are discarded), then ``context_depth`` inner
    levels further expand the receptive field (their skip connections are
    retained for the decoder). The decoder only upsamples back to output
    resolution, not full 1-bp resolution.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (strands). Default is 2.
    base_channels : int, optional
        Number of channels after the stem conv; doubles at each encoder level
        up to max_channels. Default is 32.
    output_stride : int, optional
        Downsampling factor for the output; must be a power of 2. The model
        returns one score per ``output_stride`` input positions. Default is 16.
    context_depth : int, optional
        Number of additional encoder/decoder levels below output resolution,
        expanding the receptive field. Default is 4.
    max_channels : int, optional
        Channel cap to prevent memory explosion in deep encoder levels.
        Default is 512.
    kernel_size : int, optional
        Odd kernel size used in DoubleConv1D blocks. Default is 3.
    name : str, optional
        Prefix for saved model and log files. Default is "TROGDOR".
    verbose : bool, optional
        Whether to print training statistics. Default is True.
    """

    def __init__(
        self,
        in_channels=2,
        base_channels=32,
        output_stride=16,
        context_depth=4,
        max_channels=512,
        kernel_size=3,
        name="TROGDOR",
        verbose=True,
        pos_weight=None,
    ):
        super(TROGDOR, self).__init__()
        self.name = name
        self.output_stride = output_stride

        n_out = int(math.log2(output_stride))
        if 2**n_out != output_stride:
            raise ValueError(f"output_stride must be a power of 2, got {output_stride}")

        # Stem
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            torch.nn.ReLU(),
        )

        # Outer encoder levels (n_out levels): skip connections are discarded
        self.outer_encoders = torch.nn.ModuleList()
        ch = base_channels
        for i in range(n_out):
            out_ch = min(base_channels * (2**i), max_channels)
            self.outer_encoders.append(EncoderBlock(ch, out_ch, kernel_size))
            ch = out_ch

        # Inner encoder levels (context_depth levels): skip connections are kept
        self.inner_encoders = torch.nn.ModuleList()
        for i in range(n_out, n_out + context_depth):
            out_ch = min(base_channels * (2**i), max_channels)
            self.inner_encoders.append(EncoderBlock(ch, out_ch, kernel_size))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = min(
            base_channels * (2 ** (n_out + context_depth)), max_channels
        )
        self.bottleneck = DoubleConv1D(ch, bottleneck_ch, kernel_size)
        ch = bottleneck_ch

        # Decoder (context_depth levels back to output resolution)
        inner_out_channels = [
            min(base_channels * (2 ** (n_out + j)), max_channels)
            for j in range(context_depth)
        ]
        self.decoders = torch.nn.ModuleList()
        for j in range(context_depth - 1, -1, -1):
            skip_ch = inner_out_channels[j]
            out_ch = (
                skip_ch
                if j > 0
                else int(min(base_channels * (2 ** (n_out - 1)), max_channels))
            )
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch, kernel_size))
            ch = out_ch

        # Output head
        self.head = torch.nn.Conv1d(ch, 1, kernel_size=1)

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor([pos_weight]))
            self.loss = BCEWithLogitsLoss(reduction="none", pos_weight=self._pos_weight)
        else:
            self._pos_weight = None
            self.loss = BCEWithLogitsLoss(reduction="none")
        self.logger = Logger(
            [
                "Epoch",
                "Iteration",
                "Train_Time",
                "Val_Time",
                "Train_BCE",
                "Val_BCE",
                "Val_AUPRC",
                "Val_Dice",
                "Saved?",
            ],
            verbose=verbose,
        )

    def forward(self, X):
        """
        Parameters
        ----------
        X : torch.Tensor, shape=(B, 2, L)
            Stranded nascent RNA sequencing data, logistically standardized.

        Returns
        -------
        y : torch.Tensor, shape=(B, 1, L // output_stride)
            Per-bin logit scores at output resolution.
        """
        X = self.stem(X)

        for enc in self.outer_encoders:
            _, X = enc(X)  # discard skip

        inner_skips = []
        for enc in self.inner_encoders:
            skip, X = enc(X)
            inner_skips.append(skip)

        X = self.bottleneck(X)

        for dec, skip in zip(self.decoders, reversed(inner_skips)):
            X = dec(X, skip)

        return self.head(X)

    def fit(
        self,
        train_data,
        optimizer,
        val_data,
        max_epochs=100,
        batch_size=64,
        early_stopping=None,
        verbose=True,
        wandb_run=None,
        bf16=True,
        scheduler=None,
    ):
        """
        Fit the model to data and validate it periodically.

        Parameters
        ----------
        train_data : torch.utils.data.DataLoader
            DataLoader yielding (X, y) where X is (B, 2, L) and y is (B, 1, L // output_stride).
        optimizer : torch.optim.Optimizer
            Optimizer for training.
        val_data : torch.utils.data.DataLoader
            DataLoader for validation.
        max_epochs : int
            Maximum number of training epochs. Default is 100.
        batch_size : int
            Batch size for prediction during validation. Default is 64.
        early_stopping : int or None
            Stop after this many epochs without improvement. Default is None.
        verbose : bool
            Whether to print training statistics. Default is True.
        wandb_run : wandb.sdk.wandb_run.Run or None
            An active wandb run to log metrics to. If None, wandb logging is
            skipped. Default is None.
        bf16 : bool
            Use bfloat16 mixed precision via torch.autocast. No GradScaler is
            needed. Set to False to train in float32. Default is True.
        scheduler : torch.optim.lr_scheduler.LRScheduler or None
            Learning rate scheduler stepped once per batch after each
            ``optimizer.step()``. Pass any PyTorch scheduler; ``None`` keeps
            the optimizer's LR fixed. Default is None.
        """
        iteration = 0
        early_stop_count = 0
        best_loss = float("inf")
        self.logger.start()

        if self._pos_weight is not None:
            self.loss = BCEWithLogitsLoss(reduction="none", pos_weight=self._pos_weight)

        for epoch in range(max_epochs):
            tic = time.time()
            epoch_loss = 0.0
            epoch_batches = 0

            for data in train_data:
                X, y = data
                X, y = X.cuda(), y.cuda()

                optimizer.zero_grad()
                self.train()

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    logits = self(X)
                    loss = self.loss(logits, y).mean()

                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/bce": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=iteration,
                    )

                epoch_loss += loss.item()
                epoch_batches += 1
                iteration += 1

            train_time = time.time() - tic
            train_loss_avg = epoch_loss / max(epoch_batches, 1)

            if verbose:
                with torch.no_grad():
                    self.eval()

                    tic = time.time()
                    y_val = []
                    logits_val = []
                    for X_val, y_val_ in val_data:
                        logits_val_ = predict(
                            self,
                            X_val,
                            batch_size=batch_size,
                            device="cuda",
                            dtype=torch.bfloat16 if bf16 else None,
                        )
                        logits_val.append(logits_val_)
                        y_val.append(y_val_)

                    y_val = torch.cat(y_val)
                    logits_val = torch.cat(logits_val)

                    pos_weight = (
                        self._pos_weight.cpu() if self._pos_weight is not None else None
                    )
                    val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_val, y_val, pos_weight=pos_weight, reduction="none"
                    )

                    logits_flat = torch.sigmoid(logits_val.reshape(-1))
                    y_flat = y_val.reshape(-1).long()
                    val_auprc = binary_auprc(logits_flat, y_flat)

                    preds_flat = (logits_flat > 0.5).float()
                    y_flat_f = y_flat.float()
                    tp = (preds_flat * y_flat_f).sum()
                    val_dice = (2 * tp) / (preds_flat.sum() + y_flat_f.sum() + 1e-8)

                    val_time = time.time() - tic
                    val_loss_ = val_loss.mean().item()

                    self.logger.add(
                        [
                            epoch,
                            iteration,
                            train_time,
                            val_time,
                            train_loss_avg,
                            val_loss_,
                            val_auprc.item(),
                            val_dice.item(),
                            (val_loss_ < best_loss),
                        ]
                    )
                    self.logger.save(f"{self.name}.log")

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/bce": val_loss_,
                                "val/auprc": val_auprc.item(),
                                "val/dice": val_dice.item(),
                            },
                            step=iteration,
                        )

                    if val_loss_ < best_loss:
                        torch.save(self.state_dict(), f"{self.name}.torch")
                        best_loss = val_loss_
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stopping is not None and early_stop_count >= early_stopping:
                break

        torch.save(self, f"{self.name}.final.torch")
