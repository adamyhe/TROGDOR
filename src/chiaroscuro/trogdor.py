# trogdor.py
# Author: Adam He <adamyhe@gmail.com>

"""
This module defines a neural network that predicts transcription initiation
regions (TIRs) from nascent RNA sequencing data. This task is framed as an
image segmentation problem and uses a 1-D U-Net architecture.
"""

import functools
import math
import time

import torch
from torcheval.metrics.functional import binary_auprc

from .logger import Logger
from .modules import DecoderBlock, DoubleConv1D, EncoderBlock


torch.backends.cudnn.benchmark = True


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
    loss_fn : callable, optional
        Loss function with signature ``(logits, targets) -> scalar``.
        Stored as ``self._loss_fn`` (wrapped with ``functools.partial`` if
        ``loss_kwargs`` is provided). Default is
        ``torch.nn.BCEWithLogitsLoss()``.
    loss_kwargs : dict or None, optional
        Keyword arguments forwarded to ``loss_fn`` via ``functools.partial``.
        Default is None.
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
        loss_fn=None,
        loss_kwargs=None,
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

        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        _kw = loss_kwargs or {}
        self._loss_fn = functools.partial(loss_fn, **_kw) if _kw else loss_fn

        self.logger = Logger(
            [
                "Epoch",
                "Iteration",
                "Train_Time",
                "Val_Time",
                "Train_Loss",
                "Val_Loss",
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
            Stranded nascent RNA sequencing data. These
            data should be normalized using
            `chiaroscuro.data_transforms.normalization`.

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
        loss_fn=None,
        loss_kwargs=None,
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
        loss_fn : callable or None, optional
            Override the instance-level ``self._loss_fn`` for this training
            run. Same signature as the constructor's ``loss_fn``. Default is
            None (use ``self._loss_fn``).
        loss_kwargs : dict or None, optional
            Keyword arguments forwarded to ``loss_fn`` via
            ``functools.partial``. Ignored when ``loss_fn`` is None.
            Default is None.
        """
        iteration = 0
        early_stop_count = 0
        best_auprc = 0.0
        self.logger.start()

        _fn = self._loss_fn
        if loss_fn is not None:
            _kw = loss_kwargs or {}
            _fn = functools.partial(loss_fn, **_kw) if _kw else loss_fn

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
                    if isinstance(_fn, torch.nn.Module):
                        _fn.to(logits.device)
                    loss = _fn(logits, y)

                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
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
                        X_val = X_val.cuda()
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                            logits_val_ = self(X_val)
                        logits_val.append(logits_val_)
                        y_val.append(y_val_.cuda())

                    y_val = torch.cat(y_val)
                    logits_val = torch.cat(logits_val)

                    val_loss_ = _fn(logits_val, y_val).item()

                    logits_flat = torch.sigmoid(logits_val.reshape(-1))
                    y_flat = y_val.reshape(-1).long()
                    val_auprc = binary_auprc(logits_flat, y_flat)

                    preds_flat = (logits_flat > 0.5).float()
                    y_flat_f = y_flat.float()
                    tp = (preds_flat * y_flat_f).sum()
                    val_dice = (2 * tp) / (preds_flat.sum() + y_flat_f.sum() + 1e-8)

                    val_time = time.time() - tic

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
                            (val_auprc.item() > best_auprc),
                        ]
                    )
                    self.logger.save(f"{self.name}.log")

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/loss": val_loss_,
                                "val/auprc": val_auprc.item(),
                                "val/dice": val_dice.item(),
                            },
                            step=iteration,
                        )

                    if val_auprc.item() > best_auprc:
                        torch.save(self.state_dict(), f"{self.name}.torch")
                        best_auprc = val_auprc.item()
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stopping is not None and early_stop_count >= early_stopping:
                break

        torch.save(self.state_dict(), f"{self.name}.final.torch")
