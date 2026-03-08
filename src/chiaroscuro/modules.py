# modules.py
# Author: Adam He <adamyhe@gmail.com>

"""
Reusable torch.nn modules for the TROGDOR U-Net architecture.
"""

import torch
import torch.nn.functional as F


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
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, : skip.shape[2]]
        return x

    def forward(self, x, skip):
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
