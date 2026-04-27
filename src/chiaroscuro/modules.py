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


class ResidualConv1D(torch.nn.Module):
    """Two Conv1d -> BN layers with a residual projection and activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, activation="silu"):
        super(ResidualConv1D, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padding.")
        pad = kernel_size // 2
        self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=pad
        )
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding=pad
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.proj = (
            torch.nn.Identity()
            if in_channels == out_channels
            else torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = _make_activation(activation)

    def forward(self, x):
        identity = self.proj(x)
        y = self.activation(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.activation(y + identity)


class DilatedResidualStack1D(torch.nn.Module):
    """Residual Conv1d stack with exponentially increasing dilation."""

    def __init__(self, channels, kernel_size=3, dilations=(1, 2, 4, 8), activation="silu"):
        super(DilatedResidualStack1D, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padding.")
        self.blocks = torch.nn.ModuleList()
        for dilation in dilations:
            pad = (kernel_size // 2) * dilation
            self.blocks.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=dilation,
                    ),
                    torch.nn.BatchNorm1d(channels),
                    _make_activation(activation),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=dilation,
                    ),
                    torch.nn.BatchNorm1d(channels),
                )
            )
        self.activation = _make_activation(activation)

    def forward(self, x):
        for block in self.blocks:
            x = self.activation(x + block(x))
        return x


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


class ResidualEncoderBlock(torch.nn.Module):
    """ResidualConv1D then learned stride-2 downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, activation="silu"):
        super(ResidualEncoderBlock, self).__init__()
        self.conv = ResidualConv1D(in_channels, out_channels, kernel_size, activation)
        self.down = torch.nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.down(skip)
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


class ResidualDecoderBlock(torch.nn.Module):
    """ConvTranspose1d upsampling, skip concat, then ResidualConv1D."""

    def __init__(
        self, in_channels, skip_channels, out_channels, kernel_size=3, activation="silu"
    ):
        super(ResidualDecoderBlock, self).__init__()
        self.up = torch.nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = ResidualConv1D(
            in_channels + skip_channels, out_channels, kernel_size, activation
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = DecoderBlock._pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _make_activation(name):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")
