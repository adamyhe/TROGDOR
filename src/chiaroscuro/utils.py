# utils.py
# Author: Adam He <adamyhe@gmail.com>

"""
A bunch of utility functions used for predicting and benchmarking TROGDOR.
"""

import numpy as np
import torch

from chiaroscuro.trogdor import TROGDOR, TROGDORResidual


def load_model(path, device):
    state = torch.load(path, weights_only=True, map_location="cpu")
    if _is_residual_state_dict(state):
        model = TROGDORResidual(**_infer_residual_kwargs(state), verbose=False)
    else:
        model = TROGDOR(**_infer_trogdor_kwargs(state), verbose=False)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _is_residual_state_dict(state):
    """Return True when a checkpoint looks like TROGDORResidual."""
    if "_activation_id" in state or "_bottleneck_dilations" in state:
        return True
    return any(".conv1.weight" in key for key in state)


def _infer_residual_kwargs(state):
    """Infer TROGDORResidual constructor kwargs from a state dict."""
    stem_weight = state["stem.0.weight"]
    stem_in_channels = stem_weight.shape[1]
    use_strand_features = _metadata_bool(
        state, "_use_strand_features", default=(stem_in_channels == 4)
    )

    n_outer = _indexed_module_count(state, "outer_encoders")
    n_inner = _indexed_module_count(state, "inner_encoders")
    kernel_size = _infer_kernel_size(state)

    return {
        "in_channels": 2 if use_strand_features else stem_in_channels,
        "base_channels": stem_weight.shape[0],
        "output_stride": 2**n_outer,
        "context_depth": n_inner,
        "max_channels": _infer_max_channels(state),
        "kernel_size": kernel_size,
        "activation": _infer_activation(state),
        "use_strand_features": use_strand_features,
        "bottleneck_dilations": _infer_bottleneck_dilations(state),
    }


def _infer_trogdor_kwargs(state):
    """Infer TROGDOR constructor kwargs from a state dict."""
    stem_weight = state["stem.0.weight"]
    n_outer = _indexed_module_count(state, "outer_encoders")
    n_inner = _indexed_module_count(state, "inner_encoders")

    return {
        "in_channels": stem_weight.shape[1],
        "base_channels": stem_weight.shape[0],
        "output_stride": 2**n_outer,
        "context_depth": n_inner,
        "max_channels": _infer_max_channels(state),
        "kernel_size": _infer_trogdor_kernel_size(state),
    }


def _indexed_module_count(state, prefix):
    indices = set()
    prefix = f"{prefix}."
    for key in state:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            indices.add(int(rest.split(".", 1)[0]))
    return len(indices)


def _infer_kernel_size(state):
    for key in (
        "outer_encoders.0.conv.conv1.weight",
        "inner_encoders.0.conv.conv1.weight",
        "head.0.conv1.weight",
    ):
        if key in state:
            return state[key].shape[2]
    return 3


def _infer_trogdor_kernel_size(state):
    for key in (
        "outer_encoders.0.conv.block.0.weight",
        "inner_encoders.0.conv.block.0.weight",
        "bottleneck.block.0.weight",
    ):
        if key in state:
            return state[key].shape[2]
    return 3


def _infer_max_channels(state):
    max_channels = 0
    for key, value in state.items():
        if key.endswith(".weight") and value.ndim == 3:
            max_channels = max(max_channels, value.shape[0])
    return max_channels or 512


def _infer_activation(state):
    activation_names = {0: "relu", 1: "gelu", 2: "silu"}
    if "_activation_id" in state:
        return activation_names[int(state["_activation_id"].item())]
    return "silu"


def _infer_bottleneck_dilations(state):
    if "_bottleneck_dilations" in state:
        return tuple(int(v) for v in state["_bottleneck_dilations"].tolist())

    block_count = _indexed_module_count(state, "bottleneck.1.blocks")
    return tuple(2**i for i in range(block_count))


def _metadata_bool(state, key, default):
    if key in state:
        return bool(int(state[key].item()))
    return default


def merge_intervals(intervals):
    """Merge abutting intervals, keeping the max value across merged spans.

    Expects ``intervals`` to be a sorted list of ``(start, end, value)`` tuples
    where adjacent intervals share an endpoint (``prev_end == next_start``).
    Overlapping intervals are not expected and not handled.

    Parameters
    ----------
    intervals : list of (int, int, float)
        Sorted (start, end, value) tuples.

    Returns
    -------
    list of [int, int, float]
        Merged intervals as mutable lists.
    """
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e, v in intervals[1:]:
        if s == merged[-1][1]:
            merged[-1][1] = e
            merged[-1][2] = max(merged[-1][2], v)
        else:
            merged.append([s, e, v])
    return merged


def encode_labels(peaks_df, chrom, chrom_len, output_stride):
    """Return a float32 binary array of length chrom_len // output_stride."""
    n_bins = chrom_len // output_stride
    labels = np.zeros(n_bins, dtype=np.float32)
    chrom_peaks = peaks_df[peaks_df["chrom"] == chrom]
    for _, row in chrom_peaks.iterrows():
        start_bin = max(0, int(row["start"]) // output_stride)
        end_bin = min(n_bins, (int(row["end"]) - 1) // output_stride + 1)
        if start_bin < end_bin:
            labels[start_bin:end_bin] = 1.0
    return labels
