# utils.py
# Author: Adam He <adamyhe@gmail.com>

"""
A bunch of utility functions used for predicting and benchmarking TROGDOR.
"""

import numpy as np
import torch

from chiaroscuro.trogdor import TROGDOR


def load_model(path, device):
    model = TROGDOR(verbose=False)
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


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
