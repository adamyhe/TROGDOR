# data_transforms.py
# Author: Adam He <adamyhe@gmail.com>

"""
Data transformation functions for pre-processing nascent RNA coverage tensors.
"""

import numpy as np
import torch


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
    This implementation is closer in spirit to the original standardization
    described in Danko et al. 2015 and Wang et al. 2019.

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
