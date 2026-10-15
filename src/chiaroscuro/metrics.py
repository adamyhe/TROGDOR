# metrics.py
# Author: Adam He <adamyhe@gmail.com>

"""
Binary classification metrics for validation and benchmarking.

These replace torcheval's ``binary_auprc`` / ``binary_auroc`` with the same
definitions: tied scores are evaluated as a single threshold.
"""

import torch


def _cumulative_counts(input, target):
    """Cumulative true/false positive counts at each distinct threshold.

    Thresholds run from the highest score down; tied scores form one
    threshold, so their counts are taken after the last of them.
    """
    input = torch.as_tensor(input).reshape(-1)
    target = torch.as_tensor(target, device=input.device).reshape(-1)
    if input.shape != target.shape:
        raise ValueError(
            f"input and target must have the same number of elements, "
            f"got {input.numel()} and {target.numel()}."
        )
    scores, order = input.sort(descending=True)
    sorted_target = target[order].to(torch.float64)
    last = torch.ones_like(scores, dtype=torch.bool)
    last[:-1] = scores[1:] != scores[:-1]
    tp = sorted_target.cumsum(0)[last]
    fp = (1 - sorted_target).cumsum(0)[last]
    return tp, fp


def binary_auprc(input, target):
    """Area under the precision-recall curve (average precision).

    Computed as the sum over distinct thresholds of precision times the gain
    in recall, as in scikit-learn's ``average_precision_score``. No linear
    interpolation is applied.

    Parameters
    ----------
    input : torch.Tensor
        Predicted scores or probabilities, any shape.
    target : torch.Tensor
        Binary labels (0/1) with the same number of elements as ``input``.

    Returns
    -------
    torch.Tensor
        Scalar float64 tensor; NaN if ``target`` has no positives.
    """
    tp, fp = _cumulative_counts(input, target)
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    recall_gain = torch.diff(recall, prepend=recall.new_zeros(1))
    return (recall_gain * precision).sum()


def binary_auroc(input, target):
    """Area under the ROC curve, using the trapezoidal rule.

    Parameters
    ----------
    input : torch.Tensor
        Predicted scores or probabilities, any shape.
    target : torch.Tensor
        Binary labels (0/1) with the same number of elements as ``input``.

    Returns
    -------
    torch.Tensor
        Scalar float64 tensor; 0.5 if ``target`` is all ones or all zeros.
    """
    tp, fp = _cumulative_counts(input, target)
    n_pos, n_neg = tp[-1], fp[-1]
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5, dtype=torch.float64, device=tp.device)
    zero = tp.new_zeros(1)
    tp = torch.cat([zero, tp])
    fp = torch.cat([zero, fp])
    return torch.trapezoid(tp, fp) / (n_pos * n_neg)
