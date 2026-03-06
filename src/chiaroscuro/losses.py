# losses.py
# Author: Adam He <adamyhe@gmail.com>

"""
Loss functions for training TROGDOR.
"""

import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.99, gamma=2.0):
    """Alpha-balanced focal loss (Lin et al. 2017).

    Down-weights easy examples via (1 - p_t)^gamma, focusing training on
    hard and uncertain bins. alpha weights the positive class.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, smooth=1.0):
    """Tversky loss for imbalanced segmentation.

    alpha weights FP, beta weights FN. beta > alpha penalises missed
    positives more than false alarms, which is appropriate when
    positives are rare and recall matters.
    """
    p = torch.sigmoid(logits)
    dims = list(range(1, p.dim()))   # all dims except batch
    tp = (p * targets).sum(dim=dims)
    fp = (p * (1 - targets)).sum(dim=dims)
    fn = ((1 - p) * targets).sum(dim=dims)
    return (1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def focal_tversky_loss(logits, targets, alpha=0.3, beta=0.7, gamma=3 / 4, smooth=1.0):
    """Focal Tversky loss (Abraham & Khan 2019, arXiv:1810.07842).

    Raises the Tversky loss to the power 1/gamma. When gamma > 1 the loss is
    magnified for all imperfect predictions, emphasising hard and missed
    regions during training.
    """
    tl = tversky_loss(logits, targets, alpha=alpha, beta=beta, smooth=smooth)
    return tl ** (1.0 / gamma)
