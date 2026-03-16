#!/usr/bin/env python3
# benchmark.py
# Author: Adam He <adamyhe@gmail.com>

"""Genome-wide benchmarking script for TROGDOR.

Loads a trained model, scores the whole genome in strided bins, converts
ground-truth peak BED regions to per-bin labels, and reports AUROC/AUPRC.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)
from torcheval.metrics.functional import binary_auprc, binary_auroc

# sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "src"))
from chiaroscuro.data_transforms import normalization, standardization
from chiaroscuro.predict import predict_genome
from chiaroscuro.utils import encode_labels, load_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark a trained TROGDOR model against ground-truth peaks."
    )
    p.add_argument(
        "-M", "--model", required=True, help="Path to model .torch state dict"
    )
    p.add_argument("-p", "--pl_bigwig", required=True, help="Plus-strand bigWig")
    p.add_argument("-m", "--mn_bigwig", required=True, help="Minus-strand bigWig")
    p.add_argument("-t", "--peaks", required=True, help="Ground-truth peak BED file")
    p.add_argument("-d", "--device", default="cuda", help="Device (default: cuda)")
    p.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Output resolution in bp (default: 16)",
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all in bigWig)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of chunks per forward pass (default: 8)",
    )
    p.add_argument(
        "--standardization",
        action="store_true",
        help="Use the deprecated global-max standardization instead of per-strand normalization",
    )
    p.add_argument(
        "-o",
        "--output_prefix",
        default=None,
        help="Output prefix for PDF plots (e.g. 'results/sample'); "
        "writes <prefix>.roc.pdf and <prefix>.prc.pdf",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-chromosome progress"
    )
    return p.parse_args()


def main():
    args = parse_args()

    transform = standardization if args.standardization else normalization
    model = load_model(args.model, args.device)

    peaks_df = pd.read_csv(
        args.peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
    )

    all_probs = []
    all_labels = []

    for chrom, chrom_len, probs in predict_genome(
        model,
        args.pl_bigwig,
        args.mn_bigwig,
        chroms=args.chroms,
        output_stride=args.output_stride,
        batch_size=args.batch_size,
        transform=transform,
        device=args.device,
        verbose=args.verbose,
    ):
        labels = encode_labels(peaks_df, chrom, chrom_len, args.output_stride)
        all_probs.append(probs)
        all_labels.append(labels)

    if not all_probs:
        print(
            "No chromosomes scored. Check --chroms and bigWig contents.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    probs_t = torch.tensor(all_probs)
    labels_t = torch.tensor(all_labels).long()

    auroc = binary_auroc(probs_t, labels_t).item()
    auprc = binary_auprc(probs_t, labels_t).item()

    n_total = len(all_labels)
    n_pos = int(all_labels.sum())
    pct = 100.0 * n_pos / n_total

    print(f"Bins:  {n_total:,} total, {n_pos:,} positive ({pct:.3f}%)")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # ROC curve (shared by both tables and plots)
    fpr_arr, tpr_arr, thresholds_roc = roc_curve(all_labels, all_probs)

    # Table 1: threshold at fixed FPR targets
    print("\nThreshold at FPR targets:")
    print(f"  {'FPR':>8}  {'TPR':>6}  {'Threshold':>10}")
    for target_fpr in (0.00001, 0.0001, 0.001, 0.01, 0.05, 0.10):
        idx = np.searchsorted(fpr_arr, target_fpr)
        idx = min(idx, len(fpr_arr) - 1)
        print(
            f"  {fpr_arr[idx]:>8.4%}  {tpr_arr[idx]:>6.1%}  {thresholds_roc[idx]:>10.6f}"
        )

    # Table 2: FPR/TPR at fixed score thresholds
    # thresholds_roc is decreasing; searchsorted needs ascending order
    thresholds_desc = thresholds_roc
    thresholds_asc = thresholds_desc[::-1]
    fpr_asc = fpr_arr[::-1]
    tpr_asc = tpr_arr[::-1]
    print("\nFPR/TPR at score thresholds:")
    print(f"  {'Threshold':>10}  {'FPR':>8}  {'TPR':>6}")
    for target_thr in (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9):
        idx = np.searchsorted(thresholds_asc, target_thr, side="left")
        idx = min(idx, len(thresholds_asc) - 1)
        print(
            f"  {thresholds_asc[idx]:>10.6f}  {fpr_asc[idx]:>8.4%}  {tpr_asc[idx]:>6.1%}"
        )

    if args.output_prefix:
        precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_probs)

        # ROC plot
        fig, ax = plt.subplots()
        ax.plot(fpr_arr, tpr_arr, lw=1.5, label=f"AUROC = {auroc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{args.output_prefix}.roc.pdf")
        plt.close(fig)

        # PRC plot
        fig, ax = plt.subplots()
        ax.plot(recall_arr, precision_arr, lw=1.5, label=f"AUPRC = {auprc:.4f}")
        ax.axhline(pct / 100, color="k", linestyle="--", lw=0.8, label="Baseline")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curve")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(f"{args.output_prefix}.prc.pdf")
        plt.close(fig)

        print(f"\nPlots written to {args.output_prefix}.roc.pdf / .prc.pdf")


if __name__ == "__main__":
    main()
