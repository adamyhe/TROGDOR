#!/usr/bin/env python3
"""Genome-wide benchmarking script for TROGDOR.

Loads a trained model, scores the whole genome in strided bins, converts
ground-truth peak BED regions to per-bin labels, and reports AUROC/AUPRC.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
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
        default=64,
        help="Number of chunks per forward pass (default: 64)",
    )
    p.add_argument(
        "--standardization",
        action="store_true",
        help="Use the deprecated global-max standardization instead of per-strand normalization",
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


if __name__ == "__main__":
    main()
