#!/usr/bin/env python3
"""Genome-wide benchmarking script for TROGDOR.

Loads a trained model, scores the whole genome in strided bins, converts
ground-truth peak BED regions to per-bin labels, and reports AUROC/AUPRC.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pybigtools
import torch
from torcheval.metrics.functional import binary_auprc, binary_auroc
from tqdm import tqdm

# sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "src"))
from chiaroscuro.predict import predict_chromosome
from chiaroscuro.trogdor import TROGDOR, normalization, standardization


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
        "--standardization",
        action="store_true",
        help="Use the deprecated global-max standardization instead of per-strand normalization",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-chromosome progress"
    )
    return p.parse_args()


def load_model(path, device):
    model = TROGDOR(verbose=False)
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model


def encode_labels(peaks_df, chrom, chrom_len, output_stride):
    """Return a float32 binary array of length chrom_len // output_stride."""
    n_bins = chrom_len // output_stride
    labels = np.zeros(n_bins, dtype=np.float32)
    chrom_peaks = peaks_df[peaks_df["chrom"] == chrom]
    for _, row in chrom_peaks.iterrows():
        start_bin = int(row["start"]) // output_stride
        end_bin = (int(row["end"]) - 1) // output_stride + 1
        start_bin = max(0, start_bin)
        end_bin = min(n_bins, end_bin)
        if start_bin < end_bin:
            labels[start_bin:end_bin] = 1.0
    return labels


def main():
    args = parse_args()

    transform = standardization if args.standardization else normalization
    model = load_model(args.model, args.device)

    pl_bw = pybigtools.open(args.pl_bigwig)
    mn_bw = pybigtools.open(args.mn_bigwig)

    chrom_sizes = pl_bw.chroms()
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())

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

    for chrom in tqdm(chroms, desc="Chromosomes", unit="chr"):
        if chrom not in chrom_sizes:
            tqdm.write(f"  Skipping {chrom} (not in bigWig)")
            continue

        chrom_len = chrom_sizes[chrom]

        pl_vals = np.nan_to_num(
            np.array(pl_bw.values(chrom, 0, chrom_len), dtype=np.float32)
        )
        mn_vals = np.abs(
            np.nan_to_num(np.array(mn_bw.values(chrom, 0, chrom_len), dtype=np.float32))
        )

        signal = torch.tensor(np.stack([pl_vals, mn_vals]))  # (2, L)

        with torch.no_grad():
            logits = predict_chromosome(
                model,
                signal,
                output_stride=args.output_stride,
                device=args.device,
                transform=transform,
                verbose=args.verbose,
            )  # (1, L // stride)

        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (n_bins,)

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
