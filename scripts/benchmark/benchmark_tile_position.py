#!/usr/bin/env python3
"""Benchmark tile centre vs tile edge auPRC for TROGDOR.

For each pair of consecutive overlapping chunks, the same genomic bins are
predicted once as an edge bin (within `overlap` bp of the chunk boundary) and
once as a centre bin (by the adjacent chunk).  This script collects those
paired predictions and reports auPRC for each role.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from torcheval.metrics.functional import binary_auprc

import pybigtools

from chiaroscuro.data_transforms import normalization
from chiaroscuro.predict import predict
from chiaroscuro.utils import encode_labels, load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark TROGDOR tile centre vs tile edge auPRC."
    )
    p.add_argument("-M", "--model", required=True, help="Path to model .torch state dict")
    p.add_argument("-p", "--pl_bigwig", required=True, help="Plus-strand bigWig")
    p.add_argument("-m", "--mn_bigwig", required=True, help="Minus-strand bigWig")
    p.add_argument("-t", "--peaks", required=True, help="Ground-truth peak BED file")
    p.add_argument("-d", "--device", default="cuda", help="Device (default: cuda)")
    p.add_argument("--chunk_size", type=int, default=262144, help="Input chunk size (default: 262144)")
    p.add_argument("--overlap", type=int, default=32768, help="Edge overlap in bp (default: 32768)")
    p.add_argument("--output_stride", type=int, default=16, help="Output stride (default: 16)")
    p.add_argument("--chroms", nargs="+", default=None, help="Chromosome whitelist")
    p.add_argument("--batch_size", type=int, default=64, help="Number of chunks per forward pass (default: 64)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def run_chunks(model, signal, chunk_size, overlap, output_stride, batch_size, device, transform):
    """Run inference on all chunks of a chromosome.

    Replicates the chunk-building logic from predict_chromosome but returns
    per-chunk results instead of stitching them.

    Returns
    -------
    list of (out_s, probs_1d)
        out_s    : global output bin index of chunk start (= s // output_stride)
        probs_1d : numpy float32 array of shape (out_chunk,)
    """
    total_length = signal.shape[1]
    stride = chunk_size - 2 * overlap

    # Build start positions (same logic as predict_chromosome)
    starts = list(range(0, total_length - chunk_size + 1, stride))
    if not starts or starts[-1] + chunk_size < total_length:
        starts.append(total_length - chunk_size)

    # Build batch tensor (with per-chunk transform)
    chunks = []
    for s in starts:
        chunk = signal[:, s : s + chunk_size]
        if transform is not None:
            chunk = transform(chunk)
        chunks.append(chunk)
    X_all = torch.stack(chunks)  # (n_chunks, 2, chunk_size)

    # Batched inference — returns (n_chunks, 1, out_chunk) on CPU
    preds = predict(model, X_all, batch_size=batch_size, device=device, verbose=False)
    preds = torch.sigmoid(preds).squeeze(1).numpy()  # (n_chunks, out_chunk)

    out_chunk = chunk_size // output_stride
    result = []
    for i, s in enumerate(starts):
        out_s = s // output_stride
        result.append((out_s, preds[i]))  # preds[i] shape: (out_chunk,)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    model = load_model(args.model, args.device)

    peaks_df = pd.read_csv(
        args.peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
    )

    chunk_size = args.chunk_size
    overlap = args.overlap
    output_stride = args.output_stride
    out_chunk = chunk_size // output_stride
    out_overlap = overlap // output_stride

    edge_preds_all = []
    cent_preds_all = []
    labels_all = []

    pl_bw = pybigtools.open(args.pl_bigwig)
    mn_bw = pybigtools.open(args.mn_bigwig)

    try:
        chrom_sizes = dict(pl_bw.chroms())
        chroms_to_score = args.chroms if args.chroms is not None else list(chrom_sizes.keys())

        with torch.no_grad():
            for chrom in chroms_to_score:
                if chrom not in chrom_sizes:
                    if args.verbose:
                        print(f"Skipping {chrom}: not in bigWig", file=sys.stderr)
                    continue

                chrom_len = chrom_sizes[chrom]
                if chrom_len < chunk_size:
                    if args.verbose:
                        print(
                            f"Skipping {chrom}: length {chrom_len} bp < chunk_size {chunk_size}",
                            file=sys.stderr,
                        )
                    continue

                if args.verbose:
                    print(f"Processing {chrom} ({chrom_len:,} bp) ...", file=sys.stderr)

                pl_vals = np.nan_to_num(
                    np.array(pl_bw.values(chrom, 0, chrom_len), dtype=np.float32)
                )
                mn_vals = np.abs(
                    np.nan_to_num(
                        np.array(mn_bw.values(chrom, 0, chrom_len), dtype=np.float32)
                    )
                )
                signal = torch.tensor(np.stack([pl_vals, mn_vals]))

                chunks = run_chunks(
                    model,
                    signal,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    output_stride=output_stride,
                    batch_size=args.batch_size,
                    device=args.device,
                    transform=normalization,
                )

                if len(chunks) < 2:
                    if args.verbose:
                        print(f"  Skipping {chrom}: only one chunk, no pairs", file=sys.stderr)
                    continue

                labels = encode_labels(peaks_df, chrom, chrom_len, output_stride)

                for i in range(len(chunks) - 1):
                    out_s_i,  p_i  = chunks[i]
                    out_s_i1, p_i1 = chunks[i + 1]

                    # Region A: right edge of chunk i  /  centre of chunk i+1
                    glob_a_start = out_s_i + out_chunk - out_overlap
                    glob_a_end   = out_s_i + out_chunk
                    edge_preds_all.append(p_i[out_chunk - out_overlap : out_chunk])
                    cent_preds_all.append(p_i1[glob_a_start - out_s_i1 : glob_a_end - out_s_i1])
                    labels_all.append(labels[glob_a_start:glob_a_end])

                    # Region B: left edge of chunk i+1  /  centre of chunk i
                    glob_b_start = out_s_i1
                    glob_b_end   = out_s_i1 + out_overlap
                    edge_preds_all.append(p_i1[0:out_overlap])
                    cent_preds_all.append(p_i[glob_b_start - out_s_i : glob_b_end - out_s_i])
                    labels_all.append(labels[glob_b_start:glob_b_end])

    finally:
        pl_bw.close()
        mn_bw.close()

    if not labels_all:
        print("No comparable regions found. Check --chroms and bigWig contents.", file=sys.stderr)
        sys.exit(1)

    edge_preds = np.concatenate(edge_preds_all).astype(np.float32)
    cent_preds = np.concatenate(cent_preds_all).astype(np.float32)
    labels     = np.concatenate(labels_all).astype(np.float32)

    n_bins = len(labels)
    n_pos  = int(labels.sum())
    pct    = 100.0 * n_pos / n_bins
    n_pairs = len(labels_all) // 2  # 2 regions per chunk pair

    print(f"Comparable bins: {out_overlap:,} per region × {len(labels_all)} regions "
          f"({n_pairs} chunk pairs) — {n_bins:,} total, {n_pos:,} positive ({pct:.3f}%)")

    labels_t     = torch.tensor(labels).long()
    cent_auprc   = binary_auprc(torch.tensor(cent_preds), labels_t).item()
    edge_auprc   = binary_auprc(torch.tensor(edge_preds), labels_t).item()

    print(f"Centre auPRC: {cent_auprc:.4f}")
    print(f"Edge   auPRC: {edge_auprc:.4f}")


if __name__ == "__main__":
    main()
