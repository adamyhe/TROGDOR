#!/usr/bin/env python3
# benchmark_tile_position.py
# Author: Adam He <adamyhe@gmail.com>

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
import pybigtools
import torch
from torcheval.metrics.functional import binary_auprc

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
    p.add_argument(
        "-M", "--model", required=True, help="Path to model .torch state dict"
    )
    p.add_argument("-p", "--pl_bigwig", required=True, help="Plus-strand bigWig")
    p.add_argument("-m", "--mn_bigwig", required=True, help="Minus-strand bigWig")
    p.add_argument("-t", "--peaks", required=True, help="Ground-truth peak BED file")
    p.add_argument("-d", "--device", default="cuda", help="Device (default: cuda)")
    p.add_argument(
        "--chunk_size",
        type=int,
        default=262144,
        help="Input chunk size (default: 262144)",
    )
    p.add_argument(
        "--overlap", type=int, default=32768, help="Edge overlap in bp (default: 32768)"
    )
    p.add_argument(
        "--output_stride", type=int, default=16, help="Output stride (default: 16)"
    )
    p.add_argument("--chroms", nargs="+", default=None, help="Chromosome whitelist")
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of chunks per forward pass (default: 8)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--diagram",
        action="store_true",
        help="Print tiling geometry diagram and exit (no model/data needed)",
    )
    return p.parse_args()


def print_diagram(chunk_size, overlap, output_stride):
    """Print an ASCII diagram of the tiling geometry in output-bin space."""
    out_chunk = chunk_size // output_stride
    out_overlap = overlap // output_stride
    width = 72

    def bar(start, end, total, char, fill=" "):
        lo = round(start * width / total)
        hi = round(end * width / total)
        return fill * lo + char * (hi - lo) + fill * (width - hi)

    def mark(pos, total, label):
        idx = round(pos * width / total)
        row = [" "] * width
        for j, c in enumerate(label):
            if idx + j < width:
                row[idx + j] = c
        return "".join(row)

    # Chunk i spans [0, out_chunk); chunk i+1 starts at stride = out_chunk - 2*out_overlap
    stride_bins = out_chunk - 2 * out_overlap
    total = out_chunk + stride_bins  # enough to show both chunks

    print("\nTiling geometry  (output-bin space)")
    print(
        f"  chunk_size={chunk_size:,}  overlap={overlap:,}  output_stride={output_stride}"
    )
    print(
        f"  out_chunk={out_chunk:,}  out_overlap={out_overlap:,}  stride_bins={stride_bins:,}"
    )
    print()

    # Chunk i bar
    ci_start, ci_end = 0, out_chunk
    bar_i = bar(ci_start, ci_end, total, "─")
    print(f"chunk i   |{bar_i}|")

    # Annotations for chunk i: edge zones and interior zone
    mid = out_chunk // 2
    half = out_overlap // 2
    ann = [" "] * width
    # Left edge of chunk i (region B edge)
    for idx in range(round(0 * width / total), round(out_overlap * width / total)):
        ann[idx] = "E"
    # Interior zone
    intr_s = round((mid - half) * width / total)
    intr_e = round((mid + half) * width / total)
    for idx in range(intr_s, intr_e):
        ann[idx] = "I"
    # Right edge of chunk i (region A edge)
    for idx in range(
        round((out_chunk - out_overlap) * width / total),
        round(out_chunk * width / total),
    ):
        ann[idx] = "E"
    print(f"          |{''.join(ann)}|")
    print(f"          |{mark(0, total, 'B←edge')}|")
    print(f"          |{mark(mid - half, total, 'I=interior')}|")
    print(f"          |{mark(out_chunk - out_overlap, total, 'A→edge')}|")
    print()

    # Chunk i+1 bar
    ci1_start = stride_bins
    ci1_end = stride_bins + out_chunk
    bar_i1 = bar(ci1_start, ci1_end, total, "─")
    print(f"chunk i+1 |{bar_i1}|")

    # Annotations for chunk i+1: centre zones (A and B)
    ann2 = [" "] * width
    # Region A as centre of chunk i+1 (left part of chunk i+1)
    for idx in range(
        round(ci1_start * width / total),
        round((ci1_start + out_overlap) * width / total),
    ):
        ann2[idx] = "C"
    # Region B as centre of chunk i+1 (right part of chunk i+1 that overlaps chunk i right edge)
    b_centre_s = ci1_start + out_chunk - 2 * out_overlap
    b_centre_e = b_centre_s + out_overlap
    for idx in range(
        round(b_centre_s * width / total), round(b_centre_e * width / total)
    ):
        ann2[idx] = "C"
    print(f"          |{''.join(ann2)}|")
    print(f"          |{mark(ci1_start, total, 'A→centre')}|")
    print(f"          |{mark(b_centre_s, total, 'B→centre')}|")
    print()
    print("Legend: E=edge bin  C=centre bin  I=interior bin")
    print()


def run_chunks(
    model, signal, chunk_size, overlap, output_stride, batch_size, device, transform
):
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
    if "--diagram" in sys.argv:
        p2 = argparse.ArgumentParser(add_help=False)
        p2.add_argument("--chunk_size", type=int, default=262144)
        p2.add_argument("--overlap", type=int, default=32768)
        p2.add_argument("--output_stride", type=int, default=16)
        known, _ = p2.parse_known_args()
        print_diagram(known.chunk_size, known.overlap, known.output_stride)
        return

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
    intr_preds_all = []
    labels_all = []
    intr_labels_all = []

    pl_bw = pybigtools.open(args.pl_bigwig)
    mn_bw = pybigtools.open(args.mn_bigwig)

    try:
        chrom_sizes = dict(pl_bw.chroms())
        chroms_to_score = (
            args.chroms if args.chroms is not None else list(chrom_sizes.keys())
        )

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
                        print(
                            f"  Skipping {chrom}: only one chunk, no pairs",
                            file=sys.stderr,
                        )
                    continue

                labels = encode_labels(peaks_df, chrom, chrom_len, output_stride)

                for i in range(len(chunks) - 1):
                    out_s_i, p_i = chunks[i]
                    out_s_i1, p_i1 = chunks[i + 1]

                    # Region A: right edge of chunk i  /  centre of chunk i+1
                    glob_a_start = out_s_i + out_chunk - out_overlap
                    glob_a_end = out_s_i + out_chunk
                    edge_preds_all.append(p_i[out_chunk - out_overlap : out_chunk])
                    cent_preds_all.append(
                        p_i1[glob_a_start - out_s_i1 : glob_a_end - out_s_i1]
                    )
                    labels_all.append(labels[glob_a_start:glob_a_end])

                    # Region B: left edge of chunk i+1  /  centre of chunk i
                    glob_b_start = out_s_i1
                    glob_b_end = out_s_i1 + out_overlap
                    edge_preds_all.append(p_i1[0:out_overlap])
                    cent_preds_all.append(
                        p_i[glob_b_start - out_s_i : glob_b_end - out_s_i]
                    )
                    labels_all.append(labels[glob_b_start:glob_b_end])

                # Interior: centre out_overlap bins of each non-boundary chunk
                mid = out_chunk // 2
                half = out_overlap // 2
                for i in range(1, len(chunks) - 1):
                    out_s_i, p_i = chunks[i]
                    loc_start = mid - half
                    loc_end = loc_start + out_overlap
                    glob_start = out_s_i + loc_start
                    glob_end = out_s_i + loc_end
                    intr_preds_all.append(p_i[loc_start:loc_end])
                    intr_labels_all.append(labels[glob_start:glob_end])

    finally:
        pl_bw.close()
        mn_bw.close()

    if not labels_all:
        print(
            "No comparable regions found. Check --chroms and bigWig contents.",
            file=sys.stderr,
        )
        sys.exit(1)

    edge_preds = np.concatenate(edge_preds_all).astype(np.float32)
    cent_preds = np.concatenate(cent_preds_all).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.float32)

    n_bins = len(labels)
    n_pos = int(labels.sum())
    pct = 100.0 * n_pos / n_bins
    n_pairs = len(labels_all) // 2  # 2 regions per chunk pair

    print(
        f"Comparable bins: {out_overlap:,} per region × {len(labels_all)} regions "
        f"({n_pairs} chunk pairs) — {n_bins:,} total, {n_pos:,} positive ({pct:.3f}%)"
    )

    labels_t = torch.tensor(labels).long()
    cent_auprc = binary_auprc(torch.tensor(cent_preds), labels_t).item()
    edge_auprc = binary_auprc(torch.tensor(edge_preds), labels_t).item()

    n_intr_chunks = len(intr_preds_all)
    if intr_preds_all:
        intr_preds = np.concatenate(intr_preds_all).astype(np.float32)
        intr_labels = np.concatenate(intr_labels_all).astype(np.float32)
        intr_auprc = binary_auprc(
            torch.tensor(intr_preds), torch.tensor(intr_labels).long()
        ).item()
        print(f"Interior auPRC: {intr_auprc:.4f}  ({n_intr_chunks} interior chunks)")
    else:
        print("Interior auPRC: N/A  (fewer than 3 chunks per chromosome)")

    print(f"Centre   auPRC: {cent_auprc:.4f}")
    print(f"Edge     auPRC: {edge_auprc:.4f}")


if __name__ == "__main__":
    main()
