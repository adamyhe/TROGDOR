#!/usr/bin/env python
# TROGDOR command-line tool
# Author: Adam He <adamyhe@gmail.com>

"""
Transcription Run-On Generates Detector Of Regulatory elements (TROGDOR) is a
deep learning method for identifying transcription initiation regions from
nascent RNA sequencing methods such as GRO-seq, PRO-seq, or ChRO-seq. It was
inspired by the dREG/dREG-HD family of SVM-based TIR detectors and uses many
of ideas from those methods. TROGDOR has been reimplemented using a pytorch
backend and is significantly better documented and easier to install than prior
methods. This command-line tool provides a convenient wrapper for the core
TROGDOR model. It takes in bigWig files of nascent RNA sequencing data and
outputs imputed transcription initiation sites.
"""

import argparse

import numpy as np
import pybigtools
import torch

from chiaroscuro.data_transforms import normalization
from chiaroscuro.predict import predict_chromosome, predict_genome
from chiaroscuro.trogdor import TROGDOR

_help = """
The following commands are available:
    pipeline / burninate  Run the full TROGDOR pipeline from raw data to peak calls
    score                 Score positions using a pre-trained TROGDOR model
    call                  Call peaks from scored positions
"""


def _bh_threshold(probs, alpha):
    m = len(probs)
    if m == 0:
        return None
    p = 1.0 - np.asarray(probs, dtype=np.float64)
    sorted_p = np.sort(p)
    passing = np.where(sorted_p <= (np.arange(1, m + 1) / m) * alpha)[0]
    if len(passing) == 0:
        return None
    return float(1.0 - sorted_p[passing[-1]])


def _merge_intervals(intervals):
    if not intervals:
        return []
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s == merged[-1][1]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def cli():
    # Create argument parser

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="The following commands are available:", required=True, dest="cmd"
    )

    # =============================================================================

    # pipeline (alias: burninate)
    parser_pipeline = subparsers.add_parser(
        "pipeline", aliases=["burninate"],
        help="Run the full TROGDOR pipeline from raw data to peak calls"
    )
    parser_pipeline.add_argument(
        "-p",
        "--pl_bigwig",
        required=True,
        type=str,
        help="bigWig file of nascent RNA sequencing data (plus strand)",
    )
    parser_pipeline.add_argument(
        "-m",
        "--mn_bigwig",
        required=True,
        type=str,
        help="bigWig file of nascent RNA sequencing data (minus strand)",
    )
    parser_pipeline.add_argument(
        "-o", "--output", required=True, type=str, help="Output bed file of peak calls"
    )
    parser_pipeline.add_argument(
        "-d", "--device", default="cuda", type=str, help="Backend device for pytorch"
    )
    parser_pipeline.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print progress statements",
    )

    # =============================================================================

    # score
    parser_score = subparsers.add_parser(
        "score", help="Score positions using a pre-trained TROGDOR model"
    )
    parser_score.add_argument(
        "-M",
        "--model",
        required=True,
        type=str,
        help="Path to a TROGDOR model state dict (.torch)",
    )
    parser_score.add_argument(
        "-p",
        "--pl_bigwig",
        required=True,
        type=str,
        help="bigWig file of nascent RNA sequencing data (plus strand)",
    )
    parser_score.add_argument(
        "-m",
        "--mn_bigwig",
        required=True,
        type=str,
        help="bigWig file of nascent RNA sequencing data (minus strand)",
    )
    parser_score.add_argument(
        "-o", "--output", required=True, type=str, help="Output bigWig file of scores"
    )
    parser_score.add_argument(
        "-d", "--device", default="cuda", type=str, help="Backend device for pytorch"
    )
    parser_score.add_argument(
        "--chunk_size",
        type=int,
        default=262144,
        help="Length of each input chunk fed to the model (default: 262144 = 2^18)",
    )
    parser_score.add_argument(
        "--overlap",
        type=int,
        default=32768,
        help="Input positions whose output bins are trimmed from each chunk edge (default: 32768 = 2^15)",
    )
    parser_score.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Output resolution in bp (default: 16). Must be a power of 2.",
    )
    parser_score.add_argument(
        "--chroms",
        nargs="*",
        default=None,
        help="Chromosomes to score (default: all chromosomes in the bigWig)",
    )
    parser_score.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress messages"
    )

    # =============================================================================

    # call peak
    parser_peaks = subparsers.add_parser(
        "peaks", help="Call peaks from the tracks predicted by the refine method"
    )
    parser_peaks.add_argument(
        "-t", "--input", required=True, type=str, help="bigWig file of TROGDOR scores"
    )
    parser_peaks.add_argument(
        "-o", "--output", required=True, type=str, help="Output bed file of peak calls"
    )
    parser_peaks.add_argument(
        "-f",
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="BH FDR threshold for reporting peaks (default: 0.05)",
    )
    parser_peaks.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress messages"
    )

    # =============================================================================

    # Load args
    args = parser.parse_args()

    # Run provided command
    if args.cmd == "score":
        model = TROGDOR()
        model.load_state_dict(
            torch.load(args.model, weights_only=True, map_location="cpu"), strict=False
        )
        model = model.to(args.device).eval()

        pl_bw = pybigtools.open(args.pl_bigwig)
        chrom_sizes = dict(pl_bw.chroms())
        pl_bw.close()

        chroms_to_score = (
            args.chroms if args.chroms is not None else list(chrom_sizes.keys())
        )

        # Build output bigWig header
        out_header = [(c, chrom_sizes[c]) for c in chroms_to_score if c in chrom_sizes]

        out_bw = pybigtools.open(args.output, "w")
        out_bw.write_header(out_header)

        for chrom, chrom_len, probs in predict_genome(
            model,
            args.pl_bigwig,
            args.mn_bigwig,
            chroms=args.chroms,
            output_stride=args.output_stride,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            transform=normalization,
            device=args.device,
            verbose=args.verbose,
        ):
            bin_indices = np.where(probs > 0)[0]
            if len(bin_indices) > 0:
                starts = (bin_indices * args.output_stride).tolist()
                ends = ((bin_indices + 1) * args.output_stride).tolist()
                values = probs[bin_indices].tolist()
                out_bw.add_intervals(chrom, starts, ends, values)

        out_bw.close()

    elif args.cmd == "peaks":
        in_bw = pybigtools.open(args.input)
        chrom_sizes = dict(in_bw.chroms())

        # Pass 1: collect all intervals per chrom and aggregate probs for BH
        chrom_intervals = {}
        all_probs = []
        for chrom, chrom_len in chrom_sizes.items():
            ivals = [
                (s, e, v)
                for s, e, v in in_bw.intervals(chrom, 0, chrom_len)
                if not np.isnan(v)
            ]
            chrom_intervals[chrom] = ivals
            all_probs.extend(v for _, _, v in ivals)
        in_bw.close()

        threshold = _bh_threshold(np.array(all_probs, dtype=np.float64), args.fdr_threshold)

        if args.verbose:
            if threshold is None:
                print("No bins pass BH FDR threshold; writing empty BED file")
            else:
                n_pass = sum(1 for p in all_probs if p >= threshold)
                print(f"BH probability threshold: {threshold:.6f} ({n_pass} bins pass)")

        with open(args.output, "w") as out_bed:
            if threshold is not None:
                for chrom in sorted(chrom_sizes):
                    passing = [
                        (s, e)
                        for s, e, v in chrom_intervals[chrom]
                        if v >= threshold
                    ]
                    for start, end in _merge_intervals(passing):
                        out_bed.write(f"{chrom}\t{start}\t{end}\n")

    elif args.cmd in ("pipeline", "burninate"):
        pass

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
