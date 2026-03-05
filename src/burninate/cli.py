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

from burninate.predict import predict_chromosome
from burninate.trogdor import load_pretrained_model, normalization

_help = """
The following commands are available:
    pipeline  Run the full TROGDOR pipeline from raw data to peak calls
    score     Score positions using a pre-trained TROGDOR model
    call      Call peaks from scored positions
"""


def chiaroscuro():
    # Create argument parser

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="The following commands are available:", required=True, dest="cmd"
    )

    # =============================================================================

    # pipeline
    parser_pipeline = subparsers.add_parser(
        "pipeline", help="Run the full TROGDOR pipeline from raw data to peak calls"
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
        "-i",
        "--input",
        required=False,
        default=None,
        type=str,
        help="Input BED file of genomic positions (optional; ignored for dense scoring)",
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
        model = load_pretrained_model()
        model = model.to(args.device).eval()

        pl_bw = pybigtools.open(args.pl_bigwig)
        mn_bw = pybigtools.open(args.mn_bigwig)

        chrom_sizes = dict(pl_bw.chroms())

        chroms_to_score = args.chroms if args.chroms is not None else list(chrom_sizes.keys())

        # Build output bigWig header
        out_header = [(c, chrom_sizes[c]) for c in chroms_to_score if c in chrom_sizes]

        out_bw = pybigtools.open(args.output, "w")
        out_bw.write_header(out_header)

        for chrom in chroms_to_score:
            if chrom not in chrom_sizes:
                if args.verbose:
                    print(f"Skipping {chrom}: not in bigWig")
                continue

            chrom_len = chrom_sizes[chrom]

            if chrom_len < args.chunk_size:
                if args.verbose:
                    print(
                        f"Skipping {chrom}: length {chrom_len} bp is shorter than "
                        f"--chunk_size {args.chunk_size}. Re-run with --chunk_size <= {chrom_len} to score it."
                    )
                continue

            if args.verbose:
                print(f"Scoring {chrom} ({chrom_len} bp)...")

            pl_vals = np.nan_to_num(
                np.array(pl_bw.values(chrom, 0, chrom_len), dtype=np.float32)
            )
            mn_vals = np.abs(np.nan_to_num(
                np.array(mn_bw.values(chrom, 0, chrom_len), dtype=np.float32)
            ))

            signal = torch.from_numpy(np.stack([pl_vals, mn_vals])).float()

            logits = predict_chromosome(
                model,
                signal,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                output_stride=args.output_stride,
                device=args.device,
                transform=normalization,
            )

            probs = torch.sigmoid(logits).squeeze(0).numpy()  # (chrom_len // output_stride,)

            # Write non-zero values as intervals; each bin i maps to [i*output_stride, (i+1)*output_stride)
            bin_indices = np.where(probs > 0)[0]
            if len(bin_indices) > 0:
                starts = (bin_indices * args.output_stride).tolist()
                ends = ((bin_indices + 1) * args.output_stride).tolist()
                values = probs[bin_indices].tolist()
                out_bw.add_intervals(chrom, starts, ends, values)

        pl_bw.close()
        mn_bw.close()
        out_bw.close()

    elif args.cmd == "peaks":
        pass

    elif args.cmd == "pipeline":
        pass

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    chiaroscuro()
