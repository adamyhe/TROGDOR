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

from chiaroscuro.commands import cmd_peaks, cmd_pipeline, cmd_score

_help = """
The following commands are available:
    pipeline / burninate      Run the full TROGDOR pipeline from raw data to peak calls
    score / thatch            Score positions using a pre-trained TROGDOR model
    peaks / consummate_vs     Call peaks from scored positions
"""


def cli():
    # Create argument parser

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="The following commands are available:", required=True, dest="cmd"
    )

    # =============================================================================

    # pipeline (alias: burninate)
    parser_pipeline = subparsers.add_parser(
        "pipeline",
        aliases=["burninate"],
        help="Run the full TROGDOR pipeline from raw data to peak calls",
    )
    parser_pipeline.add_argument(
        "-M",
        "--model",
        required=False,
        default=None,
        type=str,
        help=(
            "Path to a TROGDOR model state dict (.torch). If omitted, the default "
            "pretrained weights are downloaded from HuggingFace Hub and cached locally."
        ),
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
        "-n",
        "--name",
        required=True,
        type=str,
        help="Output filename prefix for scores bigWig and peak calls.",
    )
    parser_pipeline.add_argument(
        "-d", "--device", default="cuda", type=str, help="Backend device for pytorch"
    )
    parser_pipeline.add_argument(
        "--chunk_size",
        type=int,
        default=262144,
        help="Length of each input chunk fed to the model (default: 262144 = 2^18)",
    )
    parser_pipeline.add_argument(
        "--overlap",
        type=int,
        default=32768,
        help="Input positions whose output bins are trimmed from each chunk edge (default: 32768 = 2^15)",
    )
    parser_pipeline.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Output resolution in bp (default: 16). Must be a power of 2.",
    )
    parser_pipeline.add_argument(
        "--chroms",
        nargs="*",
        default=None,
        help="Chromosomes to score (default: all chromosomes in the bigWig)",
    )
    parser_pipeline.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of chunks per forward pass (default: 8)",
    )
    parser_pipeline.add_argument(
        "-s",
        "--min_score",
        type=float,
        default=0.9,
        help="Minimum raw score to include as a candidate bin before BH correction (default: 0.9)",
    )
    parser_pipeline.add_argument(
        "-f",
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="BH FDR threshold for reporting peaks (default: 0.1)",
    )
    parser_pipeline.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print progress statements",
    )
    parser_pipeline.set_defaults(func=cmd_pipeline)

    # =============================================================================

    # score
    parser_score = subparsers.add_parser(
        "score",
        aliases=["thatch"],
        help="Score positions using a pre-trained TROGDOR model",
    )
    parser_score.add_argument(
        "-M",
        "--model",
        required=False,
        default=None,
        type=str,
        help=(
            "Path to a TROGDOR model state dict (.torch). If omitted, the default "
            "pretrained weights are downloaded from HuggingFace Hub and cached locally."
        ),
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
        "-n",
        "--name",
        required=True,
        type=str,
        help="Output filename prefix; produces {name}.prob.bw and {name}.qval.bw",
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
        "--batch_size",
        type=int,
        default=8,
        help="Number of chunks per forward pass (default: 8)",
    )
    parser_score.add_argument(
        "-s",
        "--min_score",
        type=float,
        default=0.9,
        help="Minimum raw score to include as a candidate bin before BH correction (default: 0.9)",
    )
    parser_score.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress messages"
    )
    parser_score.set_defaults(func=cmd_score)

    # =============================================================================

    # call peak
    parser_peaks = subparsers.add_parser(
        "peaks",
        aliases=["consummate_vs"],
        help="Call peaks from the tracks predicted by the refine method",
    )
    parser_peaks.add_argument(
        "-i", "--input", required=True, type=str, help="bigWig file of TROGDOR scores"
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
    parser_peaks.set_defaults(func=cmd_peaks)

    # =============================================================================

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
