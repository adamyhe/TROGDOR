#!/usr/bin/env python
# TROGDOR command-line tool
# Author: Adam He <adamyhe@gmail.com>

"""
Transcription Run-On Grants Detection Of Regulatory elements (TROGDOR) is a
deep learning method for identifying transcription initiation regions from
nascent RNA sequencing methods such as GRO-seq, PRO-seq, or ChRO-seq. It was
inspired by the dREG/dREG-HD family of SVM-based TIR detectors and uses many
ideas from those methods. TROGDOR has been implemented as a deep neural network
using a PyTorch backend and is significantly better documented and easier to
install than these prior methods. This command-line tool provides a convenient
wrapper for the core TROGDOR model. It takes in bigWig files of nascent RNA
sequencing data and outputs imputed TIRs.
"""

import argparse

from cli.commands import cmd_fdr, cmd_peaks, cmd_pipeline, cmd_score

_help = """
The following commands are available:
    pipeline / burninate      Run the full TROGDOR pipeline from raw data to peak calls
    score / thatch            Score positions using a pre-trained TROGDOR model
    peaks / consummate_vs     Call peaks from scored positions
    fdr / fire_dragon         Calculate empirical FDR (requires ground truth peaks)
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
        help="bigWig file of nascent RNA sequencing data 3'-end coverage (plus strand)",
    )
    parser_pipeline.add_argument(
        "-m",
        "--mn_bigwig",
        required=True,
        type=str,
        help="bigWig file of nascent RNA sequencing data 3'-end coverage (minus strand)",
    )
    parser_pipeline.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output BED file path for peak calls (e.g. sample.peaks.bed.gz).",
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
        help="Number of chunks per forward pass (default: 8). Reduce if OOM.",
    )
    parser_pipeline.add_argument(
        "-s",
        "--min_score",
        type=float,
        default=0.95,
        help="Minimum probability to store/report a bin (default: 0.95)",
    )
    parser_pipeline.add_argument(
        "-b",
        "--save_bigwig",
        default=None,
        type=str,
        help="If provided, save the intermediate probability bigWig to this path instead of a temp file.",
    )
    parser_pipeline.add_argument(
        "--peak_mode",
        choices=["simple", "refined", "profile"],
        default="profile",
        dest="mode",
        help="Peak caller to use after scoring (default: simple).",
    )
    parser_pipeline.add_argument(
        "--max_gap",
        type=int,
        default=0,
        help="Refined/profile caller only: merge candidate blocks separated by at most this many bp.",
    )
    parser_pipeline.add_argument(
        "--min_width",
        type=int,
        default=0,
        help="Refined/profile caller only: discard called peaks narrower than this many bp.",
    )
    parser_pipeline.add_argument(
        "--max_width",
        type=int,
        default=None,
        help="Profile caller only: discard called peaks wider than this many bp.",
    )
    parser_pipeline.add_argument(
        "--seed_score",
        type=float,
        default=None,
        help="Profile caller only: permissive score threshold used to seed candidate blocks.",
    )
    parser_pipeline.add_argument(
        "--smooth_bins",
        type=int,
        default=1,
        help="Profile caller only: number of score bins for local smoothing (default: 1).",
    )
    parser_pipeline.add_argument(
        "--valley_fraction",
        type=float,
        default=0.5,
        help="Profile caller only: split adjacent summits when the valley is below this fraction of the weaker summit.",
    )
    parser_pipeline.add_argument(
        "--boundary_fraction",
        type=float,
        default=0.2,
        help="Profile caller only: trim boundaries to bins above this fraction of the local summit.",
    )
    parser_pipeline.add_argument(
        "--min_support_signal",
        type=float,
        default=0.0,
        help="Refined/profile caller only: require this minimum raw plus/minus signal within each peak (default: 0, disabled).",
    )
    parser_pipeline.add_argument(
        "--calibrate",
        action="store_true",
        help="Estimate an empirical FDR curve from streamed probabilities and write raw plus calibrated peak BEDs.",
    )
    parser_pipeline.add_argument(
        "--raw_output",
        default=None,
        help="Raw peak BED path written with --calibrate before FDR filtering. Defaults to a .raw sibling of --output.",
    )
    parser_pipeline.add_argument(
        "--calibration_fdr_target",
        type=float,
        default=0.05,
        help="Empirical FDR target used with --calibrate (default: 0.05).",
    )
    parser_pipeline.add_argument(
        "--calibration_curve",
        default=None,
        help="Optional TSV path for the streamed empirical FDR curve.",
    )
    parser_pipeline.add_argument(
        "--calibration_stat",
        choices=["max", "mean"],
        default="max",
        help="Per-peak streamed probability statistic used for calibration (default: max).",
    )
    parser_pipeline.add_argument(
        "--null_scope",
        choices=["candidate", "genome"],
        default="candidate",
        help="Null placement scope for --calibrate: thresholded candidate intervals or chromosome-wide (default: candidate).",
    )
    parser_pipeline.add_argument(
        "--n_shuffle",
        type=int,
        default=20,
        help="Number of empirical null shuffles used with --calibrate (default: 20).",
    )
    parser_pipeline.add_argument(
        "--n_thresholds",
        type=int,
        default=200,
        help="Number of thresholds in the empirical FDR curve used with --calibrate (default: 200).",
    )
    parser_pipeline.add_argument(
        "--calibration_seed",
        type=int,
        default=0,
        help="Random seed for streamed empirical calibration (default: 0).",
    )
    parser_pipeline.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for chunk preprocessing (default: 4). "
        "Set to 1–4 on Linux/CUDA for additional throughput.",
    )
    parser_pipeline.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print progress statements",
    )
    parser_pipeline.set_defaults(func=cmd_pipeline)

    # =============================================================================

    # score (alias: thatch)
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
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output bigWig file path (e.g. sample.prob.bw).",
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
        default=0.95,
        help="Storage threshold; bins with raw prob below this are omitted from the output bigWig (default: 0.95)"
        " Should be explicitly set to 0 for use with FDR calculation tool (in which case this command will be"
        " slow and use more RAM.",
    )
    parser_score.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for chunk preprocessing (default: 4). "
        "Set to 1–4 on Linux/CUDA for additional throughput.",
    )
    parser_score.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress messages"
    )
    parser_score.set_defaults(func=cmd_score)

    # =============================================================================

    # call peak (alias: consummate_vs)
    parser_peaks = subparsers.add_parser(
        "peaks",
        aliases=["consummate_vs"],
        help="Call peaks from the predicted probability scores of a TROGDOR model",
    )
    parser_peaks.add_argument(
        "-i", "--input", required=True, type=str, help="bigWig file of TROGDOR scores"
    )
    parser_peaks.add_argument(
        "-o", "--output", required=True, type=str, help="Output bed file of peak calls"
    )
    parser_peaks.add_argument(
        "-s",
        "--min_score",
        type=float,
        default=0.95,
        help="Minimum probability to report a bin as a peak (default: 0.95)",
    )
    parser_peaks.add_argument(
        "--mode",
        choices=["simple", "refined", "profile"],
        default="simple",
        help="Peak-calling mode. 'simple' preserves historical threshold-and-merge behaviour; 'refined' adds max_gap/min_width and summit columns; 'profile' adds local-maxima/valley splitting.",
    )
    parser_peaks.add_argument(
        "--max_gap",
        type=int,
        default=0,
        help="Refined/profile mode only: merge candidate blocks separated by at most this many bp.",
    )
    parser_peaks.add_argument(
        "--min_width",
        type=int,
        default=0,
        help="Refined/profile mode only: discard called peaks narrower than this many bp.",
    )
    parser_peaks.add_argument(
        "--max_width",
        type=int,
        default=None,
        help="Profile mode only: discard called peaks wider than this many bp.",
    )
    parser_peaks.add_argument(
        "--seed_score",
        type=float,
        default=None,
        help="Profile mode only: permissive score threshold used to seed candidate blocks.",
    )
    parser_peaks.add_argument(
        "--smooth_bins",
        type=int,
        default=1,
        help="Profile mode only: number of score bins for local smoothing (default: 1).",
    )
    parser_peaks.add_argument(
        "--valley_fraction",
        type=float,
        default=0.5,
        help="Profile mode only: split adjacent summits when the valley is below this fraction of the weaker summit.",
    )
    parser_peaks.add_argument(
        "--boundary_fraction",
        type=float,
        default=0.0,
        help="Profile mode only: trim boundaries to bins above this fraction of the local summit.",
    )
    parser_peaks.add_argument(
        "--min_support_signal",
        type=float,
        default=0.0,
        help="Refined/profile mode only: require this minimum raw plus/minus signal within each peak (default: 0, disabled).",
    )
    parser_peaks.add_argument(
        "--support_plus_bigwig",
        default=None,
        help="Plus-strand raw signal bigWig used with --min_support_signal.",
    )
    parser_peaks.add_argument(
        "--support_minus_bigwig",
        default=None,
        help="Minus-strand raw signal bigWig used with --min_support_signal.",
    )
    parser_peaks.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress messages"
    )
    parser_peaks.set_defaults(func=cmd_peaks)

    # =============================================================================

    # fdr (alias: fire_dragon)
    parser_fdr = subparsers.add_parser(
        "fdr",
        aliases=["fire_dragon"],
        help="Estimate empirical FDR from a probability bigWig and a candidate peak BED",
    )
    parser_fdr.add_argument(
        "-b", "--bigwig", required=True, help="Probability bigWig file"
    )
    parser_fdr.add_argument(
        "-t", "--peaks", required=True, help="Candidate peak BED file (optionally .gz)"
    )
    parser_fdr.add_argument(
        "--stat",
        choices=["max", "mean"],
        default="max",
        help="Summary statistic per peak (default: max)",
    )
    parser_fdr.add_argument(
        "--n_shuffle",
        type=int,
        default=1,
        help="Number of independent genome shuffles to average over (default: 1)",
    )
    parser_fdr.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0)",
    )
    parser_fdr.add_argument(
        "--fdr_target",
        type=float,
        default=None,
        help="FDR target for reporting the score threshold; omit to skip (default: 0.05)",
    )
    parser_fdr.add_argument(
        "--n_thresholds",
        type=int,
        default=200,
        help="Number of evenly-spaced thresholds to evaluate (default: 200)",
    )
    parser_fdr.add_argument(
        "--output",
        default=None,
        help="Write TSV table of threshold/FDR/N_real/N_null to this path",
    )
    parser_fdr.add_argument(
        "--figure",
        default=None,
        help="Save FDR-vs-threshold plot to this path (e.g. fdr_curve.png)",
    )
    parser_fdr.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all in bigWig)",
    )
    parser_fdr.add_argument(
        "--mark_score",
        type=float,
        default=None,
        help="Annotate the FDR and recall at this score threshold on the figure",
    )
    parser_fdr.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-chromosome progress"
    )
    parser_fdr.set_defaults(func=cmd_fdr)

    # =============================================================================

    _ALIASES = {"burninate", "thatch", "consummate_vs", "fire_dragon"}
    subparsers._choices_actions = [
        a for a in subparsers._choices_actions if a.dest not in _ALIASES
    ]
    subparsers.metavar = "{pipeline,score,peaks,fdr}"

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
