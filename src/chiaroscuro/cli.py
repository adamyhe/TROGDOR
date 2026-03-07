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
import io
import subprocess

import numpy as np
import pybigtools
import torch
import tqdm
from huggingface_hub import hf_hub_download

from chiaroscuro.data_transforms import normalization
from chiaroscuro.predict import predict_genome
from chiaroscuro.trogdor import TROGDOR

HF_REPO_ID = "adamyhe/TROGDOR"
HF_MODEL_FILENAME = "TROGDOR.torch"

_help = """
The following commands are available:
    pipeline / burninate  Run the full TROGDOR pipeline from raw data to peak calls
    score                 Score positions using a pre-trained TROGDOR model
    call                  Call peaks from scored positions
"""



def _merge_intervals(intervals):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e, v in intervals[1:]:
        if s == merged[-1][1]:
            merged[-1][1] = e
            merged[-1][2] = max(merged[-1][2], v)
        else:
            merged.append([s, e, v])
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
        "pipeline",
        aliases=["burninate"],
        help="Run the full TROGDOR pipeline from raw data to peak calls",
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
        "-s",
        "--min_score",
        type=float,
        default=0.9,
        help="Minimum raw score to include as a candidate bin before BH correction (default: 0.9)",
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

    # =============================================================================

    # Load args
    args = parser.parse_args()

    # Run provided command
    if args.cmd == "score":
        model_path = args.model
        if model_path is None:
            if args.verbose:
                print(
                    f"No model specified — downloading default weights from {HF_REPO_ID}..."
                )
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        model = TROGDOR()
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu"), strict=False
        )
        model = model.to(args.device).eval()

        pl_bw = pybigtools.open(args.pl_bigwig)
        pl_chrom_sizes = dict(pl_bw.chroms())
        pl_bw.close()

        mn_bw = pybigtools.open(args.mn_bigwig)
        mn_chroms = set(mn_bw.chroms().keys())
        mn_bw.close()

        chrom_sizes = {c: size for c, size in pl_chrom_sizes.items() if c in mn_chroms}

        chroms_to_score = (
            args.chroms if args.chroms is not None else list(chrom_sizes.keys())
        )

        # Pass 1: score the genome and collect all nonzero bins
        all_intervals = []  # list of (chrom, start, end, prob)
        chrom_dict = {}
        for chrom, chrom_len, probs in predict_genome(
            model,
            args.pl_bigwig,
            args.mn_bigwig,
            chroms=chroms_to_score,
            output_stride=args.output_stride,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            transform=normalization,
            device=args.device,
            verbose=args.verbose,
        ):
            chrom_dict[chrom] = chrom_len
            bin_indices = np.where(probs >= args.min_score)[0]
            for i in bin_indices:
                all_intervals.append((
                    chrom,
                    int(i * args.output_stride),
                    int((i + 1) * args.output_stride),
                    float(probs[i]),
                ))

        # Pass 2: BH correction over all collected bins
        all_probs_arr = np.array([v for _, _, _, v in all_intervals])
        m = len(all_probs_arr)
        if m > 0:
            p = 1.0 - all_probs_arr
            sort_idx = np.argsort(p)
            sorted_p = p[sort_idx]
            raw_q = sorted_p * m / np.arange(1, m + 1)
            q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
            q_values = np.empty(m)
            q_values[sort_idx] = q_sorted
        else:
            q_values = np.array([])

        if args.verbose:
            n_pass = int(np.sum(q_values < 1.0))
            print(f"Writing {n_pass} bins with 1−q > 0 out of {m} total nonzero bins")

        # Build prob → (1 − q) lookup; on ties keep the highest value
        prob_to_val = {}
        for prob, q in zip(all_probs_arr.tolist(), q_values.tolist()):
            val = 1.0 - q
            if val > 0 and (prob not in prob_to_val or val > prob_to_val[prob]):
                prob_to_val[prob] = val

        def _fdr_intervals():
            for chrom, start, end, prob in all_intervals:
                val = prob_to_val.get(prob)
                if val is not None:
                    yield chrom, start, end, val

        out_bw = pybigtools.open(args.output, "w")
        out_bw.write(chrom_dict, _fdr_intervals())

    elif args.cmd == "peaks":
        in_bw = pybigtools.open(args.input)
        chrom_sizes = dict(in_bw.chroms())

        # Pass 1: collect all intervals per chrom and aggregate probs for BH
        chrom_intervals = {}
        all_probs = []
        for chrom, chrom_len in tqdm.tqdm(
            chrom_sizes.items(),
            desc="Loading scores",
            disable=not args.verbose,
        ):
            ivals = [
                (s, e, v)
                for s, e, v in in_bw.records(chrom, 0, chrom_len)
                if not np.isnan(v)
            ]
            chrom_intervals[chrom] = ivals
            all_probs.extend(v for _, _, v in ivals)
        in_bw.close()

        threshold = 1.0 - args.fdr_threshold

        if args.verbose:
            n_pass = sum(1 for p in all_probs if p >= threshold)
            if n_pass == 0:
                print("No bins pass FDR threshold; writing empty BED file")
            else:
                print(f"Score threshold: {threshold:.6f} ({n_pass} bins pass)")

        def _write_peaks(out_bed):
            for chrom in sorted(chrom_sizes):
                passing = [(s, e, v) for s, e, v in chrom_intervals[chrom] if v >= threshold]
                for start, end, max_v in _merge_intervals(passing):
                    q = 1.0 - max_v
                    out_bed.write(f"{chrom}\t{start}\t{end}\t{q:.6g}\n")

        if args.output.endswith(".gz"):
            with open(args.output, "wb") as raw_out:
                proc = subprocess.Popen(["bgzip"], stdin=subprocess.PIPE, stdout=raw_out)
                with io.TextIOWrapper(proc.stdin, encoding="utf-8") as out_bed:
                    _write_peaks(out_bed)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"bgzip exited with code {proc.returncode}")
        else:
            with open(args.output, "w") as out_bed:
                _write_peaks(out_bed)

    elif args.cmd in ("pipeline", "burninate"):
        pass

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
