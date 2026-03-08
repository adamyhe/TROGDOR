# commands.py
# Author: Adam He <adamyhe@gmail.com>

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
from chiaroscuro.utils import bh_correct, merge_intervals

HF_REPO_ID = "adamyhe/TROGDOR"
HF_MODEL_FILENAME = "TROGDOR.torch"



def cmd_score(args):
    """Run the ``score`` subcommand: genome-wide TIR scoring to a bigWig.

    Loads a TROGDOR model (downloading pretrained weights from HuggingFace Hub
    if ``args.model`` is ``None``), slides it across all requested chromosomes,
    and writes an output bigWig whose values are BH-adjusted scores (1 − q).

    Only bins with a raw model probability ≥ ``args.min_score`` are retained as
    candidates before BH correction, which is applied globally across all
    chromosomes. The written value for each candidate bin is ``1 − q``, so bins
    that pass a downstream FDR threshold ``α`` have written values ≥ ``1 − α``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes:

        ``model`` (str or None)
            Path to a ``.torch`` state-dict, or ``None`` to use pretrained weights.
        ``pl_bigwig`` (str)
            Path to the plus-strand coverage bigWig.
        ``mn_bigwig`` (str)
            Path to the minus-strand coverage bigWig.
        ``output`` (str)
            Path for the output bigWig.
        ``device`` (str)
            PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
        ``chunk_size`` (int)
            Input window length fed to the model per chunk.
        ``overlap`` (int)
            Edge bins trimmed from each chunk to avoid boundary artefacts.
        ``output_stride`` (int)
            Model output resolution in bp.
        ``chroms`` (list of str or None)
            Chromosomes to score; ``None`` scores all shared chromosomes.
        ``batch_size`` (int)
            Number of chunks per forward pass.
        ``min_score`` (float)
            Pre-filter threshold on raw model probability.
        ``verbose`` (bool)
            Whether to print progress messages.
    """
    model_path = args.model
    if model_path is None:
        if args.verbose:
            print(
                f"No model specified — downloading default weights from {HF_REPO_ID}..."
            )
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("Warning: CUDA not available, falling back to MPS.")
            device = "mps"
        else:
            print("Warning: CUDA not available, falling back to CPU.")
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU.")
        device = "cpu"

    model = TROGDOR()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cpu"), strict=False
    )
    model = model.to(device).eval()

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

    # Pass 1: score the genome and collect all bins with score >= min_score
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
        batch_size=args.batch_size,
        transform=normalization,
        device=device,
        verbose=args.verbose,
    ):
        chrom_dict[chrom] = chrom_len
        bin_indices = np.where(probs >= args.min_score)[0]
        for i in bin_indices:
            all_intervals.append(
                (
                    chrom,
                    int(i * args.output_stride),
                    int((i + 1) * args.output_stride),
                    float(probs[i]),
                )
            )

    # Pass 2: BH correction over pre-filtered candidates
    all_probs_arr = np.array([v for _, _, _, v in all_intervals])
    m = len(all_probs_arr)
    q_values = bh_correct(all_probs_arr)

    if args.verbose:
        print(
            f"Writing {m} candidate bins (score >= {args.min_score}) with BH-corrected q-values"
        )

    # Build prob → (1 − q) lookup; on ties keep the highest value
    prob_to_val = {}
    for prob, q in zip(all_probs_arr.tolist(), q_values.tolist()):
        val = 1.0 - q
        if prob not in prob_to_val or val > prob_to_val[prob]:
            prob_to_val[prob] = val

    def _fdr_intervals():
        for chrom, start, end, prob in all_intervals:
            val = prob_to_val.get(prob)
            if val is not None:
                yield chrom, start, end, val

    out_bw = pybigtools.open(args.output, "w")
    out_bw.write(chrom_dict, _fdr_intervals())


def cmd_peaks(args):
    """Run the ``peaks`` subcommand: convert a scored bigWig to BED peak calls.

    Reads the bigWig produced by ``cmd_score`` (whose values are ``1 − q``),
    applies a score threshold of ``1 − fdr_threshold`` to select passing bins,
    merges abutting passing bins into peak regions, and writes a BED file with
    columns ``chrom``, ``start``, ``end``, ``q_value``.

    Output is written as plain text unless ``args.output`` ends with ``".gz"``,
    in which case it is piped through ``bgzip``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes:

        ``input`` (str)
            Path to the scored bigWig (output of ``cmd_score``).
        ``output`` (str)
            Path for the output BED (or ``.bed.gz``) file.
        ``fdr_threshold`` (float)
            BH FDR threshold; bins with score ≥ ``1 − fdr_threshold`` are reported.
        ``verbose`` (bool)
            Whether to print progress messages.
    """
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
            passing = [
                (s, e, v) for s, e, v in chrom_intervals[chrom] if v >= threshold
            ]
            for start, end, max_v in merge_intervals(passing):
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


def cmd_pipeline(args):
    """Run the full pipeline: score the genome, then call peaks.

    Convenience wrapper that runs ``cmd_score`` followed by ``cmd_peaks``.
    Output paths are derived from ``args.name``:

    - ``{name}.prob.bw`` — intermediate scored bigWig (kept)
    - ``{name}.peaks.bed.gz`` — final bgzipped BED of peak calls

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes are the union of those
        required by ``cmd_score`` and ``cmd_peaks``:

        ``model`` (str or None), ``pl_bigwig`` (str), ``mn_bigwig`` (str),
        ``name`` (str), ``device`` (str), ``chunk_size`` (int),
        ``overlap`` (int), ``output_stride`` (int), ``batch_size`` (int),
        ``chroms`` (list or None), ``min_score`` (float),
        ``fdr_threshold`` (float), ``verbose`` (bool).
    """
    cmd_score(
        argparse.Namespace(
            model=args.model,
            pl_bigwig=args.pl_bigwig,
            mn_bigwig=args.mn_bigwig,
            output=f"{args.name}.prob.bw",
            device=args.device,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            output_stride=args.output_stride,
            batch_size=args.batch_size,
            chroms=args.chroms,
            min_score=args.min_score,
            verbose=args.verbose,
        )
    )
    cmd_peaks(
        argparse.Namespace(
            input=f"{args.name}.prob.bw",
            output=f"{args.name}.peaks.bed.gz",
            fdr_threshold=args.fdr_threshold,
            verbose=args.verbose,
        )
    )
