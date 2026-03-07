# commands.py
# Author: Adam He <adamyhe@gmail.com>

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


def _merge_intervals(intervals):
    """Merge abutting intervals, keeping the max value across merged spans.

    Expects ``intervals`` to be a sorted list of ``(start, end, value)`` tuples
    where adjacent intervals share an endpoint (``prev_end == next_start``).
    Overlapping intervals are not expected and not handled.

    Parameters
    ----------
    intervals : list of (int, int, float)
        Sorted (start, end, value) tuples.

    Returns
    -------
    list of [int, int, float]
        Merged intervals as mutable lists.
    """
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
        transform=normalization,
        device=args.device,
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
