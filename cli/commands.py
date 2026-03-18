# commands.py
# Author: Adam He <adamyhe@gmail.com>

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pybigtools
import torch
import tqdm
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from chiaroscuro.data_transforms import normalization
from chiaroscuro.predict import predict_genome
from chiaroscuro.stats import compute_fdr, score_peaks, shuffle_peaks
from chiaroscuro.utils import load_model, merge_intervals

HF_REPO_ID = "adamyhe/TROGDOR"
HF_MODEL_FILENAME = "TROGDOR.torch"


def cmd_score(args):
    """Run the ``score`` subcommand: genome-wide TIR scoring to a bigWig.

    Loads a TROGDOR model (loading pretrained weights from the local HuggingFace
    cache, or downloading them from HuggingFace Hub, if ``args.model`` is
    ``None``), slides it across all requested chromosomes,
    and writes one output bigWig to ``args.output``:

    - ``args.output`` — raw model probabilities for candidate bins
      (raw prob ≥ ``args.min_score``)

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
            Full output bigWig path (e.g. ``sample.prob.bw``).
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
            Storage threshold; bins with raw prob below this are omitted from
            the output bigWig.
        ``verbose`` (bool)
            Whether to print progress messages.
    """
    model_path = args.model
    if model_path is None:
        cached = try_to_load_from_cache(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        if cached is not None:
            if args.verbose:
                print(f"Loading pretrained weights from cache: {cached}")
            model_path = cached
        else:
            if args.verbose:
                print(
                    f"No model specified — downloading pretrained weights from {HF_REPO_ID}..."
                )
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            warnings.warn("CUDA not available, falling back to MPS.")
            device = "mps"
        else:
            warnings.warn("CUDA not available, falling back to CPU.")
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS not available, falling back to CPU.")
        device = "cpu"

    model = load_model(model_path, device)

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
        num_workers=getattr(args, "num_workers", 0),
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

    m = len(all_intervals)

    if args.verbose:
        print(
            f"Writing {m} candidate bins (score >= {args.min_score}) to {args.output}."
        )

    def _raw_intervals():
        for chrom, start, end, prob in all_intervals:
            yield chrom, start, end, prob

    out_bw = pybigtools.open(args.output, "w")
    out_bw.write(chrom_dict, _raw_intervals())


def cmd_peaks(args):
    """Run the ``peaks`` subcommand: convert a scored bigWig to BED peak calls.

    Reads ``prob.bw`` produced by ``cmd_score``, applies a direct probability
    threshold (``args.min_score``) to select passing bins, merges abutting
    passing bins into peak regions, and writes a BED file with columns
    ``chrom``, ``start``, ``end``, ``score``.

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
        ``min_score`` (float)
            Minimum probability to report a bin as a peak.
        ``verbose`` (bool)
            Whether to print progress messages.
    """
    in_bw = pybigtools.open(args.input)
    chrom_sizes = dict(in_bw.chroms())

    chrom_intervals = {}
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
    in_bw.close()

    threshold = args.min_score

    if args.verbose:
        n_pass = sum(
            1
            for ivals in chrom_intervals.values()
            for _, _, v in ivals
            if v >= threshold
        )
        if n_pass == 0:
            print("No bins pass threshold; writing empty BED file")
        else:
            print(f"Score threshold: {threshold:.6f} ({n_pass} bins pass)")

    def _write_peaks(out_bed):
        n = 0
        for chrom in sorted(chrom_sizes):
            passing = [
                (s, e, v) for s, e, v in chrom_intervals[chrom] if v >= threshold
            ]
            for start, end, max_v in merge_intervals(passing):
                out_bed.write(f"{chrom}\t{start}\t{end}\t{max_v:.6g}\n")
                n += 1
        return n

    out_path = args.output
    if out_path.endswith(".gz") and shutil.which("bgzip") is None:
        out_path = out_path[:-3]  # strip .gz
        warnings.warn(f"bgzip not found; writing uncompressed BED to {out_path}")

    if out_path.endswith(".gz"):
        with open(out_path, "wb") as raw_out:
            proc = subprocess.Popen(["bgzip"], stdin=subprocess.PIPE, stdout=raw_out)
            with io.TextIOWrapper(proc.stdin, encoding="utf-8") as out_bed:
                n_peaks = _write_peaks(out_bed)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"bgzip exited with code {proc.returncode}")
    else:
        with open(out_path, "w") as out_bed:
            n_peaks = _write_peaks(out_bed)

    if args.verbose:
        print(f"{n_peaks} peaks written to {out_path}")


def cmd_pipeline(args):
    """Run the full pipeline: score the genome, then call peaks.

    Convenience wrapper that runs ``cmd_score`` followed by ``cmd_peaks``.
    The intermediate probability bigWig is written to a temporary file and
    deleted after peaks are called.  Only the final BED is kept:

    - ``output`` — bgzipped BED of peak calls (full path specified by caller)

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes are the union of those
        required by ``cmd_score`` and ``cmd_peaks``:

        ``model`` (str or None), ``pl_bigwig`` (str), ``mn_bigwig`` (str),
        ``output`` (str), ``save_bigwig`` (str or None), ``device`` (str),
        ``chunk_size`` (int), ``overlap`` (int), ``output_stride`` (int),
        ``batch_size`` (int), ``chroms`` (list or None), ``min_score`` (float),
        ``verbose`` (bool).
    """

    def _run(bw_prefix):
        cmd_score(
            argparse.Namespace(
                model=args.model,
                pl_bigwig=args.pl_bigwig,
                mn_bigwig=args.mn_bigwig,
                output=f"{bw_prefix}.prob.bw",
                device=args.device,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                output_stride=args.output_stride,
                batch_size=args.batch_size,
                chroms=args.chroms,
                min_score=args.min_score,
                verbose=args.verbose,
                num_workers=args.num_workers,
            )
        )
        cmd_peaks(
            argparse.Namespace(
                input=f"{bw_prefix}.prob.bw",
                output=args.output,
                min_score=args.min_score,
                verbose=args.verbose,
            )
        )

    if args.save_bigwig is not None:
        bw_prefix = (
            args.save_bigwig[: -len(".prob.bw")]
            if args.save_bigwig.endswith(".prob.bw")
            else args.save_bigwig
        )
        _run(bw_prefix)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            _run(os.path.join(tmpdir, "tmp"))


def cmd_fdr(args):
    """Run the ``fdr`` subcommand: estimate empirical FDR from a probability bigWig.

    Summarises bigWig scores over candidate peaks and a shuffled null set,
    then estimates FDR(t) = min(1, N_null(t) / N_real(t)) across thresholds.
    Prints the score threshold at the requested FDR target. Optionally writes a
    TSV table and/or an FDR-vs-threshold figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes:

        ``bigwig`` (str)
            Path to the probability bigWig file.
        ``peaks`` (str)
            Path to the candidate peak BED file (optionally .gz).
        ``stat`` (str)
            Summary statistic per peak; ``"max"`` or ``"mean"``.
        ``n_shuffle`` (int)
            Number of independent genome shuffles to average over.
        ``seed`` (int)
            Random seed for shuffling.
        ``fdr_target`` (float)
            FDR target for reporting the score threshold.
        ``n_thresholds`` (int)
            Number of evenly-spaced thresholds to evaluate.
        ``output`` (str or None)
            Path to write TSV table of threshold/FDR/N_real/N_null.
        ``figure`` (str or None)
            Path to save FDR-vs-threshold plot.
        ``chroms`` (list of str or None)
            Chromosome whitelist; ``None`` uses all chromosomes in the bigWig.
        ``verbose`` (bool)
            Whether to print per-chromosome progress.
    """
    bw = pybigtools.open(args.bigwig)
    chrom_sizes = dict(bw.chroms())
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes.keys())

    peaks_df = pd.read_csv(
        args.peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
    )
    peaks_df = peaks_df[peaks_df["chrom"].isin(chroms)].reset_index(drop=True)

    if len(peaks_df) == 0:
        print("No peaks found on the requested chromosomes.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(
            f"Scoring {len(peaks_df):,} real peaks from {args.bigwig}...",
            flush=True,
        )
    real_scores = score_peaks(bw, peaks_df, chrom_sizes, args.stat, chroms)
    real_scores = real_scores[~np.isnan(real_scores)]

    if len(real_scores) == 0:
        print("All real peak scores are NaN; check bigWig coverage.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    null_score_lists = []
    for i in range(args.n_shuffle):
        if args.verbose:
            print(f"  Shuffle {i + 1}/{args.n_shuffle}...", flush=True)
        null_df = shuffle_peaks(peaks_df, chrom_sizes, chroms, rng)
        s = score_peaks(bw, null_df, chrom_sizes, args.stat, chroms, False)
        null_score_lists.append(s[~np.isnan(s)])
    bw.close()

    null_scores = np.concatenate(null_score_lists)
    thresholds, n_real, n_null, fdr = compute_fdr(
        real_scores, null_scores, args.n_shuffle, args.n_thresholds
    )

    # ---- Find threshold at FDR target ----
    passing = np.where(fdr <= args.fdr_target)[0]
    if len(passing) == 0:
        threshold_at_target = float("nan")
        n_peaks_at_target = 0
    else:
        first_pass = passing[0]
        threshold_at_target = float(thresholds[first_pass])
        n_peaks_at_target = int(n_real[first_pass])

    # ---- Print summary ----
    print(f"Real peaks:       {len(real_scores):,}")
    print(f"Null peaks:       {len(null_scores):,} ({args.n_shuffle} shuffle(s))")
    print(f"Score stat:       {args.stat}")
    print(f"FDR target:       {args.fdr_target:.3f}")
    if np.isnan(threshold_at_target):
        print(f"Score threshold:  N/A (FDR {args.fdr_target} never reached)")
    else:
        print(f"Score threshold:  {threshold_at_target:.6f}")
        print(f"Peaks at target:  {n_peaks_at_target:,}")

    # ---- Optional TSV output ----
    if args.output is not None:
        table = pd.DataFrame(
            {
                "threshold": thresholds,
                "n_real": n_real.astype(int),
                "n_null": n_null,
                "fdr": fdr,
            }
        )
        table.to_csv(args.output, sep="\t", index=False, float_format="%.6g")
        print(f"Saved FDR table to {args.output}")

    # ---- Optional figure ----
    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(args.bigwig, fontsize=9)

        # Left — score distributions
        ax = axes[0]
        bins = np.linspace(thresholds[0], thresholds[-1], 60)
        ax.hist(
            real_scores,
            bins=bins,
            density=True,
            alpha=0.6,
            color="steelblue",
            label="real",
        )
        if len(null_scores) > 0:
            ax.hist(
                null_scores,
                bins=bins,
                density=True,
                alpha=0.5,
                color="salmon",
                label="null",
            )
        if not np.isnan(threshold_at_target):
            ax.axvline(
                threshold_at_target,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"t={threshold_at_target:.3f} (FDR={args.fdr_target})",
            )
        ax.set_xlabel(f"Peak score ({args.stat})")
        ax.set_ylabel("Density")
        ax.set_title("Score distributions")
        ax.legend(fontsize=8)

        # Right — FDR curve
        ax = axes[1]
        ax.plot(thresholds, fdr, color="black", linewidth=1.5)
        ax.axhline(
            args.fdr_target,
            color="firebrick",
            linestyle="--",
            linewidth=0.8,
            label=f"FDR={args.fdr_target}",
        )
        if not np.isnan(threshold_at_target):
            ax.axvline(
                threshold_at_target,
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label=f"t={threshold_at_target:.3f}",
            )
        ax.set_xlabel(f"Score threshold ({args.stat})")
        ax.set_ylabel("Empirical FDR")
        ax.set_title("FDR vs threshold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"Saved figure to {args.figure}")
