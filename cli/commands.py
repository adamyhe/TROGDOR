# commands.py
# Author: Adam He <adamyhe@gmail.com>

import argparse
import io
import math
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import tqdm
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from chiaroscuro.bigwig import (
    IntervalWriter,
    chrom_sizes as bw_chrom_sizes,
    infer_bin_size,
    open_bigwig,
    read_runs,
    read_values,
    split_runs_arrays,
)
from chiaroscuro.data_transforms import normalization
from chiaroscuro.peaks import call_peaks, call_profile_peaks, resolve_seed_score
from chiaroscuro.predict import _select_chroms_to_score, predict_genome
from chiaroscuro.stats import (
    compute_fdr,
    score_peaks,
    shuffle_peaks,
)
from chiaroscuro.utils import load_model

HF_REPO_ID = "adamyhe/TROGDOR"
HF_MODEL_FILENAME = "TROGDOR.torch"


def _load_trogdor_model(model_path, device, verbose=False):
    if model_path is None:
        cached = try_to_load_from_cache(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        if cached is not None:
            if verbose:
                print(f"Loading pretrained weights from cache: {cached}")
            model_path = cached
        else:
            if verbose:
                print(
                    f"No model specified — downloading pretrained weights from {HF_REPO_ID}..."
                )
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)

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

    return load_model(model_path, device), device


def _shared_chrom_sizes(pl_bigwig, mn_bigwig):
    pl_chrom_sizes = bw_chrom_sizes(open_bigwig(pl_bigwig))
    mn_chroms = set(bw_chrom_sizes(open_bigwig(mn_bigwig)))

    return {c: size for c, size in pl_chrom_sizes.items() if c in mn_chroms}


def _peak_params(args):
    threshold = args.min_score
    mode = getattr(args, "mode", "profile")
    params = {
        "threshold": threshold,
        "mode": mode,
        "max_gap": getattr(args, "max_gap", 0),
        "min_width": getattr(args, "min_width", 0),
        "max_width": getattr(args, "max_width", None),
        "seed_score": getattr(args, "seed_score", None),
        "smooth_bins": getattr(args, "smooth_bins", 3),
        "valley_fraction": getattr(args, "valley_fraction", 0.5),
        "boundary_fraction": getattr(args, "boundary_fraction", 0.95),
        "split_merge_rule": getattr(args, "split_merge_rule", "threshold"),
        "split_merge_cutoff": getattr(args, "split_merge_cutoff", 0.5),
        "max_merge_distance": getattr(args, "max_merge_distance", None),
        "min_support_signal": getattr(args, "min_support_signal", 0.0),
        "support_plus": getattr(args, "support_plus_bigwig", None),
        "support_minus": getattr(args, "support_minus_bigwig", None),
    }

    if mode not in {"simple", "refined", "profile"}:
        raise ValueError(f"Unknown peak-calling mode: {mode}")
    if params["max_gap"] < 0:
        raise ValueError("--max_gap must be >= 0.")
    if params["min_width"] < 0:
        raise ValueError("--min_width must be >= 0.")
    if params["max_width"] is not None and params["max_width"] < 0:
        raise ValueError("--max_width must be >= 0.")
    if params["smooth_bins"] < 1:
        raise ValueError("--smooth_bins must be >= 1.")
    if not 0 <= params["valley_fraction"] <= 1:
        raise ValueError("--valley_fraction must be in [0, 1].")
    if not 0 <= params["boundary_fraction"] <= 1:
        raise ValueError("--boundary_fraction must be in [0, 1].")
    if params["split_merge_rule"] not in {"threshold", "learned"}:
        raise ValueError("--split_merge_rule must be 'threshold' or 'learned'.")
    if not 0 <= params["split_merge_cutoff"] <= 1:
        raise ValueError("--split_merge_cutoff must be in [0, 1].")
    if params["max_merge_distance"] is not None and params["max_merge_distance"] <= 0:
        raise ValueError("--max_merge_distance must be > 0.")
    if params["seed_score"] is not None and params["seed_score"] > threshold:
        raise ValueError("--seed_score must be <= --min_score.")
    if mode == "profile":
        params["seed_score"] = resolve_seed_score(threshold, params["seed_score"])
    if params["min_support_signal"] > 0 and mode not in {"refined", "profile"}:
        raise ValueError(
            "--min_support_signal is only supported with --mode refined/profile."
        )
    if params["min_support_signal"] > 0 and (
        params["support_plus"] is None or params["support_minus"] is None
    ):
        raise ValueError(
            "--min_support_signal requires both --support_plus_bigwig and "
            "--support_minus_bigwig."
        )

    return params


def _candidate_threshold(params):
    if params["mode"] == "profile" and params["seed_score"] is not None:
        return min(params["threshold"], params["seed_score"])
    return params["threshold"]


def _open_support_handles(params):
    if params["min_support_signal"] <= 0:
        return None
    return (
        open_bigwig(params["support_plus"]),
        open_bigwig(params["support_minus"]),
    )


def _has_support(support_handles, params, chrom, start, end):
    if support_handles is None:
        return True
    pl_bw, mn_bw = support_handles
    pl = np.nan_to_num(read_values(pl_bw, chrom, start, end))
    mn = np.abs(np.nan_to_num(read_values(mn_bw, chrom, start, end)))
    return max(pl.max(initial=0.0), mn.max(initial=0.0)) >= params["min_support_signal"]


def _call_chrom_peaks(chrom, intervals, params, support_handles=None):
    peaks = []
    threshold = params["threshold"]
    mode = params["mode"]

    if mode == "simple":
        for peak in call_peaks(intervals, min_score=threshold):
            peaks.append({"chrom": chrom, **peak})
    elif mode == "refined":
        for peak in call_peaks(
            intervals,
            min_score=threshold,
            max_gap=params["max_gap"],
            min_width=params["min_width"],
        ):
            if not _has_support(
                support_handles, params, chrom, peak["start"], peak["end"]
            ):
                continue
            peaks.append({"chrom": chrom, **peak})
    elif mode == "profile":
        for peak in call_profile_peaks(
            intervals,
            min_score=threshold,
            seed_score=params["seed_score"],
            max_gap=params["max_gap"],
            min_width=params["min_width"],
            max_width=params["max_width"],
            smooth_bins=params["smooth_bins"],
            valley_fraction=params["valley_fraction"],
            boundary_fraction=params["boundary_fraction"],
            split_merge_rule=params["split_merge_rule"],
            split_merge_cutoff=params["split_merge_cutoff"],
            max_merge_distance=params["max_merge_distance"],
        ):
            if not _has_support(
                support_handles, params, chrom, peak["start"], peak["end"]
            ):
                continue
            peaks.append({"chrom": chrom, **peak})

    return peaks


def _write_peak_record(out_bed, peak, params):
    if params["mode"] == "simple":
        out_bed.write(
            f"{peak['chrom']}\t{peak['start']}\t{peak['end']}\t{peak['score']:.6g}\n"
        )
    else:
        out_bed.write(
            f"{peak['chrom']}\t{peak['start']}\t{peak['end']}\t"
            f"{peak['score']:.6g}\t{peak['summit_start']}\t"
            f"{peak['summit_end']}\t{peak['summit_score']:.6g}\n"
        )


def _write_chrom_peaks(out_bed, chrom, intervals, params, support_handles=None):
    peaks = _call_chrom_peaks(chrom, intervals, params, support_handles)
    for peak in peaks:
        _write_peak_record(out_bed, peak, params)
    return len(peaks)


def _write_peak_file(output, write_func):
    out_path = output
    if out_path.endswith(".gz") and shutil.which("bgzip") is None:
        out_path = out_path[:-3]  # strip .gz
        warnings.warn(f"bgzip not found; writing uncompressed BED to {out_path}")

    if out_path.endswith(".gz"):
        with open(out_path, "wb") as raw_out:
            proc = subprocess.Popen(["bgzip"], stdin=subprocess.PIPE, stdout=raw_out)
            with io.TextIOWrapper(proc.stdin, encoding="utf-8") as out_bed:
                n_peaks = write_func(out_bed)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"bgzip exited with code {proc.returncode}")
    else:
        with open(out_path, "w") as out_bed:
            n_peaks = write_func(out_bed)

    return n_peaks, out_path


def _stream_scores(args, model, device, valid_chroms, header):
    """Score ``valid_chroms`` and write each to ``args.output`` as it's scored."""
    m = 0
    with IntervalWriter(args.output, header) as writer:
        for chrom, _, probs in predict_genome(
            model,
            args.pl_bigwig,
            args.mn_bigwig,
            chroms=valid_chroms,
            output_stride=args.output_stride,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            transform=normalization,
            device=device,
            verbose=args.verbose,
            num_workers=getattr(args, "num_workers", 0),
        ):
            bin_indices = np.flatnonzero(probs >= args.min_score)
            writer.write(
                chrom,
                bin_indices * args.output_stride,
                (bin_indices + 1) * args.output_stride,
                probs[bin_indices],
            )
            m += len(bin_indices)

    if args.verbose:
        print(f"Wrote {m} candidate bins (score >= {args.min_score}) to {args.output}.")


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
    model, device = _load_trogdor_model(args.model, args.device, args.verbose)
    chrom_sizes = _shared_chrom_sizes(args.pl_bigwig, args.mn_bigwig)

    chroms_to_score = (
        args.chroms if args.chroms is not None else list(chrom_sizes.keys())
    )

    # The header lists the chromosomes that will actually be scored, in
    # scoring order, so each one can be written as soon as it's scored rather
    # than holding a whole genome of (dense, with -s 0) bins until the end.
    pl_sizes = bw_chrom_sizes(open_bigwig(args.pl_bigwig))
    valid_chroms = _select_chroms_to_score(chroms_to_score, pl_sizes, args.chunk_size)
    header = {chrom: pl_sizes[chrom] for chrom in valid_chroms}

    try:
        _stream_scores(args, model, device, valid_chroms, header)
    except BaseException:
        # Don't leave a valid-looking bigWig with chromosomes missing.
        if os.path.exists(args.output):
            os.remove(args.output)
        raise


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
    params = _peak_params(args)
    candidate_threshold = _candidate_threshold(params)

    in_bw = open_bigwig(args.input)
    chrom_sizes = bw_chrom_sizes(in_bw)

    # Every caller ignores bins below the candidate threshold, so drop them
    # per chromosome as it is read: a dense (`score -s 0`) track then never
    # sits in memory whole. The bin size is still inferred from the unfiltered
    # runs, so filtering can't change it.
    chrom_runs = {}
    bin_size = None
    for chrom, chrom_len in tqdm.tqdm(
        chrom_sizes.items(), desc="Loading scores", disable=not args.verbose
    ):
        starts, ends, values = read_runs(in_bw, chrom, chrom_len)
        chrom_bin = infer_bin_size([(starts, ends, values)])
        if chrom_bin:
            bin_size = math.gcd(bin_size or 0, chrom_bin)
        keep = values.astype(np.float64) >= candidate_threshold
        chrom_runs[chrom] = (starts[keep], ends[keep], values[keep])

    if args.verbose:
        # Equal-valued neighbouring bins read back as one run; count bins.
        n_pass = (
            sum(int(((e - s) // bin_size).sum()) for s, e, _ in chrom_runs.values())
            if bin_size
            else 0
        )
        if n_pass == 0:
            print("No bins pass threshold; writing empty BED file")
        else:
            print(f"Candidate threshold: {candidate_threshold:.6f} ({n_pass} bins pass)")

    support_handles = _open_support_handles(params)

    def _write_peaks(out_bed):
        n = 0
        for chrom in sorted(chrom_sizes):
            # Split runs back into the score bigWig's fixed-width bins, whose
            # width every run boundary is a multiple of, one chromosome at a time.
            runs = chrom_runs.pop(chrom)
            intervals = split_runs_arrays(*runs, bin_size) if bin_size else []
            n += _write_chrom_peaks(out_bed, chrom, intervals, params, support_handles)
        return n

    n_peaks, out_path = _write_peak_file(args.output, _write_peaks)

    if args.verbose:
        print(f"{n_peaks} peaks written to {out_path}")


def cmd_pipeline(args):
    """Run the full pipeline: score the genome, then call peaks.

    Convenience wrapper that scores the genome, then calls peaks. By default,
    peaks are called directly from streamed per-chromosome predictions without
    materializing an intermediate score bigWig. If ``save_bigwig`` is provided,
    the score bigWig is written and then re-used for peak calling.

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

    def _run_with_bigwig(bw_prefix):
        peak_args = argparse.Namespace(
            **{
                **vars(args),
                "support_plus_bigwig": args.pl_bigwig,
                "support_minus_bigwig": args.mn_bigwig,
            }
        )
        score_min = _candidate_threshold(_peak_params(peak_args))
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
                min_score=score_min,
                verbose=args.verbose,
                num_workers=args.num_workers,
            )
        )
        cmd_peaks(
            argparse.Namespace(
                input=f"{bw_prefix}.prob.bw",
                output=args.output,
                min_score=args.min_score,
                mode=getattr(args, "mode", "profile"),
                max_gap=getattr(args, "max_gap", 0),
                min_width=getattr(args, "min_width", 0),
                max_width=getattr(args, "max_width", None),
                seed_score=getattr(args, "seed_score", None),
                smooth_bins=getattr(args, "smooth_bins", 3),
                valley_fraction=getattr(args, "valley_fraction", 0.5),
                boundary_fraction=getattr(args, "boundary_fraction", 0.95),
                split_merge_rule=getattr(args, "split_merge_rule", "threshold"),
                split_merge_cutoff=getattr(args, "split_merge_cutoff", 0.5),
                max_merge_distance=getattr(args, "max_merge_distance", None),
                min_support_signal=getattr(args, "min_support_signal", 0.0),
                support_plus_bigwig=args.pl_bigwig,
                support_minus_bigwig=args.mn_bigwig,
                verbose=args.verbose,
            )
        )

    def _run_direct():
        peak_args = argparse.Namespace(
            **{
                **vars(args),
                "support_plus_bigwig": args.pl_bigwig,
                "support_minus_bigwig": args.mn_bigwig,
            }
        )
        params = _peak_params(peak_args)
        candidate_threshold = _candidate_threshold(params)
        model, device = _load_trogdor_model(args.model, args.device, args.verbose)
        chrom_sizes = _shared_chrom_sizes(args.pl_bigwig, args.mn_bigwig)
        chroms_to_score = (
            args.chroms if args.chroms is not None else list(chrom_sizes.keys())
        )
        support_handles = _open_support_handles(params)

        def _write_peaks(out_bed):
            n = 0
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
                bin_indices = np.where(probs >= candidate_threshold)[0]
                intervals = (
                    bin_indices * args.output_stride,
                    (bin_indices + 1) * args.output_stride,
                    probs[bin_indices],
                )
                n += _write_chrom_peaks(
                    out_bed, chrom, intervals, params, support_handles
                )
            return n

        n_peaks, out_path = _write_peak_file(args.output, _write_peaks)

        if args.verbose:
            print(f"{n_peaks} peaks written to {out_path}")

    if args.save_bigwig is not None:
        bw_prefix = args.save_bigwig.removesuffix(".prob.bw")
        _run_with_bigwig(bw_prefix)
    else:
        _run_direct()


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
            Number of empirical FDR thresholds to evaluate.
        ``threshold_grid`` (str)
            Threshold selection strategy.
        ``output`` (str or None)
            Path to write TSV table of threshold/FDR/N_real/N_null.
        ``figure`` (str or None)
            Path to save FDR-vs-threshold plot.
        ``chroms`` (list of str or None)
            Chromosome whitelist; ``None`` uses all chromosomes in the bigWig.
        ``verbose`` (bool)
            Whether to print per-chromosome progress.
    """
    bw = open_bigwig(args.bigwig)
    chrom_sizes = bw_chrom_sizes(bw)
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
        s = score_peaks(
            bw, null_df, chrom_sizes, args.stat, chroms, verbose=args.verbose
        )
        null_score_lists.append(s[~np.isnan(s)])

    null_scores = np.concatenate(null_score_lists)
    thresholds, n_real, n_null, fdr = compute_fdr(
        real_scores,
        null_scores,
        args.n_shuffle,
        args.n_thresholds,
        getattr(args, "threshold_grid", "quantile"),
    )

    # ---- Find threshold at FDR target ----
    threshold_at_target = float("nan")
    n_peaks_at_target = 0
    first_pass = None
    if args.fdr_target is not None:
        passing = np.where(fdr <= args.fdr_target)[0]
        if len(passing) > 0:
            first_pass = passing[0]
            threshold_at_target = float(thresholds[first_pass])
            n_peaks_at_target = int(n_real[first_pass])

    # ---- Print summary ----
    print(f"Real peaks:       {len(real_scores):,}")
    print(f"Null peaks:       {len(null_scores):,} ({args.n_shuffle} shuffle(s))")
    print(f"Score stat:       {args.stat}")
    if args.fdr_target is not None:
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

        recall = n_real / len(real_scores)

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
        if args.fdr_target is not None and not np.isnan(threshold_at_target):
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

        # Right — FDR curve overlaid with recall
        ax = axes[1]
        ax.plot(thresholds, fdr, color="black", linewidth=1.5, label="FDR")
        if args.fdr_target is not None:
            ax.axhline(
                args.fdr_target,
                color="firebrick",
                linestyle="--",
                linewidth=0.8,
                label=f"FDR={args.fdr_target}",
            )
        if args.fdr_target is not None and not np.isnan(threshold_at_target):
            ax.axvline(
                threshold_at_target,
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label=f"t={threshold_at_target:.3f}",
            )
        ax.set_xlabel(f"Score threshold ({args.stat})")
        ax.set_ylabel("Empirical FDR")
        ax.set_title("FDR and recall vs threshold")
        ax.set_ylim(0, 1.05)

        ax2 = ax.twinx()
        ax2.plot(
            thresholds,
            recall,
            color="steelblue",
            linewidth=1.5,
            linestyle="-",
            label="Recall",
        )
        ax2.set_ylabel("Recall (fraction of real peaks)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(0, 1.05)

        mark = getattr(args, "mark_score", None)
        if mark is not None and thresholds[0] <= mark <= thresholds[-1]:
            idx = int(np.searchsorted(thresholds, mark, side="left"))
            idx = min(idx, len(thresholds) - 1)
            fdr_at_mark = float(fdr[idx])
            recall_at_mark = float(recall[idx])
            ax.axvline(mark, color="darkorange", linestyle=":", linewidth=1.2)
            ax.annotate(
                f"FDR={fdr_at_mark:.3f}",
                xy=(mark, fdr_at_mark),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7,
                color="black",
            )
            ax2.annotate(
                f"recall={recall_at_mark:.3f}",
                xy=(mark, recall_at_mark),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=7,
                color="steelblue",
            )

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        plt.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"Saved figure to {args.figure}")
