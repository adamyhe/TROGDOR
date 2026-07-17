# commands.py
# Author: Adam He <adamyhe@gmail.com>

import argparse
import io
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pybigtools
import torch
import tqdm
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from chiaroscuro.calibration import (
    candidate_intervals_to_bed3,
    finite_scores,
    null_log_records,
    records_to_bed3,
    records_to_summit_bed3,
    score_centered_windows_from_array,
    score_centered_windows_from_bigwig,
    score_peak_records_from_array,
    score_peak_records_from_bigwig,
    write_calibration_figure,
    write_calibration_table,
    write_null_log,
)
from chiaroscuro.data_transforms import normalization
from chiaroscuro.peaks import call_peaks, call_profile_peaks, resolve_seed_score
from chiaroscuro.predict import predict_genome
from chiaroscuro.stats import (
    compute_fdr,
    score_peaks,
    score_peaks_from_array,
    select_fdr_threshold,
    shuffle_peaks,
    shuffle_peaks_within_intervals,
    subtract_intervals_df,
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
    pl_bw = pybigtools.open(pl_bigwig)
    pl_chrom_sizes = dict(pl_bw.chroms())
    pl_bw.close()

    mn_bw = pybigtools.open(mn_bigwig)
    mn_chroms = set(mn_bw.chroms().keys())
    mn_bw.close()

    return {c: size for c, size in pl_chrom_sizes.items() if c in mn_chroms}


def _peak_params(args):
    threshold = args.min_score
    mode = getattr(args, "mode", "simple")
    params = {
        "threshold": threshold,
        "mode": mode,
        "max_gap": getattr(args, "max_gap", 0),
        "min_width": getattr(args, "min_width", 0),
        "max_width": getattr(args, "max_width", None),
        "seed_score": getattr(args, "seed_score", None),
        "smooth_bins": getattr(args, "smooth_bins", 3),
        "valley_fraction": getattr(args, "valley_fraction", 0.5),
        "boundary_fraction": getattr(args, "boundary_fraction", 0.0),
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
        pybigtools.open(params["support_plus"]),
        pybigtools.open(params["support_minus"]),
    )


def _has_support(support_handles, params, chrom, start, end):
    if support_handles is None:
        return True
    pl_bw, mn_bw = support_handles
    pl = np.nan_to_num(np.array(pl_bw.values(chrom, start, end), dtype=np.float32))
    mn = np.abs(
        np.nan_to_num(np.array(mn_bw.values(chrom, start, end), dtype=np.float32))
    )
    return max(pl.max(initial=0.0), mn.max(initial=0.0)) >= params["min_support_signal"]


def _call_chrom_peaks(chrom, intervals, params, support_handles=None):
    peaks = []
    threshold = params["threshold"]
    mode = params["mode"]

    if mode == "simple":
        passing = [(s, e, v) for s, e, v in intervals if v >= threshold]
        for peak in call_peaks(passing, min_score=threshold):
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
            f"{peak['chrom']}\t{peak['start']}\t{peak['end']}\t"
            f"{peak['score']:.6g}\n"
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


def _validate_calibration_args(args):
    if args.n_shuffle <= 0:
        raise ValueError("--n_shuffle must be > 0 when --calibrate is used.")
    if args.n_thresholds <= 1:
        raise ValueError("--n_thresholds must be > 1 when --calibrate is used.")
    if not 0 <= args.calibration_fdr_target <= 1:
        raise ValueError("--calibration_fdr_target must be in [0, 1].")
    if args.calibration_stat not in {"summit", "smoothed_summit", "max", "mean"}:
        raise ValueError(
            "--calibration_stat must be summit, smoothed_summit, max, or mean."
        )
    if getattr(args, "calibration_smooth_bins", 1) < 1:
        raise ValueError("--calibration_smooth_bins must be >= 1.")
    if getattr(args, "threshold_grid", "quantile") not in {
        "quantile",
        "linear",
        "logit",
        "unique",
    }:
        raise ValueError("--threshold_grid must be quantile, linear, logit, or unique.")
    if getattr(args, "calibration_plot_scale", "logit") not in {"logit", "score"}:
        raise ValueError("--calibration_plot_scale must be logit or score.")
    null_exclusion_margin = getattr(args, "null_exclusion_margin", None)
    if null_exclusion_margin is not None and null_exclusion_margin < 0:
        raise ValueError("--null_exclusion_margin must be >= 0.")
    if getattr(args, "raw_output", None) == args.output:
        raise ValueError("--raw_output must differ from --output.")


def _exclude_peaks_from_allowed(allowed_df, chrom_peaks, chrom, margin):
    """Subtract called peaks (± margin) from a candidate-null allowed region.

    No-op when ``margin`` is ``None`` (the default), preserving prior
    behavior where candidate-null placement could land on real peaks'
    own footprint.
    """
    if margin is None:
        return allowed_df
    exclude_df = records_to_bed3(chrom_peaks)
    return subtract_intervals_df(allowed_df, exclude_df, margin=margin, chroms=[chrom])


def _report_null_placement_shortfall(null_placement_stats, null_exclusion_margin, verbose):
    """Warn when --null_exclusion_margin leaves too little territory to place
    the requested number of null draws (chrom_expected = n_peaks * n_shuffle).
    """
    if not verbose or null_exclusion_margin is None or not null_placement_stats:
        return
    total_expected = sum(exp for _, exp, _ in null_placement_stats)
    if total_expected == 0:
        return
    total_placed = sum(placed for _, _, placed in null_placement_stats)
    short = [(c, exp, p) for c, exp, p in null_placement_stats if p < exp]
    print(
        f"Null placement after --null_exclusion_margin={null_exclusion_margin}: "
        f"{total_placed:,}/{total_expected:,} draws placed "
        f"({len(short)}/{len(null_placement_stats)} chromosomes short)"
    )
    if short:
        worst = sorted(short, key=lambda t: t[2] - t[1])[:5]
        for chrom, exp, placed in worst:
            print(
                f"  {chrom}: placed {placed:,}/{exp:,} null draws — candidate "
                "territory may be exhausted after exclusion; consider a smaller margin"
            )


def _default_raw_peak_output(output):
    if output.endswith(".bed.gz"):
        return f"{output[:-len('.bed.gz')]}.raw.bed.gz"
    if output.endswith(".bed"):
        return f"{output[:-len('.bed')]}.raw.bed"
    if output.endswith(".gz"):
        return f"{output[:-len('.gz')]}.raw.gz"
    return f"{output}.raw"


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

    If ``args.calibrate`` is set, peaks are instead calibrated against an
    empirical null built by shuffling peaks within the input bigWig, mirroring
    ``cmd_pipeline``'s streamed calibration but reading scores directly from
    the bigWig instead of re-running the model. Raw (pre-filter) and
    calibrated peak BEDs are written separately.

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
    if getattr(args, "calibrate", False):
        _validate_calibration_args(args)

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

    params = _peak_params(args)

    if getattr(args, "calibrate", False):
        try:
            _run_peaks_calibrated(args, in_bw, chrom_sizes, chrom_intervals, params)
        finally:
            in_bw.close()
        return

    in_bw.close()

    if args.verbose:
        n_pass = sum(
            1
            for ivals in chrom_intervals.values()
            for _, _, v in ivals
            if v >= _candidate_threshold(params)
        )
        if n_pass == 0:
            print("No bins pass threshold; writing empty BED file")
        else:
            print(
                f"Candidate threshold: {_candidate_threshold(params):.6f} "
                f"({n_pass} bins pass)"
            )

    support_handles = _open_support_handles(params)

    def _write_peaks(out_bed):
        n = 0
        for chrom in sorted(chrom_sizes):
            n += _write_chrom_peaks(
                out_bed, chrom, chrom_intervals[chrom], params, support_handles
            )
        return n

    try:
        n_peaks, out_path = _write_peak_file(args.output, _write_peaks)
    finally:
        if support_handles is not None:
            support_handles[0].close()
            support_handles[1].close()

    if args.verbose:
        print(f"{n_peaks} peaks written to {out_path}")


def _run_peaks_calibrated(args, in_bw, chrom_sizes, chrom_intervals, params):
    candidate_threshold = _candidate_threshold(params)
    support_handles = _open_support_handles(params)
    rng = np.random.default_rng(args.calibration_seed)
    raw_output = args.raw_output or _default_raw_peak_output(args.output)
    calibration_smooth_bins = getattr(args, "calibration_smooth_bins", 5)

    peak_records = []
    real_score_lists = []
    null_score_lists = []
    null_log_path = getattr(args, "calibration_null_log", None)
    null_log_rows = [] if null_log_path is not None else None
    null_exclusion_margin = getattr(args, "null_exclusion_margin", None)
    null_placement_stats = [] if null_exclusion_margin is not None else None

    try:
        for chrom in sorted(chrom_sizes):
            intervals = chrom_intervals[chrom]
            chrom_peaks = _call_chrom_peaks(chrom, intervals, params, support_handles)
            if len(chrom_peaks) == 0:
                continue

            peak_records.extend(chrom_peaks)
            real_scores = score_peak_records_from_bigwig(
                chrom_peaks,
                in_bw,
                chrom_sizes,
                chrom,
                args.calibration_stat,
                getattr(args, "output_stride", 16),
                calibration_smooth_bins,
            )
            real_score_lists.append(np.asarray(real_scores, dtype=np.float32))

            if args.null_scope == "candidate":
                candidate_ivals = [
                    (s, e, v) for s, e, v in intervals if v >= candidate_threshold
                ]
                allowed_df = candidate_intervals_to_bed3(chrom, candidate_ivals)
                allowed_df = _exclude_peaks_from_allowed(
                    allowed_df,
                    chrom_peaks,
                    chrom,
                    getattr(args, "null_exclusion_margin", None),
                )
            else:
                allowed_df = pd.DataFrame(
                    [(chrom, 0, int(chrom_sizes[chrom]))],
                    columns=["chrom", "start", "end"],
                )

            null_source_df = (
                records_to_summit_bed3(chrom_peaks)
                if args.calibration_stat in {"summit", "smoothed_summit"}
                else records_to_bed3(chrom_peaks)
            )
            chrom_placed = 0
            for _ in range(args.n_shuffle):
                null_df = shuffle_peaks_within_intervals(
                    null_source_df, allowed_df, [chrom], rng
                )
                chrom_placed += len(null_df)
                if args.calibration_stat == "smoothed_summit":
                    null_scores = score_centered_windows_from_bigwig(
                        null_df,
                        in_bw,
                        chrom_sizes,
                        chrom,
                        getattr(args, "output_stride", 16),
                        calibration_smooth_bins,
                    )
                else:
                    null_scores = score_peaks(
                        in_bw,
                        null_df,
                        chrom_sizes,
                        (
                            "max"
                            if args.calibration_stat == "summit"
                            else args.calibration_stat
                        ),
                        [chrom],
                    )
                null_score_lists.append(finite_scores(null_scores))
                if null_log_rows is not None:
                    null_log_rows.extend(null_log_records(null_df, null_scores))
            if null_placement_stats is not None:
                null_placement_stats.append(
                    (chrom, len(null_source_df) * args.n_shuffle, chrom_placed)
                )
    finally:
        if support_handles is not None:
            support_handles[0].close()
            support_handles[1].close()

    _report_null_placement_shortfall(
        null_placement_stats, null_exclusion_margin, args.verbose
    )

    if null_log_path is not None:
        write_null_log(null_log_path, null_log_rows)
        if args.verbose:
            print(f"{len(null_log_rows):,} null draws logged to {null_log_path}")

    if len(peak_records) == 0:
        def _write_empty(out_bed):
            return 0

        raw_n_peaks, raw_out_path = _write_peak_file(raw_output, _write_empty)
        n_peaks, out_path = _write_peak_file(args.output, _write_empty)
        if args.verbose:
            print(
                "No peaks before calibration; wrote empty raw/calibrated "
                f"BEDs to {raw_out_path} and {out_path}"
            )
        return

    real_scores_all = (
        np.concatenate(real_score_lists)
        if real_score_lists
        else np.array([], dtype=np.float32)
    )
    real_scores = finite_scores(real_scores_all)
    null_scores = (
        np.concatenate(null_score_lists)
        if null_score_lists
        else np.array([], dtype=np.float32)
    )

    if len(real_scores) == 0:
        raise ValueError("No finite peak scores available for calibration.")

    thresholds, n_real, n_null, fdr = compute_fdr(
        real_scores,
        null_scores,
        args.n_shuffle,
        args.n_thresholds,
        getattr(args, "threshold_grid", "quantile"),
    )
    threshold_at_target, n_at_target = select_fdr_threshold(
        thresholds, n_real, fdr, args.calibration_fdr_target
    )

    if args.calibration_curve is not None:
        write_calibration_table(
            args.calibration_curve, thresholds, n_real, n_null, fdr, len(real_scores)
        )
    calibration_figure = getattr(args, "calibration_figure", None)
    if calibration_figure is not None:
        write_calibration_figure(
            calibration_figure,
            real_scores,
            null_scores,
            thresholds,
            n_real,
            fdr,
            args.calibration_stat,
            args.calibration_fdr_target,
            threshold_at_target,
            getattr(args, "calibration_plot_scale", "logit"),
        )

    if np.isnan(threshold_at_target):
        passing = np.zeros(len(peak_records), dtype=bool)
    else:
        passing = real_scores_all >= threshold_at_target

    def _write_raw(out_bed):
        for peak in peak_records:
            _write_peak_record(out_bed, peak, params)
        return len(peak_records)

    def _write_calibrated(out_bed):
        n = 0
        for peak, keep in zip(peak_records, passing):
            if keep:
                _write_peak_record(out_bed, peak, params)
                n += 1
        return n

    raw_n_peaks, raw_out_path = _write_peak_file(raw_output, _write_raw)
    n_peaks, out_path = _write_peak_file(args.output, _write_calibrated)

    if args.verbose:
        print(f"Real peaks scored: {len(real_scores):,}")
        print(
            f"Null peaks scored: {len(null_scores):,} ({args.n_shuffle} shuffle(s))"
        )
        print(f"Calibration stat: {args.calibration_stat}")
        print(f"FDR target: {args.calibration_fdr_target:.3f}")
        if np.isnan(threshold_at_target):
            print("Score threshold: N/A (target FDR never reached)")
        else:
            print(f"Score threshold: {threshold_at_target:.6f}")
            print(f"Peaks at target: {n_at_target:,}")
        print(f"{raw_n_peaks} raw peaks written to {raw_out_path}")
        print(f"{n_peaks} calibrated peaks written to {out_path}")
        if calibration_figure is not None:
            print(f"Saved calibration figure to {calibration_figure}")


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
    if getattr(args, "calibrate", False):
        _validate_calibration_args(args)

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
                mode=getattr(args, "mode", "simple"),
                max_gap=getattr(args, "max_gap", 0),
                min_width=getattr(args, "min_width", 0),
                max_width=getattr(args, "max_width", None),
                seed_score=getattr(args, "seed_score", None),
                smooth_bins=getattr(args, "smooth_bins", 3),
                valley_fraction=getattr(args, "valley_fraction", 0.5),
                boundary_fraction=getattr(args, "boundary_fraction", 0.0),
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
                intervals = [
                    (
                        int(i * args.output_stride),
                        int((i + 1) * args.output_stride),
                        float(probs[i]),
                    )
                    for i in bin_indices
                ]
                n += _write_chrom_peaks(
                    out_bed, chrom, intervals, params, support_handles
                )
            return n

        try:
            n_peaks, out_path = _write_peak_file(args.output, _write_peaks)
        finally:
            if support_handles is not None:
                support_handles[0].close()
                support_handles[1].close()

        if args.verbose:
            print(f"{n_peaks} peaks written to {out_path}")

    def _run_direct_calibrated():
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
        rng = np.random.default_rng(args.calibration_seed)
        raw_output = args.raw_output or _default_raw_peak_output(args.output)
        calibration_smooth_bins = getattr(args, "calibration_smooth_bins", 5)

        peak_records = []
        real_score_lists = []
        null_score_lists = []
        null_log_path = getattr(args, "calibration_null_log", None)
        null_log_rows = [] if null_log_path is not None else None
        null_exclusion_margin = getattr(args, "null_exclusion_margin", None)
        null_placement_stats = [] if null_exclusion_margin is not None else None

        try:
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
                intervals = [
                    (
                        int(i * args.output_stride),
                        int(min((i + 1) * args.output_stride, chrom_len)),
                        float(probs[i]),
                    )
                    for i in bin_indices
                ]
                chrom_peaks = _call_chrom_peaks(
                    chrom, intervals, params, support_handles
                )
                if len(chrom_peaks) == 0:
                    continue

                peak_records.extend(chrom_peaks)
                chrom_peaks_df = records_to_bed3(chrom_peaks)
                real_scores = score_peak_records_from_array(
                    chrom_peaks,
                    probs,
                    chrom,
                    args.output_stride,
                    args.calibration_stat,
                    calibration_smooth_bins,
                )
                real_score_lists.append(np.asarray(real_scores, dtype=np.float32))

                if args.null_scope == "candidate":
                    allowed_df = candidate_intervals_to_bed3(chrom, intervals)
                    allowed_df = _exclude_peaks_from_allowed(
                        allowed_df,
                        chrom_peaks,
                        chrom,
                        getattr(args, "null_exclusion_margin", None),
                    )
                else:
                    allowed_df = pd.DataFrame(
                        [(chrom, 0, int(chrom_len))],
                        columns=["chrom", "start", "end"],
                    )

                null_source_df = (
                    records_to_summit_bed3(chrom_peaks)
                    if args.calibration_stat in {"summit", "smoothed_summit"}
                    else chrom_peaks_df
                )
                chrom_placed = 0
                for _ in range(args.n_shuffle):
                    null_df = shuffle_peaks_within_intervals(
                        null_source_df,
                        allowed_df,
                        [chrom],
                        rng,
                    )
                    chrom_placed += len(null_df)
                    if args.calibration_stat == "smoothed_summit":
                        null_scores = score_centered_windows_from_array(
                            null_df,
                            probs,
                            chrom,
                            args.output_stride,
                            calibration_smooth_bins,
                        )
                    else:
                        null_scores = score_peaks_from_array(
                            probs,
                            null_df,
                            chrom,
                            args.output_stride,
                            (
                                "max"
                                if args.calibration_stat == "summit"
                                else args.calibration_stat
                            ),
                        )
                    null_score_lists.append(finite_scores(null_scores))
                    if null_log_rows is not None:
                        null_log_rows.extend(null_log_records(null_df, null_scores))
                if null_placement_stats is not None:
                    null_placement_stats.append(
                        (chrom, len(null_source_df) * args.n_shuffle, chrom_placed)
                    )
        finally:
            if support_handles is not None:
                support_handles[0].close()
                support_handles[1].close()

        _report_null_placement_shortfall(
            null_placement_stats, null_exclusion_margin, args.verbose
        )

        if null_log_path is not None:
            write_null_log(null_log_path, null_log_rows)
            if args.verbose:
                print(f"{len(null_log_rows):,} null draws logged to {null_log_path}")

        if len(peak_records) == 0:
            def _write_empty(out_bed):
                return 0

            raw_n_peaks, raw_out_path = _write_peak_file(raw_output, _write_empty)
            n_peaks, out_path = _write_peak_file(args.output, _write_empty)
            if args.verbose:
                print(
                    "No peaks before calibration; wrote empty raw/calibrated "
                    f"BEDs to {raw_out_path} and {out_path}"
                )
            return

        real_scores_all = (
            np.concatenate(real_score_lists)
            if real_score_lists
            else np.array([], dtype=np.float32)
        )
        real_scores = finite_scores(real_scores_all)
        null_scores = (
            np.concatenate(null_score_lists)
            if null_score_lists
            else np.array([], dtype=np.float32)
        )

        if len(real_scores) == 0:
            raise ValueError("No finite peak scores available for calibration.")
        if args.n_shuffle <= 0:
            raise ValueError("--n_shuffle must be > 0 when --calibrate is used.")

        thresholds, n_real, n_null, fdr = compute_fdr(
            real_scores,
            null_scores,
            args.n_shuffle,
            args.n_thresholds,
            getattr(args, "threshold_grid", "quantile"),
        )
        threshold_at_target, n_at_target = select_fdr_threshold(
            thresholds, n_real, fdr, args.calibration_fdr_target
        )

        if args.calibration_curve is not None:
            write_calibration_table(
                args.calibration_curve,
                thresholds,
                n_real,
                n_null,
                fdr,
                len(real_scores),
            )
        calibration_figure = getattr(args, "calibration_figure", None)
        if calibration_figure is not None:
            write_calibration_figure(
                calibration_figure,
                real_scores,
                null_scores,
                thresholds,
                n_real,
                fdr,
                args.calibration_stat,
                args.calibration_fdr_target,
                threshold_at_target,
                getattr(args, "calibration_plot_scale", "logit"),
            )

        if np.isnan(threshold_at_target):
            passing = np.zeros(len(peak_records), dtype=bool)
        else:
            passing = real_scores_all >= threshold_at_target

        def _write_raw(out_bed):
            for peak in peak_records:
                _write_peak_record(out_bed, peak, params)
            return len(peak_records)

        def _write_calibrated(out_bed):
            n = 0
            for peak, keep in zip(peak_records, passing):
                if keep:
                    _write_peak_record(out_bed, peak, params)
                    n += 1
            return n

        raw_n_peaks, raw_out_path = _write_peak_file(raw_output, _write_raw)
        n_peaks, out_path = _write_peak_file(args.output, _write_calibrated)

        if args.verbose:
            print(f"Real peaks scored: {len(real_scores):,}")
            print(
                f"Null peaks scored: {len(null_scores):,} "
                f"({args.n_shuffle} shuffle(s))"
            )
            print(f"Calibration stat: {args.calibration_stat}")
            print(f"FDR target: {args.calibration_fdr_target:.3f}")
            if np.isnan(threshold_at_target):
                print("Score threshold: N/A (target FDR never reached)")
            else:
                print(f"Score threshold: {threshold_at_target:.6f}")
                print(f"Peaks at target: {n_at_target:,}")
            print(f"{raw_n_peaks} raw peaks written to {raw_out_path}")
            print(f"{n_peaks} calibrated peaks written to {out_path}")
            if calibration_figure is not None:
                print(f"Saved calibration figure to {calibration_figure}")

    if args.save_bigwig is not None:
        if getattr(args, "calibrate", False):
            raise ValueError(
                "--calibrate uses streamed probabilities; omit --save_bigwig."
            )
        bw_prefix = (
            args.save_bigwig[: -len(".prob.bw")]
            if args.save_bigwig.endswith(".prob.bw")
            else args.save_bigwig
        )
        _run_with_bigwig(bw_prefix)
    elif getattr(args, "calibrate", False):
        _run_direct_calibrated()
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
        s = score_peaks(
            bw, null_df, chrom_sizes, args.stat, chroms, verbose=args.verbose
        )
        null_score_lists.append(s[~np.isnan(s)])
    bw.close()

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
        ax2.plot(thresholds, recall, color="steelblue", linewidth=1.5, linestyle="-", label="Recall")
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
