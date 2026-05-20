#!/usr/bin/env python3
"""Evaluate TROGDOR calls against a panel of imperfect reference BEDs.

The manifest is a TSV with required columns:

    cell_type, prob_bw, calls_bed, reference_name, reference_bed

Optional annotation columns:

    tss_bed, promoter_bed, enhancer_bed, bidirectional_bed, gene_body_bed,
    blacklist_bed, reproducible_bed

The script reports reference-dependent empirical metrics and stratifies
unmatched TROGDOR calls instead of treating every unmatched call as a simple
false positive.
"""

import argparse

import numpy as np
import pandas as pd
import pybigtools
from torcheval.metrics.functional import binary_auprc, binary_auroc
import torch

from chiaroscuro.utils import encode_labels


OPTIONAL_BEDS = [
    "tss_bed",
    "promoter_bed",
    "enhancer_bed",
    "bidirectional_bed",
    "gene_body_bed",
    "blacklist_bed",
    "reproducible_bed",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TROGDOR calls across a multi-reference truth panel."
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="TSV manifest with cell_type/prob_bw/calls_bed/reference_name/reference_bed.",
    )
    p.add_argument(
        "--chrom_sizes",
        required=True,
        help="Tab-separated chrom.sizes file.",
    )
    p.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for output TSV files.",
    )
    p.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Bin size in bp for bin-level metrics (default: 16).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=200,
        help="Half-width in bp for centre-window hits (default: 200).",
    )
    p.add_argument(
        "--near_tss_window",
        type=int,
        default=1000,
        help="Distance in bp for near-TSS unmatched category (default: 1000).",
    )
    p.add_argument(
        "--broad_width",
        type=int,
        default=1000,
        help="Peak width in bp used to flag broad unmatched calls (default: 1000).",
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Chromosome whitelist (default: all chroms in chrom.sizes).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_chrom_sizes(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "length"])
    return dict(zip(df["chrom"].astype(str), df["length"].astype(int)))


def read_optional_bed(path):
    if pd.isna(path) or str(path).strip() == "":
        return None
    return read_bed(path)


def read_bed(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        compression="infer",
        dtype={"chrom": str, "start": int, "end": int},
    )


def coverage_fractions(query_df, subject_df, chrom):
    q = query_df[query_df["chrom"] == chrom].sort_values("start")
    s = subject_df[subject_df["chrom"] == chrom].sort_values("start")
    if len(q) == 0:
        return np.empty(0, dtype=np.float32)
    if len(s) == 0:
        return np.zeros(len(q), dtype=np.float32)

    q_starts = q["start"].to_numpy(dtype=np.int64)
    q_ends = q["end"].to_numpy(dtype=np.int64)
    q_lens = (q_ends - q_starts).astype(np.float32)
    s_starts = s["start"].to_numpy(dtype=np.int64)
    s_ends = s["end"].to_numpy(dtype=np.int64)

    fracs = np.empty(len(q), dtype=np.float32)
    for i in range(len(q)):
        qs, qe = q_starts[i], q_ends[i]
        lo = np.searchsorted(s_ends, qs + 1, side="left")
        hi = np.searchsorted(s_starts, qe, side="left")
        if lo >= hi:
            fracs[i] = 0.0
        else:
            covered = np.sum(
                np.minimum(s_ends[lo:hi], qe) - np.maximum(s_starts[lo:hi], qs)
            )
            fracs[i] = covered / q_lens[i]
    return fracs


def center_window_hits(query_df, subject_df, chrom, window):
    q = query_df[query_df["chrom"] == chrom].sort_values("start")
    s = subject_df[subject_df["chrom"] == chrom].sort_values("start")
    if len(q) == 0:
        return np.empty(0, dtype=bool), np.zeros(len(s), dtype=bool)
    if len(s) == 0:
        return np.zeros(len(q), dtype=bool), np.empty(0, dtype=bool)

    centers = ((q["start"].to_numpy() + q["end"].to_numpy()) // 2)
    w_starts = centers - window
    w_ends = centers + window
    s_starts = s["start"].to_numpy()
    s_ends = s["end"].to_numpy()

    q_hits = np.zeros(len(q), dtype=bool)
    s_hits = np.zeros(len(s), dtype=bool)
    for i in range(len(q)):
        lo = np.searchsorted(s_ends, w_starts[i] + 1, side="left")
        hi = np.searchsorted(s_starts, w_ends[i], side="left")
        if lo < hi:
            q_hits[i] = True
            s_hits[lo:hi] = True
    return q_hits, s_hits


def _overlaps_any(query_df, subject_df, chrom, window=0):
    q = query_df[query_df["chrom"] == chrom].sort_values("start")
    if len(q) == 0:
        return np.empty(0, dtype=bool)
    if subject_df is None:
        return np.zeros(len(q), dtype=bool)

    s = subject_df[subject_df["chrom"] == chrom].sort_values("start")
    if len(s) == 0:
        return np.zeros(len(q), dtype=bool)

    q_starts = q["start"].to_numpy(dtype=np.int64) - window
    q_ends = q["end"].to_numpy(dtype=np.int64) + window
    s_starts = s["start"].to_numpy(dtype=np.int64)
    s_ends = s["end"].to_numpy(dtype=np.int64)

    lo = np.searchsorted(s_ends, q_starts + 1, side="left")
    hi = np.searchsorted(s_starts, q_ends, side="left")
    return lo < hi


def _load_scores(prob_bw, truth_df, chrom_sizes, chroms, output_stride):
    bw = pybigtools.open(prob_bw)
    probs_all = []
    labels_all = []
    for chrom in chroms:
        if chrom not in bw.chroms() or chrom not in chrom_sizes:
            continue
        chrom_len = chrom_sizes[chrom]
        raw = np.nan_to_num(np.array(bw.values(chrom, 0, chrom_len), dtype=np.float32))
        n_bins = chrom_len // output_stride
        if len(raw) < n_bins:
            continue
        probs_all.append(raw[:n_bins])
        labels_all.append(encode_labels(truth_df, chrom, chrom_len, output_stride))
    bw.close()
    if not probs_all:
        return None, None
    return np.concatenate(probs_all), np.concatenate(labels_all)


def _reference_metrics(row, chrom_sizes, chroms, args):
    calls_df = read_bed(row["calls_bed"])
    truth_df = read_bed(row["reference_bed"])
    chroms_eval = [
        c
        for c in chroms
        if c in chrom_sizes
        and c in set(calls_df["chrom"].unique())
        and c in set(truth_df["chrom"].unique())
    ]
    if not chroms_eval:
        return None, calls_df, truth_df

    bin_auroc = float("nan")
    bin_auprc = float("nan")
    probs, labels = _load_scores(
        row["prob_bw"], truth_df, chrom_sizes, chroms_eval, args.output_stride
    )
    if probs is not None and labels is not None and labels.sum() > 0:
        prob_t = torch.from_numpy(probs)
        label_t = torch.from_numpy(labels).long()
        bin_auroc = binary_auroc(prob_t, label_t).item()
        bin_auprc = binary_auprc(prob_t, label_t).item()

    gt_fracs = []
    cand_fracs = []
    cand_win_hits = []
    gt_win_hits = []
    for chrom in chroms_eval:
        gt_fracs.append(coverage_fractions(truth_df, calls_df, chrom))
        cand_fracs.append(coverage_fractions(calls_df, truth_df, chrom))
        cw, gw = center_window_hits(calls_df, truth_df, chrom, args.window)
        cand_win_hits.append(cw)
        gt_win_hits.append(gw)

    gt_fracs = np.concatenate(gt_fracs)
    cand_fracs = np.concatenate(cand_fracs)
    cand_win_hits = np.concatenate(cand_win_hits)
    gt_win_hits = np.concatenate(gt_win_hits)

    metrics = {
        "cell_type": row["cell_type"],
        "reference_name": row["reference_name"],
        "n_chroms": len(chroms_eval),
        "n_reference_peaks": len(gt_fracs),
        "n_calls": len(cand_fracs),
        "bin_auroc": bin_auroc,
        "bin_auprc": bin_auprc,
        "gt_covered_rate": float(np.mean(gt_fracs > 0)) if len(gt_fracs) else np.nan,
        "gt_mean_coverage": float(np.mean(gt_fracs)) if len(gt_fracs) else np.nan,
        "call_overlap_rate": float(np.mean(cand_fracs > 0)) if len(cand_fracs) else np.nan,
        "call_mean_coverage": float(np.mean(cand_fracs)) if len(cand_fracs) else np.nan,
        "center_window_sensitivity": float(np.mean(gt_win_hits))
        if len(gt_win_hits)
        else np.nan,
        "center_window_specificity": float(np.mean(cand_win_hits))
        if len(cand_win_hits)
        else np.nan,
        "empirical_peak_fdr": 1.0 - float(np.mean(cand_win_hits))
        if len(cand_win_hits)
        else np.nan,
    }
    return metrics, calls_df, truth_df


def _unmatched_categories(row, calls_df, truth_df, chroms, args):
    optional = {name: read_optional_bed(row.get(name, "")) for name in OPTIONAL_BEDS}
    records = []
    for chrom in chroms:
        calls = calls_df[calls_df["chrom"] == chrom].sort_values("start")
        if len(calls) == 0:
            continue
        matched = _overlaps_any(calls_df, truth_df, chrom, window=args.window)
        unmatched = calls.loc[~matched].copy()
        if len(unmatched) == 0:
            continue

        flags = {
            "near_tss": _overlaps_any(unmatched, optional["tss_bed"], chrom, args.near_tss_window),
            "annotated_tss_overlap": _overlaps_any(unmatched, optional["tss_bed"], chrom),
            "promoter_overlap": _overlaps_any(unmatched, optional["promoter_bed"], chrom),
            "distal_no_promoter_overlap": ~_overlaps_any(unmatched, optional["promoter_bed"], chrom),
            "enhancer_overlap": _overlaps_any(unmatched, optional["enhancer_bed"], chrom),
            "bidirectional_overlap": _overlaps_any(unmatched, optional["bidirectional_bed"], chrom),
            "gene_body_overlap": _overlaps_any(unmatched, optional["gene_body_bed"], chrom),
            "broad_call": (
                unmatched["end"].to_numpy(dtype=np.int64)
                - unmatched["start"].to_numpy(dtype=np.int64)
            )
            >= args.broad_width,
            "blacklist_overlap": _overlaps_any(unmatched, optional["blacklist_bed"], chrom),
            "reproducible_overlap": _overlaps_any(unmatched, optional["reproducible_bed"], chrom),
        }
        support_keys = [
            "near_tss",
            "annotated_tss_overlap",
            "promoter_overlap",
            "enhancer_overlap",
            "bidirectional_overlap",
            "gene_body_overlap",
            "blacklist_overlap",
            "reproducible_overlap",
        ]
        unsupported = ~np.logical_or.reduce([flags[k] for k in support_keys])
        total = len(unmatched)
        for category, values in flags.items():
            records.append(
                {
                    "cell_type": row["cell_type"],
                    "reference_name": row["reference_name"],
                    "chrom": chrom,
                    "category": category,
                    "n_unmatched": total,
                    "n_category": int(values.sum()),
                    "fraction": float(values.mean()),
                }
            )
        records.append(
            {
                "cell_type": row["cell_type"],
                "reference_name": row["reference_name"],
                "chrom": chrom,
                "category": "unsupported_by_panel",
                "n_unmatched": total,
                "n_category": int(unsupported.sum()),
                "fraction": float(unsupported.mean()),
            }
        )
    return records


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, sep="\t")
    required = {"cell_type", "prob_bw", "calls_bed", "reference_name", "reference_bed"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required column(s): {sorted(missing)}")

    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    chroms = args.chroms if args.chroms is not None else sorted(chrom_sizes)

    metric_rows = []
    category_rows = []
    for _, row in manifest.iterrows():
        if args.verbose:
            print(f"{row['cell_type']} / {row['reference_name']}", flush=True)
        metrics, calls_df, truth_df = _reference_metrics(row, chrom_sizes, chroms, args)
        if metrics is None:
            continue
        metric_rows.append(metrics)
        category_rows.extend(_unmatched_categories(row, calls_df, truth_df, chroms, args))

    metrics_df = pd.DataFrame(metric_rows)
    categories_df = pd.DataFrame(category_rows)

    metrics_path = f"{args.output_prefix}.reference_metrics.tsv"
    categories_path = f"{args.output_prefix}.unmatched_categories.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False, float_format="%.6g")
    categories_df.to_csv(categories_path, sep="\t", index=False, float_format="%.6g")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {categories_path}")


if __name__ == "__main__":
    main()
