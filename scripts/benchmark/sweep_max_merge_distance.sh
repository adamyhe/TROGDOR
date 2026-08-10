#!/usr/bin/env bash
# sweep_max_merge_distance.sh
# Author: Adam He <adamyhe@gmail.com>
#
# Sweeps --max_merge_distance on top of the current best single-threshold
# config (profile mode, seed_score=0.3 -- see docs/peak_calling_handoff.md
# item 2b) and benchmarks each resulting candidate BED against ground truth
# with compare_peaks.py. Same direct-benchmark methodology already used for
# the seed_score/boundary_fraction sweeps (docs/peak_calling_handoff.md item
# 2b, 2a) -- see docs/peak_calling_handoff.md item 2c-i for what this sweep
# is meant to answer.
#
# Usage
# -----
#   SCORES_BW=GM12878.trogdor.prob.bw \
#   TRUTH_BED=GM12878.positive.bed.gz \
#   CHROM_SIZES=hg19.chrom.sizes \
#   ./scripts/benchmark/sweep_max_merge_distance.sh
#
# Required env vars
# ------------------
#   SCORES_BW    Scored probability bigWig (output of `trogdor score`),
#                stored densely enough to cover --seed_score (0.3 here).
#   TRUTH_BED    Ground truth peak BED (e.g. GM12878.positive.bed.gz).
#   CHROM_SIZES  Tab-separated chrom.sizes file matching SCORES_BW's assembly.
#
# Optional env vars
# ------------------
#   OUTDIR       Where candidate BEDs + per-run compare_peaks.py logs go
#                (default: ./sweep_max_merge_distance).
#   SEED_SCORE   Held fixed across the sweep (default: 0.3, item 2b's tuned point).
#   MIN_SCORE    Held fixed across the sweep (default: 0.95, the CLI default).
#   DISTANCES    Space-separated max_merge_distance values, bp (default: "200 500 1000 2000").
#   CONDA_ENV    Conda env with trogdor + compare_peaks.py's deps (default: torch).

set -euo pipefail

: "${SCORES_BW:?Set SCORES_BW to the scored probability bigWig}"
: "${TRUTH_BED:?Set TRUTH_BED to the ground truth peak BED}"
: "${CHROM_SIZES:?Set CHROM_SIZES to a matching chrom.sizes file}"

OUTDIR="${OUTDIR:-./sweep_max_merge_distance}"
SEED_SCORE="${SEED_SCORE:-0.3}"
MIN_SCORE="${MIN_SCORE:-0.95}"
DISTANCES="${DISTANCES:-200 500 1000 2000}"
CONDA_ENV="${CONDA_ENV:-torch}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE_PEAKS="$REPO_ROOT/scripts/benchmark/compare_peaks.py"

mkdir -p "$OUTDIR"

run() { conda run -n "$CONDA_ENV" "$@"; }

# Pull the first bare decimal number following a label out of a
# compare_peaks.py log line -- robust to exact column spacing, unlike a
# fixed awk field index.
extract_metric() {
    grep -m1 "$2" "$1" | grep -oE '[0-9]+\.[0-9]+' | head -1
}

SUMMARY="$OUTDIR/summary.tsv"
printf 'max_merge_distance\tn_peaks\tbin_precision\tbin_recall\tbin_f1\tbin_jaccard\tpeak_sensitivity\tpeak_ppv\n' > "$SUMMARY"

run_one() {
    local label="$1" mmd_args=("${@:2}")
    local bed="$OUTDIR/peaks.${label}.bed"
    local log="$OUTDIR/compare.${label}.log"

    echo "[$label] calling peaks..."
    run trogdor peaks -i "$SCORES_BW" -o "$bed" \
        --mode profile --min_score "$MIN_SCORE" --seed_score "$SEED_SCORE" \
        "${mmd_args[@]+"${mmd_args[@]}"}" -v

    echo "[$label] benchmarking against truth..."
    run python "$COMPARE_PEAKS" -c "$bed" -t "$TRUTH_BED" -g "$CHROM_SIZES" -v \
        > "$log" 2>&1

    local n_peaks bin_p bin_r bin_f1 bin_j peak_sens peak_ppv
    n_peaks=$(wc -l < "$bed" | tr -d ' ')
    bin_p=$(extract_metric "$log" '^Precision:')
    bin_r=$(extract_metric "$log" '^Recall:')
    bin_f1=$(extract_metric "$log" '^F1:')
    bin_j=$(extract_metric "$log" '^Jaccard:')
    peak_sens=$(extract_metric "$log" 'Sensitivity (GT covered')
    peak_ppv=$(extract_metric "$log" 'PPV (candidate covered')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$n_peaks" "$bin_p" "$bin_r" "$bin_f1" "$bin_j" "$peak_sens" "$peak_ppv" \
        >> "$SUMMARY"
    echo "[$label] done -> $log"
}

# Baseline: seed_score=0.3, no distance cap -- the current best
# single-threshold point (item 2b), for direct comparison against each
# swept max_merge_distance value.
run_one "none"

for d in $DISTANCES; do
    run_one "$d" --max_merge_distance "$d"
done

echo
echo "=== Summary (also written to $SUMMARY) ==="
column -t -s "$(printf '\t')" "$SUMMARY"
