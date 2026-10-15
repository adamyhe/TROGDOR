# bigwig.py
# Author: Adam He <adamyhe@gmail.com>

"""
Thin helpers around figwig for the bigWig access patterns TROGDOR uses:
whole-region value reads, interval (run) reads of scored bigWigs, and
interval writes.
"""

import math

import figwig
import numpy as np

# Width of each window a single region read is split into, so figwig can
# decompress one region on several threads.
_READ_WINDOW = 1 << 20
# Bases per chunk when scanning a chromosome for records, to bound memory.
_RECORD_CHUNK = 1 << 24


def open_bigwig(path):
    """Open a bigWig for reading. Returns a ``figwig.BigWigReader``."""
    return figwig.BigWigReader(path)


def chrom_sizes(bw):
    """Return ``{chrom: length}`` for an open reader, in file order."""
    return dict(bw.chrom_sizes)


def read_values(bw, chrom, start, end, missing=0.0):
    """Read per-base values over ``[start, end)`` as a float32 array.

    Bases no interval covers are ``missing`` (0.0, as with pybigtools'
    ``values()``); bases past the end of the chromosome are NaN.
    """
    width = end - start
    if width <= 0:
        return np.zeros(0, dtype=np.float32)
    window = min(width, _READ_WINDOW)
    starts = np.arange(start, end, window, dtype=np.int64)
    vals = bw.read(chrom, starts, window, missing=missing)
    return vals.reshape(-1)[:width]


def read_runs(bw, chrom, chrom_len):
    """Return ``(starts, ends, values)`` of the covered runs on ``chrom``.

    A run is a maximal stretch of bases with the same non-NaN value; bases
    no interval covers are skipped. Adjacent intervals in the file with equal
    values come back as one run (see :func:`split_runs`).
    """
    run_starts, run_ends, run_vals = [], [], []
    for offset in range(0, chrom_len, _RECORD_CHUNK):
        v = read_values(
            bw, chrom, offset, min(offset + _RECORD_CHUNK, chrom_len), missing=np.nan
        )
        valid = ~np.isnan(v)
        same = (v[1:] == v[:-1]) | (~valid[1:] & ~valid[:-1])
        bounds = np.flatnonzero(~same) + 1
        s = np.concatenate([[0], bounds])
        e = np.concatenate([bounds, [len(v)]])
        keep = valid[s]
        run_starts.append(s[keep] + offset)
        run_ends.append(e[keep] + offset)
        run_vals.append(v[s[keep]])

    if not run_starts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float32)
    starts = np.concatenate(run_starts).astype(np.int64)
    ends = np.concatenate(run_ends).astype(np.int64)
    values = np.concatenate(run_vals).astype(np.float32)

    # Rejoin runs that were cut at a chunk boundary.
    joined = np.flatnonzero((ends[:-1] == starts[1:]) & (values[:-1] == values[1:]))
    if len(joined):
        first = np.ones(len(starts), dtype=bool)
        first[joined + 1] = False
        last = np.r_[np.flatnonzero(first)[1:] - 1, len(starts) - 1]
        starts, ends, values = starts[first], ends[last], values[first]
    return starts, ends, values


def infer_bin_size(run_arrays):
    """Infer the bin width of a binned track from its runs.

    ``run_arrays`` is an iterable of ``(starts, ends, values)``. A TROGDOR
    score bigWig holds bins of ``output_stride`` bp aligned to multiples of
    it, so every run boundary is a multiple of the stride; their GCD recovers
    it. Returns ``None`` if there are no runs.
    """
    g = 0
    for starts, ends, _ in run_arrays:
        if len(starts):
            g = math.gcd(g, int(np.gcd.reduce(starts)), int(np.gcd.reduce(ends)))
    return g or None


def split_runs(starts, ends, values, bin_size):
    """Split runs into ``bin_size``-aligned bins, as a list of ``(s, e, v)``."""
    if len(starts) == 0:
        return []
    k_start = starts // bin_size
    k_end = -(-ends // bin_size)
    counts = k_end - k_start
    run_idx = np.repeat(np.arange(len(starts)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    k = k_start[run_idx] + offsets
    bin_starts = np.maximum(k * bin_size, starts[run_idx])
    bin_ends = np.minimum((k + 1) * bin_size, ends[run_idx])
    return list(
        zip(bin_starts.tolist(), bin_ends.tolist(), values[run_idx].tolist())
    )


def write_intervals(path, chrom_sizes, intervals):
    """Write intervals to a bigWig.

    Parameters
    ----------
    path : str
        Output bigWig path.
    chrom_sizes : dict
        ``{chrom: length}``; chromosomes are written in this order.
    intervals : dict
        ``{chrom: (starts, ends, values)}`` of sorted, non-overlapping
        intervals. Chromosomes absent from it are written empty.
    """
    with figwig.BigWigWriter(path, chrom_sizes) as writer:
        for chrom in chrom_sizes:
            if chrom not in intervals:
                continue
            starts, ends, values = intervals[chrom]
            if len(starts) == 0:
                continue
            writer.write(
                chrom,
                np.asarray(starts, dtype=np.int64),
                np.asarray(values, dtype=np.float32),
                ends=np.asarray(ends, dtype=np.int64),
                missing=np.nan,
            )
