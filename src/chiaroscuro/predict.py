# predict.py
# Author: Adam He <adamyhe@gmail.com>

"""
predict_chromosome tiles model predictions across a whole chromosome.
predict_genome wraps predict_chromosome to score the whole genome.
"""

import queue
import threading
import warnings


import numpy as np
import pybigtools
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange


class _ChunkDataset(Dataset):
    """Lazy Dataset of overlapping signal chunks for a single chromosome.

    Slices chunks from a pre-loaded signal tensor on demand, applying an
    optional per-chunk transform (e.g. normalization). Used by
    ``predict_chromosome`` to feed a DataLoader without pre-materializing all
    chunks at once.

    Parameters
    ----------
    signal : torch.Tensor, shape=(2, total_length)
        Full chromosome signal (plus and minus strands).
    starts : list of int
        Start position (in input coordinates) of each chunk.
    chunk_size : int
        Number of input positions per chunk.
    transform : callable or None, optional
        Function applied to each ``(2, chunk_size)`` chunk before it is
        returned. Default is None.
    """

    def __init__(self, signal, starts, chunk_size, transform=None):
        self.signal = signal
        self.starts = starts
        self.chunk_size = chunk_size
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        chunk = self.signal[:, s : s + self.chunk_size]
        if self.transform is not None:
            chunk = self.transform(chunk)
        return chunk


class _ChunkDataset(Dataset):
    """Lazy Dataset of overlapping signal chunks for a single chromosome.

    Slices chunks from a pre-loaded signal tensor on demand, applying an
    optional per-chunk transform (e.g. normalization). Used by
    ``predict_chromosome`` to feed a DataLoader without pre-materializing all
    chunks at once.

    Parameters
    ----------
    signal : torch.Tensor, shape=(2, total_length)
        Full chromosome signal (plus and minus strands).
    starts : list of int
        Start position (in input coordinates) of each chunk.
    chunk_size : int
        Number of input positions per chunk.
    transform : callable or None, optional
        Function applied to each ``(2, chunk_size)`` chunk before it is
        returned. Default is None.
    """

    def __init__(self, signal, starts, chunk_size, transform=None):
        self.signal = signal
        self.starts = starts
        self.chunk_size = chunk_size
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        chunk = self.signal[:, s : s + self.chunk_size]
        if self.transform is not None:
            chunk = self.transform(chunk)
        return chunk


def predict_chromosome(
    model,
    signal,
    chunk_size=262144,
    overlap=32768,
    output_stride=16,
    batch_size=64,
    device="cuda",
    transform=None,
    dtype="auto",
    desc=None,
    verbose=False,
    num_workers=0,
):
    """Score a full chromosome with chunked sliding-window inference.

    Tiles the chromosome into overlapping chunks, runs model forward passes in
    batches, and stitches results by writing only the center crop of each
    prediction into the output tensor (avoiding edge artifacts).

    Parameters
    ----------
    model : torch.nn.Module
        A model that accepts (B, 2, chunk_size) and returns
        (B, 1, chunk_size // output_stride).
    signal : torch.Tensor, shape=(2, total_length)
        Single chromosome signal (raw, unnormalized).
    transform : callable or None, optional
        A function applied to each chunk independently before the forward pass.
        Receives a (2, chunk_size) tensor and returns a transformed tensor of
        the same shape. Use this for per-chunk normalization so that the
        normalization matches training. Default is None (no transform).
    chunk_size : int, optional
        Length of each input chunk fed to the model. Must be divisible by
        output_stride. Default is 262144 (2^18).
    overlap : int, optional
        Number of input positions whose corresponding output bins are cropped
        from each edge of a chunk's prediction. Must be divisible by
        output_stride. Default is 32768 (2^15).
    output_stride : int, optional
        Ratio of input positions to output bins; must match the model's
        downsampling factor. Default is 16.
    batch_size : int, optional
        Number of chunks to process in one forward pass. Default is 8.
    device : str, optional
        Device for inference. Default is "cuda".
    dtype : str or torch.dtype, optional
        Compute dtype for forward passes. ``"auto"`` selects bfloat16 on CUDA
        (if supported) or float32 otherwise. ``None`` uses the model's current
        parameter dtype. float16 is not supported. Default is ``"auto"``.
    desc : str, optional
        Description for tqdm progress bar. Default is None.
    verbose : bool, optional
        Whether to display a tqdm progress bar over chunks. Default is False.
    num_workers : int, optional
        Number of DataLoader worker processes for parallel chunk loading and
        transform. ``0`` means chunks are loaded in the main process. Default
        is 0.

    Returns
    -------
    scores : torch.Tensor, shape=(1, total_length // output_stride)
        Per-bin logit scores for the chromosome.
    """
    if chunk_size % output_stride != 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be divisible by output_stride ({output_stride})"
        )
    if overlap % output_stride != 0:
        raise ValueError(
            f"overlap ({overlap}) must be divisible by output_stride ({output_stride})"
        )
    total_length = signal.shape[1]
    stride = chunk_size - 2 * overlap

    if total_length < chunk_size:
        raise ValueError(
            f"total_length ({total_length}) is smaller than chunk_size ({chunk_size}). "
            f"Pass chunk_size <= {total_length} to score this contig."
        )

    # Build list of chunk start positions (in input coordinates)
    starts = list(range(0, total_length - chunk_size + 1, stride))
    # Ensure the last chunk reaches the end
    if not starts or starts[-1] + chunk_size < total_length:
        starts.append(total_length - chunk_size)

    # DataLoader-based streaming inference
    dataset = _ChunkDataset(signal, starts, chunk_size, transform)
    pin = torch.device(device).type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # dtype resolution (mirrors predict())
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except Exception:
            dtype = torch.float32
    elif dtype == "auto":
        device_type = torch.device(device).type
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if dtype == torch.float16:
        raise ValueError(
            "float16 is not supported; use dtype='auto', torch.bfloat16, or torch.float32."
        )

    all_preds = []
    device_type = torch.device(device).type
    autocast_enabled = dtype == torch.bfloat16
    bar = trange(len(loader), disable=not verbose, desc=desc)
    with torch.no_grad():
        for batch in loader:
            X_ = batch.to(dtype=dtype, device=device, non_blocking=pin)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
                y_ = model(X_)
            all_preds.append(y_.cpu())
            bar.update()
    bar.close()

    preds = torch.cat(all_preds)  # (n_chunks, 1, out_chunk)

    # Stitch center crops into the output tensor
    out_total = total_length // output_stride
    out_chunk = chunk_size // output_stride
    out_overlap = overlap // output_stride

    scores = torch.zeros(1, out_total)

    for chunk_idx, s in enumerate(starts):
        is_first = chunk_idx == 0
        is_last = s + chunk_size >= total_length

        out_s = s // output_stride
        pred_start = 0 if is_first else out_overlap
        pred_end = out_chunk if is_last else out_chunk - out_overlap
        out_start = out_s if is_first else out_s + out_overlap
        out_end = out_total if is_last else out_s + out_chunk - out_overlap

        scores[:, out_start:out_end] = preds[chunk_idx, :, pred_start:pred_end]

    return scores


def _read_chrom(pl_bw, mn_bw, chrom, chrom_len):
    """Read one chromosome from open pybigtools handles into a signal tensor.

    Parameters
    ----------
    pl_bw : pybigtools.BBIRead
        Open plus-strand bigWig handle.
    mn_bw : pybigtools.BBIRead
        Open minus-strand bigWig handle.
    chrom : str
        Chromosome name to query.
    chrom_len : int
        Length of the chromosome in base pairs.

    Returns
    -------
    signal : torch.Tensor, shape=(2, chrom_len), dtype=float32
        Stacked plus- and minus-strand coverage; NaNs replaced with 0,
        minus-strand values made non-negative.
    """
    pl_vals = np.nan_to_num(np.array(pl_bw.values(chrom, 0, chrom_len), dtype=np.float32))
    mn_vals = np.abs(np.nan_to_num(np.array(mn_bw.values(chrom, 0, chrom_len), dtype=np.float32)))
    return torch.tensor(np.stack([pl_vals, mn_vals]))


def _prefetch_worker(pl_bigwig, mn_bigwig, chroms_to_load, chrom_sizes, chunk_size, q):
    """Background thread target that reads chromosomes and enqueues them.

    Opens its own bigWig handles (so the main thread is not blocked), reads
    each chromosome in ``chroms_to_load`` sequentially, and puts
    ``(chrom, chrom_len, signal)`` tuples onto ``q``.  A ``None`` sentinel is
    always placed on the queue when the worker exits (whether normally or due
    to an exception).  If an exception is raised, the exception object itself
    is placed on the queue before the sentinel so the main thread can re-raise
    it.

    Parameters
    ----------
    pl_bigwig : str
        Path to the plus-strand bigWig file.
    mn_bigwig : str
        Path to the minus-strand bigWig file.
    chroms_to_load : list of str
        Chromosomes to read, in order.
    chrom_sizes : dict
        Mapping of chromosome name to length in base pairs.
    chunk_size : int
        Minimum chromosome length; passed through for context (not used
        directly — filtering is done before the worker is started).
    q : queue.Queue
        Thread-safe queue shared with the consumer (main thread).
    """
    pl_bw = pybigtools.open(pl_bigwig)
    mn_bw = pybigtools.open(mn_bigwig)
    try:
        for chrom in chroms_to_load:
            chrom_len = chrom_sizes[chrom]
            signal = _read_chrom(pl_bw, mn_bw, chrom, chrom_len)
            q.put((chrom, chrom_len, signal))
    except Exception as exc:
        q.put(exc)
    finally:
        pl_bw.close()
        mn_bw.close()
        q.put(None)  # sentinel


def predict_genome(
    model,
    pl_bigwig,
    mn_bigwig,
    chroms=None,
    output_stride=16,
    chunk_size=262144,
    overlap=32768,
    batch_size=64,
    transform=None,
    device="cuda",
    dtype="auto",
    verbose=False,
    num_workers=0,
):
    """Score all chromosomes in a pair of bigWig files.

    Generator that yields ``(chrom, chrom_len, probs)`` for each scored
    chromosome. Chromosomes absent from the bigWig or shorter than
    ``chunk_size`` are skipped (with a warning when ``verbose=True``).

    Parameters
    ----------
    model : torch.nn.Module
    pl_bigwig : str
        Path to the plus-strand bigWig file.
    mn_bigwig : str
        Path to the minus-strand bigWig file.
    chroms : list of str or None
        Chromosomes to score. If None, all chromosomes in the plus-strand
        bigWig are scored.
    output_stride : int
        Ratio of input positions to output bins. Default 16.
    chunk_size : int
        Input chunk length. Default 262144.
    overlap : int
        Edge overlap in input positions. Default 32768.
    batch_size : int
        Number of chunks per forward pass. Default 8.
    transform : callable or None
        Per-chunk transform applied before inference. Default None.
    device : str
        Device for inference. Default "cuda".
    dtype : str or torch.dtype, optional
        Compute dtype passed through to ``predict_chromosome``. ``"auto"``
        selects bfloat16 on CUDA (if supported) or float32 otherwise.
        Default is ``"auto"``.
    verbose : bool
        Print per-chromosome progress. Default False.
    num_workers : int, optional
        Number of DataLoader worker processes for chunk preprocessing within
        each chromosome. Passed through to ``predict_chromosome``. Default 0.

    Yields
    ------
    chrom : str
    chrom_len : int
    probs : np.ndarray, shape=(chrom_len // output_stride,), dtype=float32
    """
    # Read chrom sizes from plus-strand bigWig to filter chromosomes up front
    _pl_bw = pybigtools.open(pl_bigwig)
    chrom_sizes = dict(_pl_bw.chroms())
    _pl_bw.close()

    chroms_to_score = chroms if chroms is not None else list(chrom_sizes.keys())

    valid_chroms = []
    for chrom in chroms_to_score:
        if chrom not in chrom_sizes:
            warnings.warn(f"Skipping {chrom}: not in bigWig")
            continue
        if chrom_sizes[chrom] < chunk_size:
            warnings.warn(
                f"Skipping {chrom}: length {chrom_sizes[chrom]} bp is shorter than "
                f"chunk_size {chunk_size}. Re-run with chunk_size <= {chrom_sizes[chrom]} to score it."
            )
            continue
        valid_chroms.append(chrom)

    # Prepare model once before the prefetch loop
    model = model.to(device).eval()

    # Start prefetch thread (buffers up to 2 chromosomes ahead)
    q = queue.Queue(maxsize=2)
    thread = threading.Thread(
        target=_prefetch_worker,
        args=(pl_bigwig, mn_bigwig, valid_chroms, chrom_sizes, chunk_size, q),
        daemon=True,
    )
    thread.start()

    with torch.no_grad():
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            chrom, chrom_len, signal = item
            logits = predict_chromosome(
                model,
                signal,
                chunk_size=chunk_size,
                overlap=overlap,
                output_stride=output_stride,
                batch_size=batch_size,
                device=device,
                desc=f"Scoring {chrom} ({chrom_len} bp)",
                transform=transform,
                dtype=dtype,
                verbose=verbose,
                num_workers=num_workers,
            )
            probs = torch.sigmoid(logits).squeeze(0).numpy()
            yield chrom, chrom_len, probs

    thread.join()
