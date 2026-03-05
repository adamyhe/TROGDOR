# predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""
The predict function is copied from tangermeme v1.0.2 by Jacob Schreiber to reduce
dependencies. predict_chromosome wraps the predict function to tile predictions
across a whole chromosome.
"""

import torch
from tqdm import trange


def predict(
    model,
    X,
    args=None,
    func=None,
    batch_size=32,
    dtype=None,
    device="cuda",
    verbose=False,
):
    """Make batched predictions in a memory-efficient manner.

    This function will take a PyTorch model and make predictions from it using
    the forward function, with optional additional arguments to the model. The
    additional arguments must have the same batch size as the examples, and the
    i-th example will be given to the model with the i-th index of each
    additional argument.

    Before starting predictions, the model is moved to the specified device. As
    predictions are being made, each batch is also moved to the specified
    device and then moved back to the CPU after predictions are made. Each batch
    is converted to the provided dtype if provided, keeping the original blob of
    examples in the original dtype. These features allow the function to work on
    massive data sets that do not fit in GPU memory. For example, the original
    sequences can be kept as 8-bit integers for compression and each batch will
    be upcast to the desired precision. If a single batch does not fit in memory,
    try lowering the batch size.


    Parameters
    ----------
    model: torch.nn.Module
            The PyTorch model to use to make predictions.

    X: torch.tensor, shape=(-1, len(alphabet), length)
            A one-hot encoded set of sequences to make predictions for.

    args: tuple or list or None, optional
            An optional set of additional arguments to pass into the model. If
            provided, each element in the tuple or list is one input to the model
            and the element must be formatted to be the same batch size as `X`. If
            None, no additional arguments are passed into the forward function.
            Default is None.

    func: function or None, optional
            A function to apply to a batch of predictions after they have been made.
            If None, do nothing to them. Default is None.

    batch_size: int, optional
            The number of examples to make predictions for at a time. Default is 32.

    dtype: str or torch.dtype or None, optional
            The dtype to use with mixed precision autocasting. If None, use the dtype of
            the *model*. This allows you to use int8 to represent large data sets and
            only convert batches to the higher precision, saving memory. Defailt is None.

    device: str or torch.device, optional
            The device to move the model and batches to when making predictions. If
            set to 'cuda' without a GPU, this function will crash and must be set
            to 'cpu'. Default is 'cuda'.

    verbose: bool, optional
            Whether to display a progress bar during predictions. Default is False.


    Returns
    -------
    y: torch.Tensor or list/tuple of torch.Tensors
            The output from the model for each input example. The precise format
            is determined by the model. If the model outputs a single tensor,
            y is a single tensor concatenated across all batches. If the model
            outputs multiple tensors, y is a list of tensors which are each
            concatenated across all batches.
    """

    model = model.to(device).eval()

    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except:
            dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if args is not None:
        for arg in args:
            if arg.shape[0] != X.shape[0]:
                raise ValueError(
                    "Arguments must have the same first " + "dimension as X"
                )

    ###

    y = []
    with torch.no_grad():
        batch_size = min(batch_size, X.shape[0])

        for start in trange(0, X.shape[0], batch_size, disable=not verbose):
            end = start + batch_size
            X_ = X[start:end].type(dtype).to(device)

            if X_.shape[0] == 0:
                continue

            with torch.autocast(device_type=device, dtype=dtype):
                if args is not None:
                    args_ = [a[start:end].type(dtype).to(device) for a in args]
                    y_ = model(X_, *args_)
                else:
                    y_ = model(X_)

            # If a post-processing function is provided, apply it to the raw output
            # from the model.
            if func is not None:
                y_ = func(y_)

            # Move to the CPU
            if isinstance(y_, torch.Tensor):
                y_ = y_.cpu()
            elif isinstance(y_, (list, tuple)):
                y_ = tuple(yi.cpu() for yi in y_)
            else:
                raise ValueError("Cannot interpret output from model.")

            y.append(y_)

    # Concatenate the outputs
    if isinstance(y[0], torch.Tensor):
        y = torch.cat(y)
    else:
        y = [torch.cat(y_) for y_ in list(zip(*y))]

    return y


def predict_chromosome(
    model,
    signal,
    chunk_size=262144,
    overlap=32768,
    output_stride=16,
    batch_size=8,
    device="cuda",
    transform=None,
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
    model = model.to(device).eval()
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

    # Output-space dimensions
    out_total = total_length // output_stride
    out_chunk = chunk_size // output_stride
    out_overlap = overlap // output_stride

    scores = torch.zeros(1, out_total)

    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]

            # Build batch tensor
            chunks = []
            for s in batch_starts:
                chunk = signal[:, s : s + chunk_size]
                if transform is not None:
                    chunk = transform(chunk)
                chunks.append(chunk)
            X_batch = torch.stack(chunks).to(device)  # (B, 2, chunk_size)

            preds = model(X_batch).cpu()  # (B, 1, out_chunk)

            # Write center crops into the output tensor (all indices in output space)
            for i, s in enumerate(batch_starts):
                chunk_idx = batch_start + i
                is_first = chunk_idx == 0
                is_last = s + chunk_size >= total_length

                out_s = s // output_stride

                if is_first:
                    out_start = out_s
                    pred_start = 0
                else:
                    out_start = out_s + out_overlap
                    pred_start = out_overlap

                if is_last:
                    out_end = out_total
                    pred_end = out_chunk
                else:
                    out_end = out_s + out_chunk - out_overlap
                    pred_end = out_chunk - out_overlap

                scores[:, out_start:out_end] = preds[i, :, pred_start:pred_end]

    return scores


MIT_LICENSE = """
MIT License

Copyright (c) 2024 Jacob Schreiber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__license__ = MIT_LICENSE
