# dataset.py
# Author: Adam He <adamyhe@gmail.com>

"""
This module defines dataset classes used for training TROGDOR. It is included in
this package purely for illustrative purposes/development convenience, and is
not used in deployment.
"""

import numpy as np
import pandas as pd
import pybigtools
import torch
import tqdm


class NascentDataset_(torch.utils.data.Dataset):
    """Dataset backed by pre-processed .npz files.

    Each .npz file must contain:
        "X" : array of shape (N, 2, L) — stranded nascent RNA coverage
        "y" : array of shape (N, 1, L // output_stride) — per-bin TSS labels (float)

    Parameters
    ----------
    npz_files : list of str
        Paths to .npz files to load.
    output_stride : int, optional
        Expected output downsampling factor. Used to document the expected y
        shape; no validation is performed at load time. Default is 16.
    transform : callable or None, optional
        Optional transform applied to the X tensor. Default is None.
    """

    def __init__(self, npz_files, output_stride=16, transform=None, rc_prob=0.0):
        X_list = []
        y_list = []
        for f in tqdm.tqdm(npz_files):
            data = np.load(f)
            X_list.append(data["X"])
            y_list.append(data["y"])
        self.X = torch.from_numpy(np.concatenate(X_list)).float()
        self.y = torch.from_numpy(np.concatenate(y_list)).float()
        self.transform = transform
        self.rc_prob = rc_prob

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_i = self.X[idx]
        y_i = self.y[idx]
        if self.transform is not None:
            X_i = self.transform(X_i)
        if self.rc_prob > 0.0 and torch.rand(1).item() < self.rc_prob:
            X_i = torch.flip(X_i[[1, 0]], dims=[1])  # swap strands, reverse length
            y_i = torch.flip(y_i, dims=[1])  # reverse label bins
        return X_i, y_i


class NascentDataset(torch.utils.data.Dataset):
    """Dataset that loads signal from BigWig files and optionally labels from TSS BED files.

    Tiles each chromosome into overlapping windows of `window_size`. In training
    mode (tss_beds provided), only chromosomes present in the BED files are tiled
    and __getitem__ returns (X, y). In inference mode (tss_beds=None), all
    chromosomes are tiled and __getitem__ returns X only.

    Parameters
    ----------
    pl_bigwigs : list of str
        Plus-strand BigWig files (one per dataset).
    mn_bigwigs : list of str
        Minus-strand BigWig files (one per dataset).
    tss_beds : list of str or None, optional
        BED files of annotated TSS positions (one per dataset). If None, the
        dataset runs in inference mode. Default is None.
    window_size : int, optional
        Length of each tiled window in bp. Default is 2**18 (262144).
    stride : int, optional
        Step between consecutive windows in bp. Default is 2**17 (131072).
    output_stride : int, optional
        Downsampling factor for the label tensor; one bin per output_stride bp.
        Default is 16.
    transform : callable or None, optional
        Optional transform applied to the signal tensor. Default is None.
    """

    def __init__(
        self,
        pl_bigwigs,
        mn_bigwigs,
        tss_beds=None,
        window_size=2**18,
        stride=2**17,
        output_stride=16,
        transform=None,
        rc_prob=0.0,
        max_jitter=0,
        tss_centered=False,
    ):
        if len(pl_bigwigs) != len(mn_bigwigs):
            raise ValueError("pl_bigwigs and mn_bigwigs must have the same length.")
        if tss_beds is not None and len(tss_beds) != len(pl_bigwigs):
            raise ValueError(
                "tss_beds, pl_bigwigs, and mn_bigwigs must all have the same length."
            )
        if tss_centered and tss_beds is None:
            raise ValueError("tss_beds is required when tss_centered=True.")

        self.window_size = window_size
        self.stride = stride
        self.output_stride = output_stride
        self.transform = transform
        self.rc_prob = rc_prob
        self.max_jitter = max_jitter
        self.tss_centered = tss_centered

        # Each element: (dataset_idx, chrom, window_start)
        self.windows = []
        # Maps (dataset_idx, chrom) -> chromosome length for jitter clamping
        self._chrom_lens = {}

        self.pl_bigwigs = pl_bigwigs
        self.mn_bigwigs = mn_bigwigs
        self.tss_beds = tss_beds

        # Pre-load TSS DataFrames once; indexed by dataset_idx in __getitem__
        self._tss_dfs = []

        # Pre-compute window indices from BigWig chromosome sizes
        for dataset_idx, pl_bw in enumerate(tqdm.tqdm(pl_bigwigs)):
            bw = pybigtools.open(pl_bw)
            chrom_sizes = bw.chroms()
            bw.close()

            if tss_beds is not None:
                tss_df = pd.read_csv(
                    tss_beds[dataset_idx],
                    sep="\t",
                    header=None,
                    names=["chrom", "start", "end"],
                )
                self._tss_dfs.append(tss_df)
                tss_chroms = set(tss_df["chrom"].values)

            for chrom, chrom_len in chrom_sizes.items():
                if tss_beds is not None and chrom not in tss_chroms:
                    continue
                self._chrom_lens[(dataset_idx, chrom)] = chrom_len

                if tss_centered:
                    chrom_tss = tss_df[tss_df["chrom"] == chrom]
                    for _, row in chrom_tss.iterrows():
                        tss_center = (int(row["start"]) + int(row["end"])) // 2
                        win_start = tss_center - window_size // 2
                        win_start = int(max(0, min(win_start, chrom_len - window_size)))
                        self.windows.append((dataset_idx, chrom, win_start))
                else:
                    for win_start in range(0, chrom_len - window_size + 1, stride):
                        self.windows.append((dataset_idx, chrom, win_start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        dataset_idx, chrom, win_start = self.windows[idx]

        if self.max_jitter > 0:
            chrom_len = self._chrom_lens[(dataset_idx, chrom)]
            jitter = torch.randint(-self.max_jitter, self.max_jitter + 1, (1,)).item()
            win_start = int(
                max(0, min(win_start + jitter, chrom_len - self.window_size))
            )

        win_end = win_start + self.window_size

        # Load signal from BigWigs
        pl_bw = pybigtools.open(self.pl_bigwigs[dataset_idx])
        mn_bw = pybigtools.open(self.mn_bigwigs[dataset_idx])
        pl_vals = np.array(pl_bw.values(chrom, win_start, win_end), dtype=np.float32)
        mn_vals = np.abs(
            np.array(mn_bw.values(chrom, win_start, win_end), dtype=np.float32)
        )
        pl_bw.close()
        mn_bw.close()

        # Replace NaN with 0
        pl_vals = np.nan_to_num(pl_vals)
        mn_vals = np.nan_to_num(mn_vals)

        X = torch.from_numpy(np.stack([pl_vals, mn_vals])).float()  # (2, L)

        if self.transform is not None:
            X = self.transform(X)

        if self.tss_beds is None:
            return X

        # Build per-bin label tensor
        chrom_tss = self._tss_dfs[dataset_idx]
        chrom_tss = chrom_tss[chrom_tss["chrom"] == chrom]
        n_bins = self.window_size // self.output_stride
        y = np.zeros(n_bins, dtype=np.float32)
        for _, row in chrom_tss.iterrows():
            tss_start = int(row["start"])
            tss_end = int(row["end"])
            first_bin = max(0, (tss_start - win_start) // self.output_stride)
            last_bin = min(n_bins - 1, (tss_end - 1 - win_start) // self.output_stride)
            if first_bin <= last_bin:
                y[first_bin : last_bin + 1] = 1.0

        y = torch.from_numpy(y).unsqueeze(0)  # (1, window_size // output_stride)

        if self.rc_prob > 0.0 and torch.rand(1).item() < self.rc_prob:
            X = torch.flip(X[[1, 0]], dims=[1])
            y = torch.flip(y, dims=[1])

        return X, y


class MixedBatchLoader:
    """Yields mixed batches combining TSS-centered and tiled-genome samples.

    One epoch = one full pass over the TSS dataset (with drop_last). The tiled
    dataset is sampled uniformly with replacement to match, so each epoch sees a
    fresh random draw from the tiled dataset of exactly the same number of batches.

    Parameters
    ----------
    tss_dataset : torch.utils.data.Dataset
        TSS-centered dataset. Defines epoch length; iterated once per epoch.
    tiled_dataset : torch.utils.data.Dataset
        Genome-wide tiled dataset. Sampled uniformly with replacement each epoch.
    batch_size : int
        Total samples per yielded batch.
    tss_fraction : float
        Fraction of each batch from tss_dataset. Must be in (0, 1).
    shuffle : bool, optional
        Whether to shuffle the TSS DataLoader each epoch. Default True.
    num_workers : int, optional
        Worker processes for both DataLoaders. Default 0.
    **dataloader_kwargs
        Forwarded to both DataLoader constructors (e.g. pin_memory).
    """

    def __init__(
        self,
        tss_dataset,
        tiled_dataset,
        batch_size=64,
        tss_fraction=7 / 8,
        shuffle=True,
        num_workers=0,
        **dataloader_kwargs,
    ):
        if not (0.0 < tss_fraction < 1.0):
            raise ValueError(f"tss_fraction must be in (0, 1), got {tss_fraction}.")

        self.tss_batch_size = max(1, round(batch_size * tss_fraction))
        self.tiled_batch_size = batch_size - self.tss_batch_size

        if self.tiled_batch_size < 1:
            raise ValueError(
                f"tiled_batch_size={self.tiled_batch_size}; increase batch_size or "
                f"decrease tss_fraction."
            )

        self.tss_loader = torch.utils.data.DataLoader(
            tss_dataset,
            batch_size=self.tss_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
            **dataloader_kwargs,
        )

        # Sample tiled windows uniformly with replacement to match TSS epoch length.
        # num_samples = exactly enough to fill one batch per TSS batch.
        tiled_num_samples = len(self.tss_loader) * self.tiled_batch_size
        tiled_sampler = torch.utils.data.RandomSampler(
            tiled_dataset, replacement=True, num_samples=tiled_num_samples
        )
        self.tiled_loader = torch.utils.data.DataLoader(
            tiled_dataset,
            batch_size=self.tiled_batch_size,
            sampler=tiled_sampler,
            num_workers=num_workers,
            **dataloader_kwargs,
        )

    def __len__(self):
        return len(self.tss_loader)  # epoch = one full pass over TSS dataset

    def __iter__(self):
        for (tss_X, tss_y), (tiled_X, tiled_y) in zip(
            self.tss_loader, self.tiled_loader
        ):
            yield torch.cat([tiled_X, tss_X], dim=0), torch.cat([tiled_y, tss_y], dim=0)
