"""
Tests for dataset.py — NascentDataset_ (npz-backed) and NascentDataset (bigwig-backed).
"""

import numpy as np
import pybigtools
import pytest
import torch

from chiaroscuro.dataset import MixedBatchLoader, NascentDataset, NascentDataset_

# ---------------------------------------------------------------------------
# NascentDataset_ (npz-backed)
# ---------------------------------------------------------------------------


class TestNascentDataset_:
    @pytest.fixture
    def npz_file(self, tmp_path):
        N, L, output_stride = 20, 256, 16
        X = np.random.rand(N, 2, L).astype(np.float32)
        y = (np.random.rand(N, 1, L // output_stride) > 0.9).astype(np.float32)
        path = str(tmp_path / "data.npz")
        np.savez(path, X=X, y=y)
        return path, N, L, output_stride

    def test_length(self, npz_file):
        """Dataset length must equal the number of examples in the npz file."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        assert len(ds) == N

    def test_multiple_files(self, tmp_path):
        """Length must equal the total number of examples across all npz files."""
        output_stride = 16
        paths = []
        total = 0
        for i in range(3):
            N = 10 + i
            X = np.random.rand(N, 2, 128).astype(np.float32)
            y = np.zeros((N, 1, 128 // output_stride), dtype=np.float32)
            p = str(tmp_path / f"data_{i}.npz")
            np.savez(p, X=X, y=y)
            paths.append(p)
            total += N
        ds = NascentDataset_(paths, output_stride=output_stride)
        assert len(ds) == total

    def test_item_shapes(self, npz_file):
        """Each item must return X of shape (2, L) and y of shape (1, L // output_stride)."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        X_i, y_i = ds[0]
        assert X_i.shape == (2, L), f"X shape: {X_i.shape}"
        assert y_i.shape == (1, L // output_stride), f"y shape: {y_i.shape}"

    def test_item_dtype(self, npz_file):
        """Both X and y must be float32 tensors."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        X_i, y_i = ds[0]
        assert X_i.dtype == torch.float32
        assert y_i.dtype == torch.float32

    def test_transform_applied(self, npz_file):
        """A transform function must be applied to X before returning."""
        path, N, L, output_stride = npz_file

        def double(x):
            return x * 2.0

        ds = NascentDataset_([path], output_stride=output_stride, transform=double)
        X_orig, _ = NascentDataset_([path], output_stride=output_stride)[0]
        X_trans, _ = ds[0]
        assert torch.allclose(X_trans, X_orig * 2.0)

    def test_label_values_in_range(self, npz_file):
        """All label values must lie in [0, 1]."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        for i in range(len(ds)):
            _, y_i = ds[i]
            assert y_i.min() >= 0.0
            assert y_i.max() <= 1.0

    def test_dataloader_compatible(self, npz_file):
        """Dataset must be compatible with torch DataLoader batching."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape == (4, 2, L)
        assert y_batch.shape == (4, 1, L // output_stride)

    def test_last_index(self, npz_file):
        """Indexing the last example (N-1) must not raise and return the correct shape."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        X_i, y_i = ds[N - 1]
        assert X_i.shape == (2, L)

    def test_empty_file_list_raises(self):
        """An empty file list must raise an exception."""
        with pytest.raises(Exception):
            NascentDataset_([])

    def test_rc_prob_zero_unchanged(self, npz_file):
        """rc_prob=0.0 must never flip strands; two datasets with same seed agree."""
        path, N, L, output_stride = npz_file
        ds_no_rc = NascentDataset_([path], output_stride=output_stride, rc_prob=0.0)
        ds_rc = NascentDataset_([path], output_stride=output_stride, rc_prob=0.0)
        for i in range(N):
            X_no_rc, y_no_rc = ds_no_rc[i]
            X_rc, y_rc = ds_rc[i]
            assert torch.allclose(X_no_rc, X_rc)
            assert torch.allclose(y_no_rc, y_rc)

    def test_rc_always_applied(self, npz_file):
        """rc_prob=1.0 must swap strands and reverse both X and y along the length axis."""
        path, N, L, output_stride = npz_file
        ds_orig = NascentDataset_([path], output_stride=output_stride, rc_prob=0.0)
        ds_rc = NascentDataset_([path], output_stride=output_stride, rc_prob=1.0)
        for i in range(N):
            X_orig, y_orig = ds_orig[i]
            X_rc, y_rc = ds_rc[i]
            assert torch.allclose(X_rc[0], torch.flip(X_orig[1], dims=[0]))
            assert torch.allclose(X_rc[1], torch.flip(X_orig[0], dims=[0]))
            assert torch.allclose(y_rc, torch.flip(y_orig, dims=[1]))

    def test_rc_preserves_shape(self, npz_file):
        """Reverse-complement augmentation must not change X or y shapes."""
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride, rc_prob=1.0)
        X_rc, y_rc = ds[0]
        assert X_rc.shape == (2, L)
        assert y_rc.shape == (1, L // output_stride)


# ---------------------------------------------------------------------------
# NascentDataset (bigwig-backed) — unit tests with synthetic BigWig files
# ---------------------------------------------------------------------------


class TestNascentDataset:
    @pytest.fixture
    def bw_files(self, tmp_path):
        pl_path = tmp_path / "pl.bw"
        mn_path = tmp_path / "mn.bw"

        bw = pybigtools.open(str(pl_path), "w")
        bw.write({"chr1": 1024}, [("chr1", 0, 1024, 5.0)])

        bw = pybigtools.open(str(mn_path), "w")
        bw.write({"chr1": 1024}, [("chr1", 0, 1024, 3.0)])

        return str(pl_path), str(mn_path)

    @pytest.fixture
    def tss_bed(self, tmp_path):
        bed_path = tmp_path / "tss.bed"
        bed_path.write_text("chr1\t128\t144\nchr1\t512\t528\n")
        return str(bed_path)

    def test_len_tiled(self, bw_files):
        """Tiled dataset length = (chrom_length - window_size) // stride + 1."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=None,
            window_size=256,
            stride=256,
            output_stride=16,
        )
        # chr1 is 1024 bp; (1024 - 256) // 256 + 1 = 4 windows
        assert len(ds) == 4

    def test_item_shapes(self, bw_files, tss_bed):
        """Each item must return X of shape (2, window_size) and y of shape (1, window_size // output_stride)."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
        )
        X, y = ds[0]
        assert X.shape == (2, 256)
        assert y.shape == (1, 16)

    def test_item_dtype(self, bw_files, tss_bed):
        """Both X and y must be float32 tensors."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
        )
        X, y = ds[0]
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_signal_values(self, bw_files, tss_bed):
        """Both strands must carry the positive BigWig values (not zeros)."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
        )
        X, _ = ds[0]
        assert X[0].min() > 0  # plus strand all 5.0
        assert X[1].min() > 0  # minus strand abs-valued, all 3.0

    def test_label_marks_tss_bin(self, bw_files, tss_bed):
        """The output bin overlapping a TSS entry must be labelled 1."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
        )
        # Window 0 covers positions 0–255; TSS at 128–144 → bin 128//16=8
        X, y = ds[0]
        assert y[0, 8] == 1.0

    def test_label_outside_window_zero(self, bw_files, tss_bed):
        """Bins with no TSS in the window must be labelled 0."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
        )
        # Window 1 covers positions 256–511; no TSS in that range
        X, y = ds[1]
        assert y.sum() == 0.0

    def test_inference_mode(self, bw_files):
        """Without tss_beds, __getitem__ must return a bare tensor (no label)."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=None,
            window_size=256,
            stride=256,
            output_stride=16,
        )
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert not isinstance(item, tuple)

    def test_transform_applied(self, bw_files, tss_bed):
        """A transform zeroing all values must produce an all-zero X."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
            transform=lambda x: x * 0,
        )
        X, _ = ds[0]
        assert X.sum() == 0.0

    def test_rc_always(self, bw_files, tss_bed):
        """rc_prob=1.0 must swap strands and reverse both X and y along the length axis."""
        pl, mn = bw_files
        ds_orig = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
            rc_prob=0.0,
        )
        ds_rc = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
            rc_prob=1.0,
        )
        X_orig, y_orig = ds_orig[0]
        X_rc, y_rc = ds_rc[0]
        # Strands swapped and length reversed
        assert torch.allclose(X_rc[0], torch.flip(X_orig[1], dims=[0]))
        assert torch.allclose(X_rc[1], torch.flip(X_orig[0], dims=[0]))
        assert torch.allclose(y_rc, torch.flip(y_orig, dims=[1]))

    def test_tss_centered_len(self, bw_files, tss_bed):
        """tss_centered=True must produce one window per TSS entry in the BED file."""
        pl, mn = bw_files
        ds = NascentDataset(
            pl_bigwigs=[pl],
            mn_bigwigs=[mn],
            tss_beds=[tss_bed],
            window_size=256,
            stride=256,
            output_stride=16,
            tss_centered=True,
        )
        # BED has 2 TSS entries
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# NascentDataset — tss_centered validation (no BigWig needed)
# ---------------------------------------------------------------------------


class TestNascentDatasetTssCentered:
    def test_tss_centered_requires_tss_beds(self):
        """tss_centered=True without tss_beds must raise ValueError."""
        with pytest.raises(ValueError, match="tss_beds"):
            NascentDataset(
                pl_bigwigs=["fake.bw"],
                mn_bigwigs=["fake.bw"],
                tss_beds=None,
                tss_centered=True,
            )


# ---------------------------------------------------------------------------
# MixedBatchLoader
# ---------------------------------------------------------------------------


class TestMixedBatchLoader:
    @pytest.fixture
    def datasets(self, tmp_path):
        N, L, output_stride = 32, 128, 16
        paths_tiled = []
        paths_tss = []
        for name, paths in [("tiled", paths_tiled), ("tss", paths_tss)]:
            X = np.random.rand(N, 2, L).astype(np.float32)
            y = (np.random.rand(N, 1, L // output_stride) > 0.9).astype(np.float32)
            p = str(tmp_path / f"{name}.npz")
            np.savez(p, X=X, y=y)
            paths.append(p)
        tiled_ds = NascentDataset_([paths_tiled[0]], output_stride=output_stride)
        tss_ds = NascentDataset_([paths_tss[0]], output_stride=output_stride)
        return tiled_ds, tss_ds, L, output_stride

    @pytest.fixture
    def loader(self, datasets):
        tiled_ds, tss_ds, L, output_stride = datasets
        return MixedBatchLoader(tiled_ds, tss_ds, batch_size=8, tss_fraction=0.25)

    def test_sub_batch_sizes(self, loader):
        """tss_batch_size and tiled_batch_size must sum to batch_size and respect tss_fraction."""
        assert loader.tss_batch_size == 2
        assert loader.tiled_batch_size == 6
        assert loader.tss_batch_size + loader.tiled_batch_size == 8

    def test_batch_total_size(self, loader):
        """Every batch must contain exactly batch_size examples."""
        for X, y in loader:
            assert X.shape[0] == 8
            assert y.shape[0] == 8

    def test_shapes_consistent(self, datasets, loader):
        """X and y shapes must be consistent across all batches."""
        _, _, L, output_stride = datasets
        for X, y in loader:
            assert X.shape == (8, 2, L)
            assert y.shape == (8, 1, L // output_stride)

    def test_epoch_length(self, loader):
        """len(loader) must equal the actual number of batches yielded per epoch."""
        assert len(loader) == sum(1 for _ in loader)

    def test_invalid_fraction_raises(self, datasets):
        """tss_fraction of 0, 1, or outside (0, 1) must raise ValueError."""
        tiled_ds, tss_ds, _, _ = datasets
        for bad_frac in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                MixedBatchLoader(tiled_ds, tss_ds, batch_size=8, tss_fraction=bad_frac)
