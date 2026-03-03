"""
Tests for dataset.py — NascentDataset_ (npz-backed) and NascentDataset (bigwig-backed).

NascentDataset tests that require actual BigWig files are marked with
@pytest.mark.integration and skipped by default.  Run them with:
    pytest -m integration
"""

import numpy as np
import pytest
import torch

from burninate.dataset import MixedBatchLoader, NascentDataset, NascentDataset_


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
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        assert len(ds) == N

    def test_multiple_files(self, tmp_path):
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
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        X_i, y_i = ds[0]
        assert X_i.shape == (2, L), f"X shape: {X_i.shape}"
        assert y_i.shape == (1, L // output_stride), f"y shape: {y_i.shape}"

    def test_item_dtype(self, npz_file):
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        X_i, y_i = ds[0]
        assert X_i.dtype == torch.float32
        assert y_i.dtype == torch.float32

    def test_transform_applied(self, npz_file):
        path, N, L, output_stride = npz_file

        def double(x):
            return x * 2.0

        ds = NascentDataset_([path], output_stride=output_stride, transform=double)
        X_orig, _ = NascentDataset_([path], output_stride=output_stride)[0]
        X_trans, _ = ds[0]
        assert torch.allclose(X_trans, X_orig * 2.0)

    def test_label_values_in_range(self, npz_file):
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        for i in range(len(ds)):
            _, y_i = ds[i]
            assert y_i.min() >= 0.0
            assert y_i.max() <= 1.0

    def test_dataloader_compatible(self, npz_file):
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape == (4, 2, L)
        assert y_batch.shape == (4, 1, L // output_stride)

    def test_last_index(self, npz_file):
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride)
        X_i, y_i = ds[N - 1]
        assert X_i.shape == (2, L)

    def test_empty_file_list_raises(self):
        with pytest.raises(Exception):
            NascentDataset_([])

    def test_rc_prob_zero_unchanged(self, npz_file):
        path, N, L, output_stride = npz_file
        ds_no_rc = NascentDataset_([path], output_stride=output_stride, rc_prob=0.0)
        ds_rc = NascentDataset_([path], output_stride=output_stride, rc_prob=0.0)
        for i in range(N):
            X_no_rc, y_no_rc = ds_no_rc[i]
            X_rc, y_rc = ds_rc[i]
            assert torch.allclose(X_no_rc, X_rc)
            assert torch.allclose(y_no_rc, y_rc)

    def test_rc_always_applied(self, npz_file):
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
        path, N, L, output_stride = npz_file
        ds = NascentDataset_([path], output_stride=output_stride, rc_prob=1.0)
        X_rc, y_rc = ds[0]
        assert X_rc.shape == (2, L)
        assert y_rc.shape == (1, L // output_stride)


# ---------------------------------------------------------------------------
# NascentDataset (bigwig-backed) — integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNascentDataset:
    """
    These tests require actual BigWig and BED files.
    Skipped unless --run-integration is passed or pytest -m integration used.
    """

    def test_placeholder(self):
        pytest.skip("Integration tests require real BigWig files.")


# ---------------------------------------------------------------------------
# NascentDataset — tss_centered validation (no BigWig needed)
# ---------------------------------------------------------------------------

class TestNascentDatasetTssCentered:
    def test_tss_centered_requires_tss_beds(self):
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
        return MixedBatchLoader(
            tiled_ds, tss_ds, batch_size=8, tss_fraction=0.25
        )

    def test_sub_batch_sizes(self, loader):
        assert loader.tss_batch_size == 2
        assert loader.tiled_batch_size == 6
        assert loader.tss_batch_size + loader.tiled_batch_size == 8

    def test_batch_total_size(self, loader):
        for X, y in loader:
            assert X.shape[0] == 8
            assert y.shape[0] == 8

    def test_shapes_consistent(self, datasets, loader):
        _, _, L, output_stride = datasets
        for X, y in loader:
            assert X.shape == (8, 2, L)
            assert y.shape == (8, 1, L // output_stride)

    def test_epoch_length(self, loader):
        assert len(loader) == sum(1 for _ in loader)

    def test_invalid_fraction_raises(self, datasets):
        tiled_ds, tss_ds, _, _ = datasets
        for bad_frac in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                MixedBatchLoader(tiled_ds, tss_ds, batch_size=8, tss_fraction=bad_frac)
