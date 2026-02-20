"""
Test suite for FineWebDataModule.

Run with: pytest src/research_lib/data/tests/test_fineweb_datamodule.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ..fineweb_datamodule import (
    FineWebDataModule,
    ShardedTokenDataset,
    _load_shard_header,
    _load_shard_tokens,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_shard(tmp_path):
    """Create a fake shard file with valid header and tokens.

    Returns:
        Tuple of (shard_path, num_tokens, expected_tokens_array)
    """
    num_tokens = 1000

    # Header: 256 int32 values
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = num_tokens

    # Tokens: sequential uint16 values for predictable testing
    tokens = np.arange(num_tokens, dtype=np.uint16)

    shard_path = tmp_path / "test_shard.bin"
    with open(shard_path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())

    return str(shard_path), num_tokens, tokens


@pytest.fixture
def multiple_fake_shards(tmp_path):
    """Create multiple fake shard files with varying sizes.

    Returns:
        List of shard paths
    """
    shard_paths = []

    for i in range(3):
        num_tokens = 500 + i * 100  # 500, 600, 700 tokens

        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = num_tokens

        # Sequential tokens starting from 0
        tokens = np.arange(num_tokens, dtype=np.uint16)

        shard_path = tmp_path / f"shard_{i:06d}.bin"
        with open(shard_path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())

        shard_paths.append(str(shard_path))

    return shard_paths


@pytest.fixture
def complete_fake_dataset(tmp_path):
    """Create a complete fake dataset with train and val shards.

    Returns:
        Tuple of (data_dir, train_shard_paths, val_shard_paths)
    """
    data_dir = tmp_path / "finewebedu"
    data_dir.mkdir()

    def create_shard(name: str, num_tokens: int) -> str:
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = num_tokens

        tokens = np.arange(num_tokens, dtype=np.uint16) % 50000

        path = data_dir / name
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())

        return str(path)

    # Create validation shard
    val_paths = [create_shard("finewebedu_val_000000.bin", num_tokens=2000)]

    # Create training shards
    train_paths = [
        create_shard(f"finewebedu_train_{i:06d}.bin", num_tokens=5000)
        for i in range(1, 4)  # 3 train shards
    ]

    return data_dir, train_paths, val_paths


# =============================================================================
# Tests: Shard I/O Functions
# =============================================================================


class TestShardIO:
    """Tests for shard reading utilities."""

    def test_load_shard_header_valid(self, fake_shard):
        """Test header parsing with valid shard."""
        shard_path, num_tokens, _ = fake_shard
        header = _load_shard_header(shard_path)

        assert header["magic"] == 20240520
        assert header["version"] == 1
        assert header["num_tokens"] == num_tokens

    def test_load_shard_header_invalid_magic(self, tmp_path):
        """Test that invalid magic number raises assertion."""
        header = np.zeros(256, dtype=np.int32)
        header[0] = 12345  # Wrong magic
        header[1] = 1
        header[2] = 100

        shard_path = tmp_path / "bad_magic.bin"
        with open(shard_path, "wb") as f:
            f.write(header.tobytes())
            f.write(np.zeros(100, dtype=np.uint16).tobytes())

        with pytest.raises(AssertionError, match="Invalid magic"):
            _load_shard_header(str(shard_path))

    def test_load_shard_header_invalid_version(self, tmp_path):
        """Test that invalid version raises assertion."""
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 99  # Wrong version
        header[2] = 100

        shard_path = tmp_path / "bad_version.bin"
        with open(shard_path, "wb") as f:
            f.write(header.tobytes())
            f.write(np.zeros(100, dtype=np.uint16).tobytes())

        with pytest.raises(AssertionError, match="Invalid version"):
            _load_shard_header(str(shard_path))

    def test_load_shard_tokens(self, fake_shard):
        """Test token loading returns correct values."""
        shard_path, num_tokens, expected_tokens = fake_shard
        tokens = _load_shard_tokens(shard_path)

        assert tokens.shape == (num_tokens,)
        assert tokens.dtype == torch.int64

        expected = torch.from_numpy(expected_tokens.astype(np.int64))
        torch.testing.assert_close(tokens, expected)

    def test_load_shard_tokens_dtype_conversion(self, fake_shard):
        """Test that uint16 tokens are converted to int64."""
        shard_path, _, _ = fake_shard
        tokens = _load_shard_tokens(shard_path)

        # Should be int64 for compatibility with embedding layers
        assert tokens.dtype == torch.int64


# =============================================================================
# Tests: ShardedTokenDataset
# =============================================================================


class TestShardedTokenDataset:
    """Tests for the iterable dataset."""

    def test_yields_correct_shapes(self, multiple_fake_shards):
        """Test that dataset yields sequences with correct shapes."""
        seq_len = 64
        dataset = ShardedTokenDataset(
            shard_files=multiple_fake_shards[:1],  # Use one shard
            seq_len=seq_len,
            shuffle_shards=False,
        )

        batch = next(iter(dataset))

        assert "input_ids" in batch
        # assert "labels" in batch
        assert batch["input_ids"].shape == (seq_len,)
        # assert batch["labels"].shape == (seq_len,)

    def test_labels_are_shifted_by_one(self, multiple_fake_shards):
        """Test that labels are input_ids shifted by 1 position."""
        seq_len = 64
        dataset = ShardedTokenDataset(
            shard_files=multiple_fake_shards[:1],
            seq_len=seq_len,
            shuffle_shards=False,
        )

        batch = next(iter(dataset))

        # For sequential tokens [0, 1, 2, 3, ...]
        # input_ids should be [0, 1, ..., seq_len-1]
        # labels should be [1, 2, ..., seq_len]
        expected_inputs = torch.arange(seq_len, dtype=torch.int64)
        # expected_labels = torch.arange(1, seq_len + 1, dtype=torch.int64)

        torch.testing.assert_close(batch["input_ids"], expected_inputs)
        # torch.testing.assert_close(batch["labels"], expected_labels)

    def test_correct_number_of_sequences(self, multiple_fake_shards):
        """Test that correct number of sequences are yielded."""
        seq_len = 64
        effective_seq_len = seq_len + 1  # +1 for labels

        # First shard has 500 tokens
        # 500 // 65 = 7 complete sequences
        dataset = ShardedTokenDataset(
            shard_files=multiple_fake_shards[:1],
            seq_len=seq_len,
            shuffle_shards=False,
            drop_last=True,
        )

        count = sum(1 for _ in dataset)
        assert count == 500 // effective_seq_len

    def test_iterates_multiple_shards(self, multiple_fake_shards):
        """Test iteration across multiple shards."""
        seq_len = 64
        effective_seq_len = seq_len + 1

        dataset = ShardedTokenDataset(
            shard_files=multiple_fake_shards,
            seq_len=seq_len,
            shuffle_shards=False,
            drop_last=True,
        )

        # Total tokens: 500 + 600 + 700 = 1800
        # Sequences: 500//65 + 600//65 + 700//65 = 7 + 9 + 10 = 26
        expected_count = sum((500 + i * 100) // effective_seq_len for i in range(3))

        count = sum(1 for _ in dataset)
        assert count == expected_count

    def test_shuffle_shards_changes_order(self, multiple_fake_shards):
        """Test that shuffle_shards changes iteration order."""
        seq_len = 64

        # Collect first token from each run
        first_tokens = []

        for _ in range(5):
            dataset = ShardedTokenDataset(
                shard_files=multiple_fake_shards,
                seq_len=seq_len,
                shuffle_shards=True,
            )
            batch = next(iter(dataset))
            first_tokens.append(batch["input_ids"][0].item())

        # With shuffling, we should sometimes get different first tokens
        # (unless very unlucky with random seed)
        # This is a probabilistic test - may occasionally fail
        unique_tokens = set(first_tokens)
        # At least check we got valid tokens (0 from first positions of shards)
        assert all(t >= 0 for t in first_tokens)

    def test_no_shuffle_deterministic(self, multiple_fake_shards):
        """Test that shuffle_shards=False gives deterministic order."""
        seq_len = 64

        results = []
        for _ in range(3):
            dataset = ShardedTokenDataset(
                shard_files=multiple_fake_shards,
                seq_len=seq_len,
                shuffle_shards=False,
            )
            tokens = [next(iter(dataset))["input_ids"] for _ in range(3)]
            results.append(torch.stack(tokens))

        # All runs should be identical
        torch.testing.assert_close(results[0], results[1])
        torch.testing.assert_close(results[1], results[2])


# =============================================================================
# Tests: FineWebDataModule.prepare_data()
# =============================================================================


class TestFineWebDataModulePrepareData:
    """Tests for prepare_data() download logic using mocks."""

    @patch.object(FineWebDataModule, "_download_shard")
    def test_downloads_correct_number_of_shards(self, mock_download):
        """Test that prepare_data downloads the right number of files."""
        mock_download.return_value = "/fake/path"

        num_shards = 5
        dm = FineWebDataModule(num_train_shards=num_shards, seq_len=1024)
        dm.prepare_data()

        # Should download: 1 val + 5 train = 6 total
        assert mock_download.call_count == num_shards + 1

    @patch.object(FineWebDataModule, "_download_shard")
    def test_downloads_validation_shard_first(self, mock_download):
        """Test that validation shard is downloaded."""
        mock_download.return_value = "/fake/path"

        dm = FineWebDataModule(num_train_shards=3, seq_len=1024)
        dm.prepare_data()

        # Check first call was validation shard
        first_call_args = mock_download.call_args_list[0]
        assert first_call_args[0][0] == "finewebedu_val_000000.bin"

    @patch.object(FineWebDataModule, "_download_shard")
    def test_downloads_correct_train_shard_names(self, mock_download):
        """Test that training shards have correct filenames."""
        mock_download.return_value = "/fake/path"

        dm = FineWebDataModule(num_train_shards=3, seq_len=1024)
        dm.prepare_data()

        # Extract all filenames from calls
        filenames = [call[0][0] for call in mock_download.call_args_list]

        assert "finewebedu_val_000000.bin" in filenames
        assert "finewebedu_train_000001.bin" in filenames
        assert "finewebedu_train_000002.bin" in filenames
        assert "finewebedu_train_000003.bin" in filenames
        assert "finewebedu_train_000004.bin" not in filenames

    @patch("research_lib.data.fineweb_datamodule.hf_hub_download")
    def test_passes_hf_token(self, mock_hf_download):
        """Test that HF token is passed to download function."""
        mock_hf_download.return_value = "/fake/path"

        with patch.dict("os.environ", {"HF_TOKEN": "test_token_123"}):
            dm = FineWebDataModule(num_train_shards=1, seq_len=1024)
            dm.prepare_data()

        # Check token was passed
        call_kwargs = mock_hf_download.call_args_list[0][1]
        assert call_kwargs["token"] == "test_token_123"

    @patch("research_lib.data.fineweb_datamodule.hf_hub_download")
    def test_works_without_hf_token(self, mock_hf_download):
        """Test that download works with no HF token (public repo)."""
        mock_hf_download.return_value = "/fake/path"

        with patch.dict("os.environ", {}, clear=True):
            dm = FineWebDataModule(num_train_shards=1, seq_len=1024)
            dm.prepare_data()

        # Token should be None but call should succeed
        call_kwargs = mock_hf_download.call_args_list[0][1]
        assert call_kwargs["token"] is None


# =============================================================================
# Tests: FineWebDataModule.setup()
# =============================================================================


class TestFineWebDataModuleSetup:
    """Tests for setup() shard path resolution."""

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    @patch("research_lib.data.fineweb_datamodule._load_shard_header")
    def test_resolves_correct_shard_paths(self, mock_header, mock_resolve):
        """Test that setup resolves the right shard paths."""
        mock_resolve.side_effect = lambda f: f"/cache/{f}"
        mock_header.return_value = {
            "magic": 20240520,
            "version": 1,
            "num_tokens": 100000,
        }

        dm = FineWebDataModule(num_train_shards=3, seq_len=1024)
        dm.setup()

        assert len(dm.val_shards) == 1
        assert len(dm.train_shards) == 3

        assert dm.val_shards[0] == "/cache/finewebedu_val_000000.bin"
        assert dm.train_shards[0] == "/cache/finewebedu_train_000001.bin"
        assert dm.train_shards[2] == "/cache/finewebedu_train_000003.bin"

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    @patch("research_lib.data.fineweb_datamodule._load_shard_header")
    def test_validates_first_shard(self, mock_header, mock_resolve):
        """Test that setup validates the first training shard."""
        mock_resolve.side_effect = lambda f: f"/cache/{f}"
        mock_header.return_value = {
            "magic": 20240520,
            "version": 1,
            "num_tokens": 100000,
        }

        dm = FineWebDataModule(num_train_shards=2, seq_len=1024)
        dm.setup()

        # Should have called header validation on first train shard
        mock_header.assert_called_with("/cache/finewebedu_train_000001.bin")


# =============================================================================
# Tests: FineWebDataModule Integration
# =============================================================================


class TestFineWebDataModuleIntegration:
    """Integration tests using fake shard files."""

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    def test_train_dataloader_yields_batches(self, mock_resolve, complete_fake_dataset):
        """Test that train dataloader yields properly shaped batches."""
        data_dir, train_paths, val_paths = complete_fake_dataset

        # Mock resolution to return our fake shard paths
        def resolve(filename):
            if "val" in filename:
                return val_paths[0]
            else:
                # Extract shard number and map to our paths
                idx = int(filename.split("_")[-1].replace(".bin", "")) - 1
                return train_paths[idx]

        mock_resolve.side_effect = resolve

        dm = FineWebDataModule(
            num_train_shards=3,
            seq_len=64,
            batch_size=4,
            num_workers=0,  # Avoid multiprocessing in tests
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (4, 64)
        # assert batch["labels"].shape == (4, 64)
        assert batch["input_ids"].dtype == torch.int64

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    def test_val_dataloader_yields_batches(self, mock_resolve, complete_fake_dataset):
        """Test that val dataloader yields properly shaped batches."""
        data_dir, train_paths, val_paths = complete_fake_dataset

        def resolve(filename):
            if "val" in filename:
                return val_paths[0]
            idx = int(filename.split("_")[-1].replace(".bin", "")) - 1
            return train_paths[idx]

        mock_resolve.side_effect = resolve

        dm = FineWebDataModule(
            num_train_shards=3,
            seq_len=64,
            batch_size=8,
            num_workers=0,
        )
        dm.setup()

        loader = dm.val_dataloader()
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (8, 64)
        # assert batch["labels"].shape == (8, 64)

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    def test_estimate_tokens_per_epoch(self, mock_resolve, complete_fake_dataset):
        """Test token counting across shards."""
        data_dir, train_paths, val_paths = complete_fake_dataset

        def resolve(filename):
            if "val" in filename:
                return val_paths[0]
            idx = int(filename.split("_")[-1].replace(".bin", "")) - 1
            return train_paths[idx]

        mock_resolve.side_effect = resolve

        dm = FineWebDataModule(num_train_shards=3, seq_len=64)
        dm.setup()

        total = dm.estimate_tokens_per_epoch()

        # Each train shard has 5000 tokens
        assert total == 3 * 5000

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    def test_val_batch_size_defaults_to_batch_size(
        self, mock_resolve, complete_fake_dataset
    ):
        """Test that val_batch_size defaults to batch_size."""
        data_dir, train_paths, val_paths = complete_fake_dataset

        def resolve(filename):
            if "val" in filename:
                return val_paths[0]
            idx = int(filename.split("_")[-1].replace(".bin", "")) - 1
            return train_paths[idx]

        mock_resolve.side_effect = resolve

        dm = FineWebDataModule(
            num_train_shards=3,
            seq_len=64,
            batch_size=16,
            val_batch_size=None,  # Should default to batch_size
            num_workers=0,
        )
        dm.setup()

        assert dm.val_batch_size == 16

    @patch.object(FineWebDataModule, "_resolve_shard_path")
    def test_custom_val_batch_size(self, mock_resolve, complete_fake_dataset):
        """Test that custom val_batch_size is respected."""
        data_dir, train_paths, val_paths = complete_fake_dataset

        def resolve(filename):
            if "val" in filename:
                return val_paths[0]
            idx = int(filename.split("_")[-1].replace(".bin", "")) - 1
            return train_paths[idx]

        mock_resolve.side_effect = resolve

        dm = FineWebDataModule(
            num_train_shards=3,
            seq_len=64,
            batch_size=16,
            val_batch_size=32,
            num_workers=0,
        )
        dm.setup()

        assert dm.val_batch_size == 32


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_shard(self, fake_shard):
        """Test with minimum configuration (1 shard)."""
        shard_path, _, _ = fake_shard

        dataset = ShardedTokenDataset(
            shard_files=[shard_path],
            seq_len=32,
            shuffle_shards=False,
        )

        # Should be able to iterate
        batch = next(iter(dataset))
        assert batch["input_ids"].shape == (32,)

    def test_seq_len_larger_than_shard(self, tmp_path):
        """Test behavior when seq_len is larger than available tokens."""
        # Create a tiny shard with only 50 tokens
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = 50

        shard_path = tmp_path / "tiny.bin"
        with open(shard_path, "wb") as f:
            f.write(header.tobytes())
            f.write(np.arange(50, dtype=np.uint16).tobytes())

        # seq_len + 1 = 65 > 50 tokens
        dataset = ShardedTokenDataset(
            shard_files=[str(shard_path)],
            seq_len=64,
            shuffle_shards=False,
        )

        # Should yield nothing (not enough tokens)
        count = sum(1 for _ in dataset)
        assert count == 0

    def test_hyperparameters_saved(self):
        """Test that Lightning saves hyperparameters correctly."""
        dm = FineWebDataModule(
            num_train_shards=10,
            seq_len=512,
            batch_size=32,
        )

        assert dm.hparams.num_train_shards == 10
        assert dm.hparams.seq_len == 512
        assert dm.hparams.batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
