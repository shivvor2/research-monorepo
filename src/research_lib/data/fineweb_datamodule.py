"""
Lightning DataModule for pre-tokenized FineWeb-Edu (10B) binary shards.

This module provides efficient data loading for the modded-nanogpt
binary token format, with support for:
    - Memory-mapped file access (no RAM explosion)
    - Multi-shard iteration
    - Distributed training (shard-based splitting)
    - Configurable sequence length and batch size
    - Automatic downloading from HuggingFace Hub

Example:
    from research_lib.data.fineweb_datamodule import FineWebDataModule

    dm = FineWebDataModule(
        num_train_shards=10,  # Use 10 shards (~1B tokens)
        seq_len=1024,
        batch_size=8,
        num_workers=4,
    )

    # Download data (only needed once, uses HF cache)
    dm.prepare_data()

    # Setup datasets
    dm.setup()

    for batch in dm.train_dataloader():
        input_ids = batch["input_ids"]  # (batch_size, seq_len)
        labels = batch["labels"]        # same, for causal LM
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional

import lightning as L
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, IterableDataset


def _load_shard_header(filepath: str) -> dict:
    """Load and validate shard header.

    The binary format uses a 256 int32 header:
        [0] = magic number (20240520)
        [1] = version (1)
        [2] = number of tokens

    Returns:
        Dict with 'magic', 'version', 'num_tokens'
    """
    header = np.fromfile(filepath, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"Invalid magic: {header[0]}"
    assert header[1] == 1, f"Invalid version: {header[1]}"
    return {
        "magic": header[0],
        "version": header[1],
        "num_tokens": header[2],
    }


def _load_shard_tokens(filepath: str) -> torch.Tensor:
    """Load tokens from a shard file.

    Tokens are stored as uint16 after the 1024-byte header.

    Returns:
        Tensor of shape (num_tokens,) with dtype int64
    """
    header = _load_shard_header(filepath)
    num_tokens = header["num_tokens"]

    # Tokens start at byte 256*4 = 1024 (after header)
    tokens = np.memmap(
        filepath,
        dtype=np.uint16,
        mode="r",
        offset=256 * 4,  # Skip header
        shape=(num_tokens,),
    )
    # Convert to torch tensor (copy to avoid memmap issues)
    return torch.from_numpy(tokens.astype(np.int64))


class ShardedTokenDataset(IterableDataset):
    """Iterable dataset over multiple token shards.

    Yields fixed-length sequences from concatenated shards.
    Handles distributed training by splitting shards across workers.
    """

    def __init__(
        self,
        shard_files: List[str],
        seq_len: int,
        shuffle_shards: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            shard_files: List of paths to .bin shard files
            seq_len: Length of sequences to yield
            shuffle_shards: Whether to shuffle shard order each epoch
            drop_last: Drop incomplete sequences at end of shard
        """
        self.shard_files = shard_files
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[dict]:
        """Yield sequences from shards."""
        # Get worker info for multi-process data loading
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading
            shard_files = self.shard_files
        else:
            # Multi-process: split shards across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_files = self.shard_files[worker_id::num_workers]

        # Shuffle shards if requested
        if self.shuffle_shards:
            indices = torch.randperm(len(shard_files)).tolist()
            shard_files = [shard_files[i] for i in indices]

        for shard_path in shard_files:
            tokens = _load_shard_tokens(shard_path)
            num_tokens = len(tokens)

            # +1 for labels (shifted by 1)
            effective_seq_len = self.seq_len + 1
            num_sequences = num_tokens // effective_seq_len

            if self.drop_last:
                # Trim to exact multiple
                tokens = tokens[: num_sequences * effective_seq_len]

            # Yield sequences
            for i in range(0, len(tokens) - effective_seq_len + 1, effective_seq_len):
                chunk = tokens[i : i + effective_seq_len]
                yield {
                    "input_ids": chunk[:-1],  # (seq_len,)
                    # "labels": chunk[1:],  # (seq_len,) shifted
                }


class FineWebDataModule(L.LightningDataModule):
    """Lightning DataModule for FineWeb-Edu (10B) pre-tokenized data.

    Downloads and loads the modded-nanogpt binary format shards from
    HuggingFace Hub. Data is cached in ~/.cache/huggingface/hub/.

    Shard naming convention:
        - finewebedu_val_000000.bin: Validation shard
        - finewebedu_train_000001.bin to _000099.bin: Training shards
        - Each shard contains ~100M tokens
    """

    # HuggingFace dataset source
    HF_REPO_ID: str = "kjj0/finewebedu10B-gpt2"
    HF_REPO_TYPE: str = "dataset"

    def __init__(
        self,
        num_train_shards: int = 99,
        seq_len: int = 1024,
        batch_size: int = 8,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
        pin_memory: bool = True,
    ):
        """
        Args:
            num_train_shards: Number of training shards to use (1-99).
                Each shard is ~100M tokens. 99 shards = full 10B dataset.
            seq_len: Sequence length for training (not including +1 for labels)
            batch_size: Training batch size per device
            num_workers: DataLoader workers per device
            val_batch_size: Validation batch size (defaults to batch_size)
            pin_memory: Pin memory for faster GPU transfer
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_train_shards = num_train_shards
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Will be populated in setup()
        self.train_shards: List[str] = []
        self.val_shards: List[str] = []

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment."""
        return os.getenv("HF_TOKEN")

    def _download_shard(self, filename: str) -> str:
        """Download a single shard and return its local path.

        Uses HuggingFace cache, so repeated calls are fast.

        Returns:
            Local path to the downloaded file
        """
        return hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=filename,
            repo_type=self.HF_REPO_TYPE,
            token=self._get_hf_token(),
        )

    def _resolve_shard_path(self, filename: str) -> str:
        """Resolve a shard filename to its cached local path.

        Raises if file is not already cached.

        Returns:
            Local path to the cached file
        """
        return hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=filename,
            repo_type=self.HF_REPO_TYPE,
            token=self._get_hf_token(),
            local_files_only=True,
        )

    def prepare_data(self) -> None:
        """Download shards from HuggingFace Hub.

        This method is called only on rank 0 in distributed training.
        Downloads are cached in ~/.cache/huggingface/hub/, so subsequent
        calls are fast no-ops.

        Downloads:
            - 1 validation shard (finewebedu_val_000000.bin)
            - num_train_shards training shards
        """
        print(f"Preparing FineWeb-Edu data ({self.num_train_shards} train shards)...")

        # Download validation shard
        print("Downloading validation shard...")
        self._download_shard("finewebedu_val_000000.bin")

        # Download training shards
        for i in range(1, self.num_train_shards + 1):
            filename = f"finewebedu_train_{i:06d}.bin"
            print(f"Downloading {filename} ({i}/{self.num_train_shards})...")
            self._download_shard(filename)

        print("Data preparation complete!")

    def setup(self, stage: Optional[str] = None) -> None:
        """Resolve shard paths from HuggingFace cache.

        This method is called on every process in distributed training.
        Assumes prepare_data() has already been called.

        Args:
            stage: Lightning stage ('fit', 'validate', 'test', 'predict')
        """
        # Resolve validation shard path
        self.val_shards = [self._resolve_shard_path("finewebedu_val_000000.bin")]

        # Resolve training shard paths
        self.train_shards = [
            self._resolve_shard_path(f"finewebedu_train_{i:06d}.bin")
            for i in range(1, self.num_train_shards + 1)
        ]

        # Log shard info
        print(f"Found {len(self.train_shards)} training shards")
        print(f"Found {len(self.val_shards)} validation shards")

        # Validate first shard
        header = _load_shard_header(self.train_shards[0])
        print(f"First shard has {header['num_tokens']:,} tokens")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = ShardedTokenDataset(
            shard_files=self.train_shards,
            seq_len=self.seq_len,
            shuffle_shards=True,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        dataset = ShardedTokenDataset(
            shard_files=self.val_shards,
            seq_len=self.seq_len,
            shuffle_shards=False,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def estimate_tokens_per_epoch(self) -> int:
        """Estimate total tokens in one training epoch.

        Requires setup() to have been called.
        """
        total = 0
        for shard in self.train_shards:
            header = _load_shard_header(shard)
            total += header["num_tokens"]
        return total
