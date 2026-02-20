"""
Download FineWeb-Edu pre-tokenized dataset.

This script downloads GPT-2 tokenized shards of FineWeb-Edu from HuggingFace.
Data is cached in ~/.cache/huggingface/hub/ (no duplicates across projects).

Usage:
    python path/to/01_download_dataset.py [num_shards]

    num_shards: Number of training shards to download (1-99).
                Each shard is ~100M tokens.
                Default: 99 (full 10B dataset)

Examples:
    # Download full dataset (~10B tokens)
    python path/to/01_download_dataset.py

    # Download 10 shards (~1B tokens) for testing
    python path/to/01_download_dataset.py 10

Requirements:
    - HF_TOKEN in .env file (optional, dataset is public)
"""

import sys

from research_lib.data.fineweb_datamodule import FineWebDataModule

# Ensure secrets are loaded (HF_TOKEN)
from research_lib.utils.secrets import check_auth


def main():

    # Check existance of local data
    check_auth()

    # Parse arguments
    num_shards = 99  # Full dataset by default
    if len(sys.argv) >= 2:
        num_shards = int(sys.argv[1])
        if not 1 <= num_shards <= 99:
            print("Error: num_shards must be between 1 and 99")
            sys.exit(1)

    print(f"FineWeb-Edu Download Script")
    print(f"=" * 40)
    print(f"Training shards: {num_shards}")
    print(f"Estimated size: ~{num_shards * 200}MB")  # ~200MB per shard
    print(f"Estimated tokens: ~{num_shards * 100}M")
    print()

    # Create datamodule and download
    dm = FineWebDataModule(
        num_train_shards=num_shards,
        seq_len=1024,  # Not used for download
    )

    dm.prepare_data()

    print()
    print("Download complete! Data cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
