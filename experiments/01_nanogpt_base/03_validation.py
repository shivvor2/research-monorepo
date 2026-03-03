"""
Validation script for NanoGPT - evaluates against the entire validation set.

Usage:
    # Validate the latest checkpoint
    python 03_validation.py

    # Validate a specific checkpoint
    python 03_validation.py checkpoint_path=/path/to/checkpoint.ckpt

    # Use test config
    python 03_validation.py --config-name=test
"""

from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import ModdedNanoGPT
from research_lib.data import FineWebDataModule
from research_lib.utils.secrets import check_auth

# =============================================================================
# Torch Flags
# =============================================================================

torch.set_float32_matmul_precision("high")


def get_state_file_path(experiment_name: str) -> Path:
    """Get path to the state file tracking the latest run for an experiment."""
    return Path(get_original_cwd()) / f".latest_run_{experiment_name}.txt"


def find_latest_checkpoint(cfg: DictConfig) -> Optional[str]:
    """
    Find the latest checkpoint for the experiment.

    Returns:
        Path to checkpoint or None if not found.
    """
    state_file = get_state_file_path(cfg.experiment.name)

    if not state_file.exists():
        # Check for completed run
        completed_file = state_file.with_suffix(".completed.txt")
        if completed_file.exists():
            state_file = completed_file
        else:
            return None

    try:
        rel_run_path = state_file.read_text().strip()
        last_run_dir = Path(get_original_cwd()) / rel_run_path
        ckpt_path = last_run_dir / cfg.checkpoint.dir / "last.ckpt"

        if ckpt_path.exists():
            return str(ckpt_path)
        return None
    except Exception:
        return None


def create_model(cfg: DictConfig) -> ModdedNanoGPT:
    """Create model from Hydra config."""
    model_config = NanoGPTConfig(
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        ff_dim=cfg.model.ff_dim,
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        padding_idx=cfg.model.padding_idx,
    )
    return ModdedNanoGPT(model_config)


def load_checkpoint(model: ModdedNanoGPT, ckpt_path: str) -> ModdedNanoGPT:
    """Load model weights from checkpoint.

    Note: Model should be compiled with torch.compile() BEFORE calling this
    function if the checkpoint was saved from a compiled model.
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning stores model state under "state_dict" with "model." prefix
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix (from DualOptimizerModule wrapping)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[6:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    print("Checkpoint loaded successfully!")
    return model


@torch.no_grad()
def validate(
    model: ModdedNanoGPT,
    dataloader,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """
    Run validation over the entire dataset.

    Returns:
        Dictionary with validation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    dtype = torch.bfloat16 if use_amp else torch.float32

    pbar = tqdm(dataloader, desc="Validating", unit="batch")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)

        # Create labels (shifted by 1)
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            logits = model(inputs)

            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

        batch_tokens = labels.numel()
        total_loss += loss.item()
        total_tokens += batch_tokens
        num_batches += 1

        # Update progress bar
        running_loss = total_loss / total_tokens
        running_ppl = torch.exp(torch.tensor(running_loss)).item()
        pbar.set_postfix(
            {
                "loss": f"{running_loss:.4f}",
                "ppl": f"{running_ppl:.2f}",
            }
        )

    # Compute final metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main validation function."""

    print("=" * 80)
    check_auth()
    print("=" * 80)

    # Print config
    print("Validation Configuration:")
    print("=" * 80)
    print(f"Model: {cfg.model.n_layer}L, {cfg.model.n_embd}D, {cfg.model.n_head}H")
    print(f"Vocab size: {cfg.model.vocab_size}")
    print(f"Block size: {cfg.model.block_size}")
    print("=" * 80)

    # Resolve checkpoint path
    ckpt_path = cfg.get("checkpoint_path", None)
    if ckpt_path is None:
        print("No checkpoint_path provided, searching for latest...")
        ckpt_path = find_latest_checkpoint(cfg)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for experiment '{cfg.experiment.name}'. "
                "Provide checkpoint_path explicitly or run training first."
            )
    else:
        ckpt_path = to_absolute_path(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Using checkpoint: {ckpt_path}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\n[1/3] Creating model...")
    model = create_model(cfg)

    # Compile BEFORE loading (to match training setup)
    if device.type == "cuda":
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    # Now load checkpoint (keys will match)
    model = load_checkpoint(model, ckpt_path)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Create datamodule
    print("\n[2/3] Setting up data...")
    datamodule = FineWebDataModule(
        num_train_shards=1,  # Not used for validation
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="validate")

    val_dataloader = datamodule.val_dataloader()

    # Run validation
    print("\n[3/3] Running validation on full validation set...")
    print("-" * 40)

    use_amp = "bf16" in cfg.training.precision or "16" in cfg.training.precision
    metrics = validate(model, val_dataloader, device, use_amp=use_amp)

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"  Loss:       {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Tokens:     {metrics['total_tokens']:,}")
    print(f"  Batches:    {metrics['num_batches']:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
