"""
Interactive demo for NanoGPT text generation using Gradio.

Usage:
    # Run with latest checkpoint
    python 04_demo.py

    # Run with specific checkpoint
    python 04_demo.py checkpoint_path=/path/to/checkpoint.ckpt

    # Custom port
    python 04_demo.py server_port=7861

Note:
    The model was trained on GPT-2 tokenized data with vocab_size padded to 50304.
    The extra tokens (50257-50303) were never seen during training, so the model
    naturally learns to not output them. We use the GPT-2 tokenizer for encoding
    and decoding.
"""

from pathlib import Path
from typing import Generator, Optional

import gradio as gr
import hydra
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
from transformers import GPT2TokenizerFast

from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import ModdedNanoGPT
from research_lib.utils.secrets import check_auth

# =============================================================================
# Torch Flags
# =============================================================================

torch.set_float32_matmul_precision("high")

# =============================================================================
# Global state for Gradio
# =============================================================================

_MODEL: Optional[ModdedNanoGPT] = None
_TOKENIZER: Optional[GPT2TokenizerFast] = None
_DEVICE: Optional[torch.device] = None
_BLOCK_SIZE: int = 1024


def get_state_file_path(experiment_name: str) -> Path:
    """Get path to the state file tracking the latest run for an experiment."""
    return Path(get_original_cwd()) / f".latest_run_{experiment_name}.txt"


def find_latest_checkpoint(cfg: DictConfig) -> Optional[str]:
    """Find the latest checkpoint for the experiment."""
    state_file = get_state_file_path(cfg.experiment.name)

    if not state_file.exists():
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
def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    """
    Generate text from a prompt with streaming output.

    Supports generating beyond the context window by using a rolling context -
    only the most recent `block_size` tokens are fed to the model, but all
    generated tokens are tracked for output.

    Args:
        prompt: Input text to continue from.
        max_new_tokens: Maximum number of tokens to generate.
            Set to 0 for unlimited (generates until EOS).
        temperature: Sampling temperature (higher = more random).
        top_k: Number of top tokens to consider for sampling.
        top_p: Nucleus sampling probability threshold.

    Yields:
        Partial generated text as each token is produced.
    """
    global _MODEL, _TOKENIZER, _DEVICE, _BLOCK_SIZE

    if _MODEL is None or _TOKENIZER is None:
        yield "Error: Model not loaded. Please restart the demo."
        return

    # Encode prompt
    encoded = _TOKENIZER(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(_DEVICE)

    # Track all generated token IDs (including prompt)
    all_token_ids = input_ids[0].tolist()

    # If prompt exceeds block size, we still keep all tokens for output
    # but only use the last block_size for model input

    _MODEL.eval()

    # Yield initial prompt
    yield prompt

    # Generate tokens
    # If max_new_tokens is 0, generate indefinitely until EOS
    token_count = 0
    max_iterations = max_new_tokens if max_new_tokens > 0 else float("inf")

    while token_count < max_iterations:
        # Get the context window (last block_size tokens)
        if len(all_token_ids) <= _BLOCK_SIZE:
            context_ids = all_token_ids
        else:
            # Rolling context: only use the most recent block_size tokens
            context_ids = all_token_ids[-_BLOCK_SIZE:]

        context = torch.tensor([context_ids], dtype=torch.long, device=_DEVICE)

        # Forward pass
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=_DEVICE.type == "cuda"
        ):
            logits = _MODEL(context)

        # Get logits for the last token
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)

        if temperature == 0:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1).item()
        else:
            next_token = torch.multinomial(probs, num_samples=1).item()

        # Check for EOS token
        if next_token == _TOKENIZER.eos_token_id:
            break

        # Append to all tokens
        all_token_ids.append(next_token)
        token_count += 1

        # Decode and yield full sequence
        output_text = _TOKENIZER.decode(all_token_ids)
        yield output_text


def create_gradio_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="NanoGPT Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 NanoGPT Text Generation Demo

            This is a 124M parameter GPT-2 style model trained on FineWeb-Edu (10B tokens).
            Enter a prompt and watch the model generate text!

            **Note:** The model was trained on educational web content, so it works best
            with informational or educational prompts.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                    value="The theory of relativity, proposed by Albert Einstein,",
                )

                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")

            with gr.Column(scale=1):
                max_tokens = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=256,
                    step=16,
                    label="Max New Tokens",
                    info="0 = unlimited (until EOS)",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random, 0 = greedy",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-K",
                    info="0 = disabled",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-P (Nucleus)",
                    info="1.0 = disabled",
                )

        output_text = gr.Textbox(
            label="Generated Text",
            lines=15,
            interactive=False,
        )

        # Example prompts
        gr.Examples(
            examples=[
                ["The theory of relativity, proposed by Albert Einstein,"],
                ["In a groundbreaking study, researchers have discovered that"],
                ["The solar system consists of"],
                ["Machine learning is a subset of artificial intelligence that"],
                ["The French Revolution began in 1789 when"],
                ["Photosynthesis is the process by which plants"],
            ],
            inputs=prompt_input,
            label="Example Prompts",
        )

        # Event handlers
        generate_btn.click(
            fn=generate,
            inputs=[prompt_input, max_tokens, temperature, top_k, top_p],
            outputs=output_text,
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[prompt_input, output_text],
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Set **Max New Tokens to 0** to generate until the model outputs an end-of-text token
            - Lower temperature (0.3-0.7) for more focused, factual text
            - Higher temperature (0.8-1.2) for more creative, varied text
            - Top-K limits vocabulary to K most likely tokens
            - Top-P (nucleus sampling) keeps tokens until cumulative probability reaches P
            - Generation can exceed context length (1024 tokens) - the model uses a rolling window
            """
        )

    return demo


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main demo function."""
    global _MODEL, _TOKENIZER, _DEVICE, _BLOCK_SIZE

    print("=" * 80)
    check_auth()
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
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_DEVICE}")

    # Store block size
    _BLOCK_SIZE = cfg.model.block_size

    # Load tokenizer (GPT-2)
    print("\n[1/3] Loading tokenizer...")
    _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Tokenizer loaded: vocab_size={_TOKENIZER.vocab_size}")

    # Create and load model
    print("\n[2/3] Loading model...")
    _MODEL = create_model(cfg)

    # NOTE: We intentionally do NOT compile the model for inference.
    # The rotary embedding's cached_freqs_seq_len changes with each sequence length,
    # causing torch.compile to recompile repeatedly. For interactive demos with
    # variable-length generation, eager mode is more practical.
    #
    # However, the checkpoint was saved with a compiled model (model._orig_mod.*),
    # so we still need to compile before loading to match the state dict keys.
    if _DEVICE.type == "cuda":
        print("Compiling model (for checkpoint compatibility)...")
        _MODEL = torch.compile(_MODEL, mode="reduce-overhead")

    _MODEL = load_checkpoint(_MODEL, ckpt_path)
    _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()

    num_params = sum(p.numel() for p in _MODEL.parameters())
    print(f"Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Create and launch Gradio interface
    print("\n[3/3] Starting Gradio interface...")
    demo = create_gradio_interface()

    # Get server settings from config
    server_port = cfg.get("server_port", 7860)
    server_name = cfg.get("server_name", "0.0.0.0")
    share = cfg.get("share", False)

    print(f"\nLaunching demo on http://{server_name}:{server_port}")
    print("Press Ctrl+C to stop the server.\n")

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    main()
