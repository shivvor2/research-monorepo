# Experiment 01: Modded NanoGPT Baseline
<!-- Yes, the readme is ai-genned, I ain't typing all of that -->

**Goal:** Train a 124M param (GPT-2 Small) model on FineWeb-Edu 10B, with reference to [`KellerJordan/modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt)

## Quick Start

### 1. Run Training (Production)
```bash
python 02_pretrain.py
```
*   **Auto-Resume:** By default, looks for a previous run in this experiment family (`nanogpt_base`) and resumes if a compatible checkpoint exists.
*   **Seed:** Fixed to `seed=null` (Non-deterministic), set to integer (e.g. `seed = 67`) for deterministic behavior

### 2. Sanity Check (Test Config)
```bash
python 02_pretrain.py --config-name=test
```
*   Runs for 100 steps with small batch sizes.
*   Uses a separate experiment history (`nanogpt_base_test`) so it doesn't mess up your main run.
*   Disables WandB logging.

### 3. Common Overrides
```bash
# Change max steps
python 02_pretrain.py training.max_steps=1000

# Force a fresh run (ignore previous checkpoints)
python 02_pretrain.py checkpoint.resume_from=null

# Change physical batch size (if OOM)
python 02_pretrain.py data.batch_size=2
```

---

## Directory Structure

```text
experiments/01_nanogpt_base/
├── conf/
│   ├── config.yaml       # MAIN config (Model, Optimizer, Schedule, Training)
│   └── test.yaml         # FAST config (Small steps, no WandB)
├── 02_pretrain.py        # Main training script (Hydra + Lightning)
├── context_state_files/  # (Implicit) .latest_run_*.txt files appear in project root
└── outputs/              # Hydra output (Logs, CSVs, Checkpoints)
    └── 2026-02-18/
        └── 12-00-00/
            └── checkpoints/  # Actual model weights
```

---

## Core Concepts

### 1. Dual Optimizers
The model parameters are split into two groups based on the `VECTOR_TARGET_MODULES` list in `02_pretrain.py`:
*   **Matrix Optimizer (NorMuon)**: Attention projections (Q,K,V,O) and MLP weights.
    *   *Config:* `optimizer.matrix`
*   **Vector Optimizer (Cautious AdamW)**: Embeddings, Head, and RMSNorms.
    *   *Config:* `optimizer.vector`

### 2. The Auto-Resume Logic
This script uses a **stateful resume mechanism** to handle Hydra directory structures:

1.  **State File:** A hidden file `.latest_run_<experiment.name>.txt` is created in the project root.
2.  **Tracking:** It points to the directory of the *most recent* successful checkpoint save.
3.  **Behavior:**
    *   If `checkpoint.resume_from: auto` (default), it reads this file.
    *   It checks for `checkpoints/last.ckpt`.
    *   It performs a **strict shape compatibility check**.
    *   If compatible, it resumes. If not, it starts fresh.

**To reset history manually:** delete the `.latest_run_nanogpt_base.txt` file in the root directory.

### 3. Batch Size Strategy
*   **Target:** The reference run uses ~130k tokens per step (Phase 1).
*   **Current Config:** `batch_size=4` * `grad_accum=16` * `seq_len=1024` = **65,536 tokens**.
*   This is ~0.5x the reference. To match reference exactly, increase `grad_accum` to 32.

---

## Outputs & Logging

*   **WandB:** Enabled by default (Project: `nanogpt-baseline`). Disable via `logging.wandb.enabled=false`.
*   **CSV:** Always enabled. Saved to `outputs/<date>/<time>/logs/csv/`.
*   **Checkpoints:** Saved to `outputs/<date>/<time>/checkpoints/`.
    *   `last.ckpt`: Always saved.
    *   `step_*.ckpt`: Top 3 based on validation loss.
