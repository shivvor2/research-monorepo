# Experiment 02: NanoMythos (Hybrid Attention + Recurrent Depth + Attention Residuals)

This experiment trains the NanoMythos architecture on FineWeb-Edu 10B, using the same training objective (causal LM) as the NanoGPT baseline but with a different architecture combining:

1. **Hybrid attention** — KimiDeltaAttention linear attention interleaved with full MHA
2. **Recurrent depth** — Weight-shared loop phase with LTI state injection + ACT halting
3. **Attention Residuals** — Pseudo-query based residual accumulation (no explicit skip connections)

## Quick Start

### 1. Download Dataset
```bash
python 01_download_dataset.py
```

### 2. Run Training (Production)
```bash
python 02_pretrain.py
```

### 3. Sanity Check (Test Config)
```bash
python 02_pretrain.py --config-name=test
```

## WandB Configuration

Runs are logged to the **NanoMythos** project by default. To configure the entity (team/user), add it to the config:

```yaml
logging:
  wandb:
    enabled: true
    project: NanoMythos
    entity: shivvor2-individual   # <-- add this if needed
```

Or override via CLI:
```bash
python 02_pretrain.py logging.wandb.entity=your-username
```

The `WANDB_API_KEY` environment variable handles authentication.

## Architecture Notes

**Hidden dimension must be a power of 2** (`n_embd`) because `flash_attn_res`
uses Triton kernels that require power-of-2 block sizes. The production config
uses `n_embd=1024` (instead of the GPT-2 small 768) to satisfy this constraint.
This yields ~150M parameters rather than ~124M.

## Grad Accum Schedule

The production config uses a step-based gradient accumulation schedule:
- **Steps 0–1999**: `grad_accum = 8` (32,768 tokens/step)
- **Steps 2000+**: `grad_accum = 16` (65,536 tokens/step)

This is configured via `training.grad_accum_schedule` (a dict mapping optimizer step → accumulation value). You can override or extend it arbitrarily:

```yaml
training:
  grad_accum_schedule:
    0: 8
    2000: 16
    10000: 32
```

To use a constant accumulation instead, set `grad_accum_schedule: null` and use `grad_accum`:
```yaml
training:
  grad_accum_schedule: null
  grad_accum: 16
```

## Dual Optimizer Partitioning

Parameters are split between **NorMuon** (matrix optimizer) and **CautiousAdamW** (vector optimizer) based on `VECTOR_TARGET_MODULES` in `02_pretrain.py`:

- **NorMuon**: Attention/MLP weight matrices (2D projections)
- **CautiousAdamW**: Embeddings, norms, biases, KDA state parameters (`A_log`, `dt_bias`), LTI injection, ACT halting, short-conv filters, FiLM embeddings, AttnRes pseudo-queries

See the inline comments in `02_pretrain.py` for the full list.
