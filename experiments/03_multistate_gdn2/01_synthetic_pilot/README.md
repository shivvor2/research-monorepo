# 01 — Synthetic Pilot (MQAR + Repeated-Key Overwrite)

The pilot for the multi-state GDN-2 interaction hypothesis: a
Zoology-style two-layer LM (layer 0 BaseConv, layer 1 the gated-delta
layer under test) trained on MQAR and on repeated-key overwrite,
measuring the pre-registered primary statistic

Δ_int = (X-GDN2 − X-KDA) − (GDN2 − KDA), X ∈ {LL, WS}

on overwrite, class `distant`, K = 2·d_k, with per-seed bootstrap CIs.
The protocol (`06_experiment_protocol.md` in the project notes) is the
source of truth for every setting: grid, seeds, decision rules, and the
calibration below.

## Quick start

```bash
# Tiny CPU smoke run (a few minutes; writes runs/mqar/<run_id>.json):
python 01_train.py --config-name=test

# One step of every layout (multirun):
python 01_train.py --config-name=debug --multirun
```

## Phase workflow (calibration is frozen before any multi-state run)

1. **Phase 1 — calibration** (selection seeds 100–101; nothing here is
   frozen yet, so the sweeps set explicit LRs):
   ```bash
   python 01_train.py +sweep=calib_dmodel --multirun    # d_model in {32, 64}
   python 01_train.py +sweep=calib_harness --multirun  # zoology vs mixer_x2
   python 01_train.py +sweep=calib_lr --multirun       # LR per core config
   ```
   Then fill in `conf/frozen_protocol.yaml` (the chosen `d_model`, the
   harness switches, the LR table), set `frozen: true`, and commit it.
   `01_train.py` refuses any run with `experiment.phase >= 2` until the
   protocol is frozen.

2. **Phase 2 — tier 1** (the Δ_int core cells, seeds 0–9):
   ```bash
   python 01_train.py +sweep=tier1_core --multirun
   ```

### Learning-rate inheritance

Only the core configurations (gate {kda, full} × layout {single, ws, ll},
both tasks) and the anchors are LRs swept. Every other tier inherits, and
no other LR is tuned:

| tier | LR entry used |
|---|---|
| gate decomposition (`b_only`, `w_only`) | `(layout, full)` of the same task |
| frozen control (`fs`) | `(ws, same gate)` |
| λ controls (`static`, `lin`) | the λ-`mlp` entry of `(layout, gate)` |
| matched bytes | `(single, full)` (its own configuration) |
| R1, long sequences | the core entries unchanged |
| anchors | their own entries, keyed by `d_model` (gate `gdn` is never calibrated in the main table) |

`01_train.py` resolves the entry, logs which one was used, and fails
loudly when a run requests an entry that does not exist (e.g. `gdn`
outside the anchor tier).

3. **Phase 3 — tiers 2–6** (gate decomposition, frozen control, λ
   controls, matched bytes, anchors): `+sweep=tier2_gates`,
   `+sweep=tier3_frozen`, `+sweep=tier4_lambda`,
   `+sweep=tier5_matched_bytes`, `+sweep=tier6_anchors`.

4. **Phase 4 — long-T tier**: `+sweep=tier_longT` (T = 2048 sparse MQAR).

## Analysis

```bash
python 02_analyze.py            # runs/ -> analysis/ (CSVs + summary.md)
```

Writes the primary endpoint (Δ_int with bootstrap CIs and the
pre-registered decision-rule readings), Δ_int for every other measured
cell, predictions 1–4, the anchor table next to Log-Linear Table 2, λ /
erase statistics per slot, and the missing-run report per tier.

## Device-count rule

All runs within one tier use the same device count. The long-T tier runs
one run per GPU (launch separate processes with `CUDA_VISIBLE_DEVICES=0`,
`=1`, …), not DDP: the models are small, so independent runs scale
perfectly while DDP mostly adds communication overhead.

## Outputs

- `runs/<task>/<run_id>.json` — one JSON per run, written at the start
  and end of fitting, at the end of testing and on failure. Top-level
  sections: `run_id`, `status` (`running` → `fit_done` → `test_done`;
  `failed` / `interrupted` on an exception, with `error`), `config` +
  `config_hash` (seed excluded), `seed`, `git`, `versions`, `device`,
  `timing`, `stopping` (`early_stopped`, `monitor`, `mode`, `best_value`,
  `best_step`), and the run's collector sections: `params` (total /
  trainable / the λ group), `state` (`bytes_per_mixer`),
  `val_final`, `mixer_stats_final`, `test` — plus `lr_resolved` and
  `decay_scheme`. Runs whose status is not `test_done` are incomplete;
  the analysis lists them. `run_id` is
  `{tier_}{task}-{layout}-{gate_mode}-{lambda_mode}-d{d}-lr{lr}-s{seed}`,
  plus a suffix for every non-default value the base ID leaves out:
  `-wd-{scheme}` (weight-decay scheme other than `grouped`),
  `-pat-{value}` (`layer_pattern` other than `zoology`),
  `-sm-{value}` (`state_mixer` other than `none`) and
  `-ev{value}` (`expand_v` other than `1`); the suffixes go before
  `-s{seed}`, in that order.
- `logs/` — CSV (and optional wandb) logs; `outputs/`, `multirun/` —
  hydra job dirs; `analysis/` — the analysis output. All git-ignored.
- Final checkpoints are **on** by default (`output.save_final_checkpoint`,
  about 2 MB per run at the pilot size): every run saves its final weights
  next to its run JSON (`runs/<task>/<run_id>.pt`), so the mechanism
  probes and finer λ breakdowns can be computed afterwards without
  retraining.

## Disposable-instance rules (rented machines)

1. One JSON per run; results leave the machine from day one.
2. `rsync -avz` the `runs/` folder off the machine regularly (and
   wandb if enabled); before destroying an instance, sync one last time
   and verify checksums locally (`sha256sum runs/**/*.json` on both
   sides).
3. Lock the environment: record `pip freeze` plus torch/CUDA versions
   with the experiment (they are in every run JSON under `versions`).

## Reference timings

One gated-delta layer, forward and backward per step (fp32, B=256,
T=256, d=32, H=2, d_k=16, C=16, W=8, λ=mlp) — sanity-check the Phase 1
timing measurement against these:

| layout | ms per step |
|---|---|
| single | 41.7 |
| ws | 64.6 |
| fs | 74.2 |
| ll | 54.1 |

Prior: the T=256 grid is roughly 15–40 minutes per run on an RTX 4070
Ti. If the calibrated total exceeds one week, rent; otherwise run
locally.
