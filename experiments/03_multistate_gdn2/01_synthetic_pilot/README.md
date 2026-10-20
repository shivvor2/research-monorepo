# 01 — Synthetic Pilot (MQAR + Repeated-Key Overwrite)

The pilot for the multi-state GDN-2 interaction hypothesis: a
Zoology-style two-layer LM (layer 0 BaseConv, layer 1 the gated-delta
layer under test) trained on MQAR and on repeated-key overwrite,
measuring the pre-registered primary statistic

Δ_int = (X-GDN2 − X-KDA) − (GDN2 − KDA), X ∈ {LL, WS}

on overwrite, class `distant`, K = 2·d_k, with per-seed bootstrap CIs.
The experiment protocol is the source of truth for every setting:
grid, seeds, decision rules, and the calibration below.

## Quick start

```bash
# Tiny CPU smoke run (a few minutes; writes runs/mqar/<run_id>.json):
python 01_train.py --config-name=test

# One step of every layout (multirun):
python 01_train.py --config-name=debug --multirun
```

## Environment

The pilot runs in the committed `uv.lock` (Linux x86_64, Python 3.13,
Lightning ≥ 2.4); `uv sync --frozen` installs exactly that environment.
**Do not run `uv lock --upgrade` or `uv add` between the first calibration
run and the end of the pilot**. Every run record will
store the SHA-256 of `uv.lock`, so a tier that mixed lock versions would
be visible.

## Phase workflow (calibration is frozen before any multi-state run)

1. **Phase 1 — calibration** (selection seeds 100–101; nothing here is
   frozen yet, so the sweeps set explicit LRs):
   ```bash
   python 01_train.py +sweep=calib_dmodel --multirun    # d_model in {32, 64}
   ```
   Record the chosen `d_model` in `conf/frozen_protocol.yaml` **and** set
   `model.d_model` in `conf/config.yaml`: once the frozen file names a
   width, `01_train.py` checks every run against it (only `calib_dmodel`,
   `calib_lr_anchors` and `tier6_anchors` are exempt - they set their own).
   ```bash
   python 01_train.py +sweep=calib_harness --multirun  # zoology vs mixer_x2
   ```
   The same for the harness switches (`harness.layer_pattern` /
   `harness.state_mixer` in the frozen protocol; `model.layer_pattern` /
   `model.state_mixer` in `config.yaml`). The anchors are **not** exempt
   from the harness: it is shared by every tier. Only then:
   ```bash
   python 01_train.py +sweep=calib_lr --multirun       # LR per core config
   ```
   Finally fill in the LR table, set `frozen: true`, and commit it.
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

## Training

Under `training.loss_on: answers` (the default) the LM head is evaluated
**only at the answer positions**: the trunk (embedding, mixers, final
norm) still runs on every position, but `lm_head` is applied to the
answer rows alone (`training.answer_head: true`, the default). This is
exact — the loss is the mean cross-entropy over the answer positions,
positions outside them appear neither in the sum nor in its denominator
and already get exactly zero gradient (`ignore_index`), the head acts on
each position separately, nothing after it mixes positions, and the
metrics read only answer positions — so the loss, every gradient and
every metric are identical to the full head in exact arithmetic (the
fp64 equivalence tests in `src/research_lib/training/tests/` check
this). In floating point the two paths differ at rounding level
(different matmul shapes and reduction order), so runs with the
answer-position head and with the full head are not bit-identical;
`training.answer_head` is therefore part of the run identity.
`training.answer_head: false` restores the full head
(for checks only; it changes the run identity).

## Running, stopping and continuing

**What `test_done` guarantees.** A run record with `status == "test_done"`
means the fit ended normally (max steps or early stop) **and** the test
pass completed on the final weights **and**, when checkpoint saving is
on, the checkpoint was fully written before the test pass. Anything else
(`running`, `fit_done`, `interrupted`, `failed`) is an incomplete run:
relaunch it.

**Stopping.**

- Ctrl-C interrupts the current run; the record is marked `interrupted`
  (Lightning ≥ 2.4 raises `SIGTERMException` on SIGTERM, which is also
  an interruption, not a failure).
- Killing the machine loses the runs in progress; the tracker never
  records a wrong `test_done`.
- `touch STOP` in the launch directory stops the sweep at the next run
  boundary: no new run starts while the file exists, and the script
  never deletes it. In a multirun this ends the whole sweep; in a
  single run it just exits.

**Continuing.** Relaunch the same command. With
`launch.skip_completed: true` (protocol tiers only — the smoke and
`debug` configs always run), a run whose record is `test_done` with the
same identity is skipped; a non-done record is overwritten by the
re-run. A record of a *different* run at the same path is refused by
`RunSummaryCallback` (the identity is the `(config_hash, seed)` pair,
with the operational keys `output`, `logging`, `launch` and
`data.num_workers` excluded from the hash).

**Several runs on one GPU (shards).** Launch the identical sweep command
in several processes, each with `launch.shard_index=k` and
`launch.num_shards=N` (e.g. `launch.shard_index=0 launch.num_shards=2`
and `launch.shard_index=1 launch.num_shards=2`). Job `k` of the sweep
runs only in the process with `shard_index == k % num_shards`, so every
shard moves along the sweep's priority order (overwrite before mqar,
seed-major). A per-run lock file (`runs/<task>/<run_id>.lock`, empty,
not a record) prevents two processes from running the same run at once:
the second one skips it.

**Curves.** The CSV logger writes next to the run records:
`runs/<task>/curves/<run_id>/version_0/metrics.csv`. A relaunch removes
a stale curves directory first (a relaunch is a fresh run).

## Analysis

The analysis answers different questions at different times, so it is
one report per question (`pilot_analysis/`), all reading the same run
records (`runs/<task>/*.json`); only `test_done` records enter any
statistic.

```bash
python 02_analyze.py --list              # every report, one line each
python 02_analyze.py                     # all: CSVs + <name>.md per report, plus summary.md
python 02_analyze.py primary anchors     # a selection: writes only those
python 02_analyze.py calibration         # the Phase-1 readings (never part of all)
```

The reports: the primary endpoint (Δ_int with bootstrap CIs and the
pre-registered decision-rule readings, with the uniform-decay
robustness table next to it), Δ_int for every measured cell, one
report per prediction (gate decomposition, transported vs the frozen
control, λ controls), write distance, the long-sequence tier, the
anchors next to Log-Linear Table 2, matched bytes, λ / erase
statistics per slot, and the missing / incomplete runs. Each report
writes `analysis/<name>.md` and its tables as `analysis/<stem>.csv`;
`all` additionally writes `analysis/summary.md` — every report's
section, in registry order.

### Calibration report

Each calibration sweep ends with a choice that is written by hand into
`conf/frozen_protocol.yaml` and `conf/config.yaml` and committed — the
report computes the readings, it never edits configuration files:

1. run a calibration sweep (`+sweep=calib_dmodel`, then
   `+sweep=calib_harness`, then `+sweep=calib_lr` and
   `+sweep=calib_lr_anchors`);
2. run `python 02_analyze.py calibration`: the d_model verdict (the
   smallest d whose mean best validation accuracy lies in [0.5, 0.95];
   fallback 32), the harness numbers next to the published range (a
   judgment: keep the default unless far outside), the selected LR per
   core configuration (best mean; ties: fewer mean steps to the best
   value, then the smaller LR), the anchor LRs, missing runs per
   calibration tier, and a `frozen_protocol.yaml` snippet filled from
   the readings (`null` for the steps that have not run);
3. copy the snippet into `conf/frozen_protocol.yaml`, make the harness
   choice (its block stays `null` in the snippet on purpose), and set
   the matching values in `conf/config.yaml` (`model.d_model`, and the
   harness switches `model.layer_pattern` / `model.state_mixer`);
4. commit before the next sweep; when everything is filled in, set
   `frozen: true` and commit — only then can Phase 2 run.

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
  `config_hash` (seed excluded; the operational keys `output`,
  `logging`, `launch` and `data.num_workers` are excluded too, recorded
  as `hash_excluded_keys`), `seed`, `git`, `versions`, `device`,
  `provenance` (SHA-256 of `uv.lock`), `timing`, `stopping`
  (`early_stopped`, `monitor`, `mode`, `best_value`, `best_step`), and
  the run's collector sections: `params` (total / trainable / the λ
  group), `state` (`bytes_per_mixer`), `val_final`,
  `mixer_stats_final`, `test` — plus `lr_resolved`, `decay_scheme`,
  `precision_flags` (the two TF32 flags, both off) and
  `final_checkpoint` (the saved file name, or `null`). Runs whose
  status is not `test_done` are incomplete; the analysis lists them.
  `run_id` is
  `{tier_}{task}-{layout}-{gate_mode}-{lambda_mode}-d{d}-lr{lr}-s{seed}`,
  plus a suffix for every non-default value the base ID leaves out:
  `-wd-{scheme}` (weight-decay scheme other than `grouped`),
  `-pat-{value}` (`layer_pattern` other than `zoology`),
  `-sm-{value}` (`state_mixer` other than `none`) and
  `-ev{value}` (`expand_v` other than `1`); the suffixes go before
  `-s{seed}`, in that order.
- `runs/<task>/curves/<run_id>/version_0/metrics.csv` — the training
  curves of each run, next to its record.
- `runs/<task>/<run_id>.lock` — empty per-run lock file (not a record;
  the analysis reads only `*.json`).
- `bench/<timestamp>/{bench.json,summary.md}` — the timing benchmark
  outputs (git-ignored).
- `logs/` — optional wandb logs; `outputs/`, `multirun/` — hydra job
  dirs; `analysis/` — the analysis output. All git-ignored.
- Final checkpoints are **off** by default (`output.save_final_checkpoint`):
  final weights are saved only for runs whose weights are needed, e.g.
  the mechanism probes (a re-run in the same environment is close but
  not bit-identical). When on, the checkpoint is saved atomically after
  fit and before the test pass, next to the run JSON
  (`runs/<task>/<run_id>.pt`, git-ignored).

## Disposable-instance rules (rented machines)

1. One JSON per run; results leave the machine from day one.
2. `rsync -avz` the `runs/` folder off the machine regularly (and
   wandb if enabled); before destroying an instance, sync one last time
   and verify checksums locally (`sha256sum runs/**/*.json` on both
   sides).
3. Lock the environment: record `pip freeze` plus torch/CUDA versions
   with the experiment (they are in every run JSON under `versions`).

## Timing (benchmark)

Before any protocol run, measure the machine: `00_benchmark.py` reports,
per core configuration (task x layout x gate, at d in {32, 64}), the
median ms per training step at batch sizes 64/128/256, the peak memory,
the GPU-busy fraction from the profiler, the data-generation time per
batch, the time of one validation pass, and the aggregate throughput of
1, 2 and 4 concurrent runs on one GPU:

```bash
python 00_benchmark.py                 # a few minutes; writes bench/<timestamp>/
```

It reuses the training script's builders, measured batches are
pre-generated (data generation is excluded from the step time), and the
validation pass is the real one (loss plus metrics over the fixed
validation set). The concurrency workers are subprocesses of the same
script with the variant's config overrides, so they compose the exact
config the training script would.

Read `bench/<timestamp>/summary.md` for:

- **overhead-bound / throughput-bound / data-bound** per configuration:
  overhead-bound means the step time is nearly flat in the batch size
  (kernel launches, Python) — run several independent runs per GPU;
  throughput-bound means it grows with the batch size — one run per
  GPU; data-bound means the data time per batch exceeds the step time
  times `data.num_workers` — raise the worker count;
- **runs per GPU**: the smallest concurrency whose aggregate throughput
  is within 10% of the best measured;
- the **grid estimate**: upper-bound GPU-hours per tier and per phase
  (no early stopping; test ~ validation), the wall clock at the
  **measured speedup** of the recommended runs per GPU (N runs side by
  side are not N times faster), and whether Phase 1 + Phase 2 exceed
  the one-week budget (then rent).

The summary ends with a **cross-check**: one real 1,000-step run of the
core configuration. The run's `timing.steps_per_second` is steps per
second, the benchmark's step time is ms per step — convert with
**ms per step = 1000 / `steps_per_second`**. The real run also pays for
data loading, logging and the two validation passes, so its step time
should come out somewhat **slower** than the benchmark's, not equal to
it.

On a machine without CUDA the benchmark measures **one** variant (step
time, data time, validation time) as a smoke check and reports the rest
of the grid and the GPU-only readings (busy fraction, peak memory,
concurrency) as not measured in the summary.

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
