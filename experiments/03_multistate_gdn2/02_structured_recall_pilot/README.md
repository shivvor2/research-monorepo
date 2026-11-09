# Structured-Recall Pilot

A structured-recall pilot of the multi-state GDN-2 project: new
synthetic recall tasks on which single-state GDN-2 should beat its
reference, run through the shared run machinery in `common/` and
`research_lib`. This
pilot shares the frozen protocol, the analysis reports and the primary
statistic with the first pilot; each task adds its own generator, its
task and harness configs, and its calibration and tier sweeps.

Every launch names its task (there is no default): the smoke and debug
configs read each task's smoke sizes from `conf/smoke/<task>.yaml`
(selected through the `smoke` group keyed on the task choice), so a new
task's smoke sizes go in exactly one new file there; model, training and
data smoke settings stay shared in `test.yaml` / `debug.yaml`.

## Quick start

```
# A tiny CPU smoke run (writes a run JSON under runs/<task>/):
python 01_train.py --config-name=test task=mqar
python 01_train.py --config-name=test task=entity_field
python 01_train.py --config-name=test task=mad_fuzzy

# The debug layouts with the default gated-delta mixer:
python 01_train.py --config-name=debug --multirun \
  task=mqar model.layout=single,ws,ll
python 01_train.py --config-name=debug --multirun \
  task=entity_field model.layout=single,ws,ll
python 01_train.py --config-name=debug --multirun \
  task=mad_fuzzy model.layout=single,ws,ll
# One attention control run:
python 01_train.py --config-name=debug task=mqar model.mixer=attention
python 01_train.py --config-name=debug task=entity_field model.mixer=attention
python 01_train.py --config-name=debug task=mad_fuzzy model.mixer=attention
```

## Tasks

Each task adds its own generator, config and sweeps.

### Entity–field

The entity–field task stores `E` entities, each holding `M` fields, so
`N = E * M` bindings. A share of the entities is updated: one of their
fields is written twice (`v_old`, then `v_new ≠ v_old`). The query is the
multi-query `[e, f, answer]` recall of Zoology (Arora et al., 2023): given an
entity and a field, recall the latest value. What it tests: an update must
erase the stale `(e, f) → v_old` binding without disturbing the updated
entity's *other* fields, whose keys share the entity's code. The screening
class is the **sibling** — a non-updated pair of an updated entity written
before the update — and its damage reading (`untouched − sibling` accuracy)
is what the erase should keep to zero. Same-scale slices read as the
overwrite task's `K` (`N = E * M` in {8, 16, 32}).

### MAD fuzzy recall harness

MAD fuzzy in-context recall (Poli et al., 2024, arXiv:2403.17844; the
`mad_fuzzy` task) is a recall task whose keys and values are short token
motifs, screened as a positive control for single-state GDN-2. The
running harness (`conf/harness/mad.yaml`) follows MAD's backbone exactly
when it does not touch the layer under test — no position embedding,
per-mixer `[mixer, SwiGLU]` pre-norm LayerNorm blocks, a final LayerNorm
and a biased linear head, the MAD backbone init and a bias-free SwiGLU —
in fp32 (the reference layer's triangular solve is precision-sensitive;
MAD uses bf16+TF32) with a separate validation set and the reported metric
being MAD's macro token accuracy at the final weights. The gated-delta
layer uses 8 heads of width 16 at `d_model=128`; MAD's attention positive
control is a different layout, 16 heads of width 8 with rotary on the full
head width. The generator (`research_lib.data.synthetic.mad_fuzzy`)
reproduces MAD's own generator call for call (fixtures in the library
tests), training runs on a fixed set for 200 epochs with a per-epoch
cosine LR, and the two AdamW groups decay only the 2-D linear weights (the
embedding, head norms, biases and the layer's λ/decay parameters are
exempt). Its `max_seq_len` (the row length) is derived, not hand-set:
`seq_len + 1` rounded up to a multiple of the harness chunk size (via the
`padded_to_chunk` config resolver), and `t_max` follows it.

### MAD fuzzy calibration workflow

LR/weight-decay selection and the length rule read each gate's **best
validation** `acc_mad` (the report's `mean_best_value`); the go/no-go gap
is read from the report's gap column, `test/acc_mad` at each gate's
selected point. The MAD sweep tiers pin `task=mad_fuzzy` (a sweep must not
run another task under a MAD tier name); the length is an ordinary task
key, so a length-resolution rerun overrides `task.seq_len=256` and
`max_seq_len` follows:

1. Run `calib_mad_single` (sweeping LR and weight decay; the report
   selects the `(lr, weight_decay)` pair) and read each gate's best
   validation `acc_mad` (re-run any look-stuck or near-edge cell at seed
   101).

   ```
   python 01_train.py +sweep=calib_mad_single task=mad_fuzzy --multirun
   ```

2. Keep length 512 if both gates read within [0.15, 0.75]; below 0.15 move
   to length 256, above 0.75 to length 1024, and repeat the single-state
   calibration at the new length (`+sweep=calib_mad_single task=mad_fuzzy
   task.seq_len=256 --multirun`; the report groups by each length
   separately and `python 02_analyze.py --seq-len mad_fuzzy=<n>` picks one
   length for the frozen snippet). The setting is chosen on accuracy level
   only, never on the gap.
3. Go/no-go: `full − kda ≥ +0.03` at each gate's best point.
4. On a go, run `calib_mad_multi` and the attention calibration
   (`calib_mad_attention` for the positive control), freeze both the LR
   and weight-decay tables in `conf/frozen_protocol.yaml` and the chosen
   length in `task_settings.mad_fuzzy.seq_len` (a tier-1 run at any other
   length is refused), then at the frozen length:

   ```
   python 01_train.py +sweep=tier1_mad_single task=mad_fuzzy task.seq_len=<frozen> --multirun
   python 01_train.py +sweep=tier1_mad_multi task=mad_fuzzy task.seq_len=<frozen> --multirun
   ```

## Screening workflow

1. Single-state calibration (per task, layout, gate) — read the `full − kda`
   gap column off the calibration report at each gate's selected point. A
   task's attention control is calibrated alongside it (e.g. `calib_ef_
   single` + `calib_ef_attention`); the attention LR feeds
   `lr_table.<task>.attention` in the frozen protocol.
2. Multi-state calibration and freeze of `conf/frozen_protocol.yaml`
   (`lr_table`, `weight_decay_table`, the harness `d_model`, and
   `task_settings.<task>.seq_len` for a task with a length-setting rule).
3. Tier-1 single-state cells — `positive_control` reports the paired gap
   over seeds with a bootstrap CI.

Outputs go under `analysis/` (one CSV per report); run records live under
the launch directory's `runs/<task>/`.

## Layout

```
01_train.py  02_analyze.py  03_pull.py
structured_recall_analysis/   protocol.py + reports/
conf/   config  test  debug  frozen_protocol  harness/{zoology, mad}  smoke/*  task/*  sweep/
```

The `task` is a config group whose file names its `harness` via
`defaults` (so a task and its harness cannot be crossed); the mechanism is
`model.mixer` (`gated_delta` / `attention`).
