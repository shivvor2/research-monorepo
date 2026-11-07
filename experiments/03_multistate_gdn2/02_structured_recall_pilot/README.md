# Structured-Recall Pilot

A structured-recall pilot of the multi-state GDN-2 project: new
synthetic recall tasks on which single-state GDN-2 should beat its
reference, run through the shared run machinery in `common/` and
`research_lib`. This
pilot shares the frozen protocol, the analysis reports and the primary
statistic with the first pilot; each task adds its own generator, its
task and harness configs, and its calibration and tier sweeps.

## Quick start

```
# A tiny CPU smoke run (writes a run JSON under runs/<task>/):
python 01_train.py --config-name=test task=mqar

# The debug layouts with the default gated-delta mixer:
python 01_train.py --config-name=debug --multirun \
  task=mqar model.layout=single,ws,ll
# One attention control run:
python 01_train.py --config-name=debug task=mqar model.mixer=attention
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

## Screening workflow

1. Single-state calibration (per task, layout, gate) — read the `full − kda`
   gap column off the calibration report at each gate's selected point.
2. Multi-state calibration and freeze of `conf/frozen_protocol.yaml`.
3. Tier-1 single-state cells — `positive_control` reports the paired gap
   over seeds with a bootstrap CI.

Outputs go under `analysis/` (one CSV per report); run records live under
the launch directory's `runs/<task>/`.

## Layout

```
01_train.py  02_analyze.py  03_pull.py
structured_recall_analysis/   protocol.py + reports/
conf/   config  test  debug  frozen_protocol  harness/{zoology, …}  task/*  sweep/
```

The `task` is a config group whose file names its `harness` via
`defaults` (so a task and its harness cannot be crossed); the mechanism is
`model.mixer` (`gated_delta` / `attention`).
