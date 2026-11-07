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

## Screening workflow

1. Single-state calibration (per task, layout, gate) — read the `full − kda`
   gap column off the calibration report at each gate's selected point.
2. Multi-state calibration and freeze of `conf/frozen_protocol.yaml`.
3. Tier-1 single-state cells — `positive_control` reports the paired gap
   over seeds with a bootstrap CI.

Outputs go under `analysis/` (one CSV per report); run records live under
the launch directory's `runs/<task>/`.

## MAD fuzzy recall harness

MAD fuzzy in-context recall (Poli et al., 2024, arXiv:2403.17844; the
`mad_fuzzy` task) is the second task of this pilot: a recall task whose
keys and values are short token motifs, screened as a positive control for
single-state GDN-2. The running harness (`conf/harness/mad.yaml`) follows
MAD's backbone exactly when it does not touch the layer under test — no
position embedding, per-mixer `[mixer, SwiGLU]` pre-norm LayerNorm blocks,
a final LayerNorm and a biased linear head, the MAD backbone init and a
bias-free SwiGLU — in fp32 (the reference layer's triangular solve is
precision-sensitive; MAD uses bf16+TF32) with a separate validation set
and the reported metric being MAD's macro token accuracy at the final
weights. The generator (`research_lib.data.synthetic.mad_fuzzy`) reproduces
MAD's own generator call for call (fixtures in the library tests), training
runs on a fixed set for 200 epochs with a per-epoch cosine LR, and the two
AdamW groups decay only the 2-D linear weights (the embedding, head norms,
biases and the layer's λ/decay parameters are exempt).

### MAD fuzzy calibration workflow

The setting rule and go/no-go are read off the calibration report's MAD
column (`test/acc_mad`) at each gate's selected point:

1. Run `calib_mad_single` and read each gate's best validation `acc_mad`
   over the LR grid (re-run any look-stuck or near-edge cell at seed 101).
2. Keep length 512 if both gates read within [0.15, 0.75]; below 0.15 move
   to length 256, above 0.75 to length 1024, and repeat the single-state
   calibration at the new length. The setting is chosen on accuracy level
   only, never on the gap.
3. Go/no-go: `full − kda ≥ +0.03` at each gate's best point.
4. On a go, run `calib_mad_multi`, freeze `conf/frozen_protocol.yaml`, then
   `tier1_mad_single` and `tier1_mad_multi`.

## Layout

```
01_train.py  02_analyze.py  03_pull.py
structured_recall_analysis/   protocol.py + reports/
conf/   config  test  debug  frozen_protocol  harness/{zoology, …}  task/*  sweep/
```

The `task` is a config group whose file names its `harness` via
`defaults` (so a task and its harness cannot be crossed); the mechanism is
`model.mixer` (`gated_delta` / `attention`).
