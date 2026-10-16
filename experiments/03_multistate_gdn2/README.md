# 03 — Transported Multi-State GDN-2

This folder holds the experiments of the transported multi-state GDN-2
project: does GDN-2's decoupled, channel-wise erase gate show a positive
*interaction* with a transported multi-state readout? The primary
statistic is

$$\Delta_{\text{int}}=(\text{LL-GDN2}-\text{LL-KDA})-(\text{GDN2}-\text{KDA}),$$

also computed with WS in place of LL. The bucket states sum exactly to
the GDN-2 state, so collateral erase is identical in single- and
multi-state; the hypothesis is that the multi-state readout raises the
*value* of preserving old content.

Everything is pre-registered in the project documents (proposal,
derivations and the experiment protocol); the protocol is the source of
truth for every experimental setting.

## Experiments

| folder | experiment |
|---|---|
| [`01_synthetic_pilot/`](01_synthetic_pilot/README.md) | MQAR + repeated-key overwrite pilot: calibration, then the Δ_int grid (tiers 1–6) and the long-T tier |
