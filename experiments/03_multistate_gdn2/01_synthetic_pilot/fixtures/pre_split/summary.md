# Synthetic pilot analysis

## Primary endpoint (core tier, overwrite, distant, K = 2 d_k)

| layout | metric | d_model | n_seeds | mean | ci_low | ci_high | reading |
|---|---|---|---|---|---|---|---|
| ll | test/acc/slice=32/class=distant | 32 | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |
| ws | test/acc/slice=32/class=distant | 32 | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |

### Under uniform weight decay (tier R1)

| layout | metric | d_model | n_seeds | mean | ci_low | ci_high | reading |
|---|---|---|---|---|---|---|---|
| ll | test/acc/slice=32/class=distant | 32 | 5 | -1.11e-16 | -1.11e-16 | -1.11e-16 | CI entirely below 0: transport makes selective erase redundant; informative, report it |
| ws | test/acc/slice=32/class=distant | 32 | 5 | -1.11e-16 | -1.11e-16 | -1.11e-16 | CI entirely below 0: transport makes selective erase redundant; informative, report it |

## Delta_int on every measured cell (core tier)

| task | layout | metric | n_seeds | mean | ci_low | ci_high | reading |
|---|---|---|---|---|---|---|---|
| overwrite | ws | test/acc | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ll | test/acc | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ws | test/acc/slice=32 | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ll | test/acc/slice=32 | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ws | test/acc/slice=32/class=distant | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ll | test/acc/slice=32/class=distant | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ws | test/acc/slice=32/class=single | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ll | test/acc/slice=32/class=single | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ws | test/acc/dist=32-64 | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| overwrite | ll | test/acc/dist=32-64 | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |

## Predictions 1-4 (each on its own tier)

| prediction | layout | gate_mode | n_seeds | mean | ci_low | ci_high |
|---|---|---|---|---|---|---|
| P1 recovered fraction (test/acc/slice=32/class=distant) | ws | b_only | 5 | 0 | 0 | 0 |
| P1 recovered fraction (test/acc/slice=32/class=distant) | ws | w_only | 5 | 0 | 0 | 0 |
| P1 recovered fraction (test/acc/slice=32/class=distant) | ll | b_only | 5 | 0 | 0 | 0 |
| P1 recovered fraction (test/acc/slice=32/class=distant) | ll | w_only | 5 | 0 | 0 | 0 |
| P2 ws - fs (test/acc/slice=32/class=distant) | ws-fs | full | 5 | -0.39 | -0.39 | -0.39 |
| P2 ws - fs (test/leak/slice=32/class=distant) | ws-fs | full | 5 | 0 | 0 | 0 |
| P2 ws - fs (test/acc/slice=32/class=distant) | ws-fs | kda | 5 | -0.54 | -0.54 | -0.54 |
| P2 ws - fs (test/leak/slice=32/class=distant) | ws-fs | kda | 5 | 0 | 0 | 0 |
| P3 ws - fs (test/acc/slice=32/class=single) | ws-fs | full | 5 | -0.39 | -0.39 | -0.39 |
| P3 ws - fs (test/acc/slice=32/class=single) | ws-fs | kda | 5 | -0.54 | -0.54 | -0.54 |
| P4 static - single-full (test/acc/slice=32/class=distant) | ws | full | 5 | 0.49 | 0.49 | 0.49 |
| P4 static - single-full (test/acc/slice=32/class=distant) | ll | full | 5 | 0.49 | 0.49 | 0.49 |

## Write distance (core tier)

| layout | gate_mode | metric | n_seeds | mean |
|---|---|---|---|---|
| ll | full | test/acc/dist=32-64 | 10 | 0.7 |
| ll | full | test/leak/dist=32-64 | 10 | 0 |
| ll | kda | test/acc/dist=32-64 | 10 | 0.5 |
| ll | kda | test/leak/dist=32-64 | 10 | 0 |
| single | full | test/acc/dist=32-64 | 10 | 0.5 |
| single | full | test/leak/dist=32-64 | 10 | 0 |
| single | kda | test/acc/dist=32-64 | 10 | 0.4 |
| single | kda | test/leak/dist=32-64 | 10 | 0 |
| ws | full | test/acc/dist=32-64 | 10 | 0.6 |
| ws | full | test/leak/dist=32-64 | 10 | 0 |
| ws | kda | test/acc/dist=32-64 | 10 | 0.45 |
| ws | kda | test/leak/dist=32-64 | 10 | 0 |

## Long-sequence tier (Delta_int)

| layout | metric | n_seeds | mean | ci_low | ci_high | reading |
|---|---|---|---|---|---|---|
| ws | test/acc | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| ll | test/acc | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |
| ws | test/acc/slice=32 | 10 | 0.05 | 0.05 | 0.05 | CI entirely above 0: supports the hypothesis at pilot scale |
| ll | test/acc/slice=32 | 10 | 0.1 | 0.1 | 0.1 | CI entirely above 0: supports the hypothesis at pilot scale |

## Anchors next to Log-Linear Table 2

| d_model | layout | n_seeds | acc_mean | acc_std | log_linear_acc | log_linear_std |
|---|---|---|---|---|---|---|
| 16 | ll | 5 | 0.99 | 0 | 40 | 1.4 |
| 16 | single | 5 | 0.99 | 0 | 38.4 | 1 |
| 32 | ll | 5 | 0.99 | 0 | 84.4 | 1.2 |
| 32 | single | 5 | 0.99 | 0 | 79 | 2.1 |
| 64 | ll | 5 | 0.99 | 0 | 99 |  |
| 64 | single | 5 | 0.99 | 0 | 99 |  |

## Missing runs per tier

| tier | expected | observed | missing |
|---|---|---|---|
| calib_dmodel | 4 | 24 | -20 |
| calib_harness | 4 | 24 | -20 |
| calib_lr | 72 | 24 | 48 |
| calib_lr_anchors | 36 | 24 | 12 |
| tier1_core | 120 | 60 | 60 |
| tier2_gates | 60 | 30 | 30 |
| tier3_frozen | 20 | 10 | 10 |
| tier4_lambda | 40 | 20 | 20 |
| tier5_matched_bytes | 10 | 5 | 5 |
| tier6_anchors | 30 | 30 | 0 |
| tier_longT | 60 | 60 | 0 |
| R1 | 30 | 30 | 0 |

## Incomplete runs (status is not test_done)

_none_

## Notes and limitations

- Missing keys in a metrics dict mean 'not measured' (empty cells are omitted by RecallMetrics), not zero.
- Every statistic above selects its rows by tier: several tiers repeat core configurations (matched bytes, the lambda controls, R1, the long-sequence tier), so filtering by configuration alone would create duplicated cells. Calibration tiers and smoke runs never enter any statistic.
