# Research Monorepo

<!-- Insert Badges Here: CI, Ruff, Black, Codecov, License -->

## What is this?
A monorepo containing:
- Custom PyTorch components and PyTorch Lightning modules
- Full training pipelines (managed with Hydra and OmegaConf)
- Training utilities: parameter partitioning helpers and a scheduling suite

## Experiments

- **03/03/26**: [Modded NanoGPT baseline](experiments/01_nanogpt_base/README.md) (Initial run).
    - Achieved 3.22 validation loss on FineWeb-Edu-10B.
- [Multi-state GDN-2 pilot](experiments/03_multistate_gdn2/README.md): MQAR + repeated-key overwrite experiments measuring the gate × multi-state interaction (calibration, Δ_int tiers, long-T tier).

## Implemented Components

### PyTorch Components

**Architectures:**
- [Modded NanoGPT (Base)](src/research_lib/architectures/modded_nanogpt_base.py): Transformer blocks with MHA and pre-norm (RMSNorm).
- [ZoologyLM](src/research_lib/architectures/zoology_lm.py): Zoology-style causal LM harness taking already-built sequence mixers (`zoology_mixers` lays out the Zoology patterns: BaseConv in even blocks, the experiment's `make_mixer()` in odd blocks, or one mixer everywhere; no positional embedding).

**Attention Layers:**
- [Multi-Head Attention](src/research_lib/layers/attention/rotary_mha.py): With RoPE embeddings, QK-Norm, and [XSA](https://arxiv.org/pdf/2603.09078).
- [Gated delta-rule token mixer (KDA / GDN-2 / GDN; **Reference**)](src/research_lib/layers/attention/gated_delta/reference.py): Single-state gated delta rule with channel-wise erase/write gates and channel-wise decay ([GDN-2](https://arxiv.org/abs/2605.22791)), reducing exactly to [KDA](https://arxiv.org/abs/2510.26692) and [Gated DeltaNet](https://arxiv.org/abs/2412.06464). Includes the GDN-2 Table 5 gate ablations, a causal short conv, and chunkwise/recurrent scan paths.
    - *Reference implementation* (naive PyTorch + autograd): use fla kernels for production-scale training.
- [Multi-state gated delta-rule mixer (WS / FS / Fenwick layouts; **Reference**)](src/research_lib/layers/attention/gated_delta/multistate.py): Transported multi-state memory over the GDN-2 transition — windowed uniform (WS), frozen control (FS) and Fenwick/log-linear ([Log-Linear](https://arxiv.org/abs/2506.04761)) layouts, with static/linear/MLP λ mixing weights initialized to λ≡1 (exactly GDN-2 at init). One class for all variants; chunkwise/recurrent paths; `layout="single"` is the single-state reference layer.
    - *Reference implementation* (naive PyTorch + autograd): use fla kernels for production-scale training.

**Optimizers:**
- [CautiousAdamW](src/research_lib/optimizers/cautious_adamw.py): Patched AdamW optimizer with cautious stepping ([paper](https://arxiv.org/abs/2510.12402)).
- [NorMuon](https://arxiv.org/abs/2505.16932): Improved Muon Optimizer ([paper](https://arxiv.org/abs/2510.05491)) featuring:
    - Polar Express orthogonalization ([paper](https://arxiv.org/abs/2505.16932))
        - Uses Triton Kernels from [`KellerJordan/modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py).
    - Adafactor-style second moment estimation.
    - Cautious weight decay ([paper](https://arxiv.org/abs/2510.12402)).

**Miscellaneous:**
- [Feed Forward](src/research_lib/layers/feed_forward.py):
    - Standard 2-layer MLP for LLMs.
- [BaseConv](src/research_lib/layers/base_conv.py): Zoology's gated depthwise causal short convolution (`y = conv_k(u) ⊙ (Wu + c) + u`).
- [Squared ReLU](src/research_lib/layers/activations.py)
- [Tanh Soft Capping](src/research_lib/layers/logit_clipping.py)


### PyTorch Lightning Components

<!-- Separate into subsections (e.g., Modules, DataModules) when membership scales -->

- [DualOptimizerModule](src/research_lib/training/modules/dual_optimizer.py): LightningModule for training with two optimizers. Supports gradient accumulation scheduling.
    - *Reference:* Implementation details write-up on [r/MachineLearning](https://www.reddit.com/r/learnmachinelearning/comments/1qiw6p0/the_global_step_trap_when_using_multiple/).
- [FineWebDataModule](src/research_lib/data/fineweb_datamodule.py): DataModule handling [`kjj0/finewebedu10B-gpt2`](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2) from Hugging Face.
- [SyntheticRecallDataModule](src/research_lib/data/synthetic/datamodule.py): MQAR (Zoology format) and repeated-key overwrite with placement classes. Streams fresh seeded training batches and holds fixed validation/test slices; includes answer-position metrics (per-slice, per-class accuracy, old-value leakage). *(Reference generators for the multi-state GDN-2 pilot.)*
- [ExperimentStateCallback](src/research_lib/training/callbacks/experiment_state.py): Callback to track the latest checkpoint path for automated experiment resumption.
- [MGDN2SyntheticRecallModule](src/research_lib/training/modules/mgdn2_synthetic_recall.py): task-specific training module of the multi-state GDN-2 synthetic pilot (answer-only loss with a full-sequence switch, `RecallMetrics` logging).
- [RunSummaryCallback](src/research_lib/training/callbacks/run_summary.py): One JSON per run, written atomically at every stage: config + hash, provenance, timing, stopping, a lifecycle `status` and the outputs of the run's `Collector`s (`metrics_by_prefix`, `param_counts` factories).
- [LayerStatsCallback](src/research_lib/training/callbacks/layer_stats.py): Collects and logs per-layer statistics during validation/test epochs for any layer following the two-attribute `StatsProvider` convention.

### Utilities
- [Optimizer Scheduling Suite](src/research_lib/training/scheduling/README.md): Flexible parameter scheduling system.
    - *Reference:* [r/MachineLearning post](https://www.reddit.com/r/MachineLearning/comments/1rfer1y/p_implementing_better_pytorch_schedulers/) (#2 post of the day: 27-02-26)
- [Parameter Partitioning Helpers](src/research_lib/training/param_utils.py): Filtering and grouping via PEFT-style pattern matching.
- [Paired Contrasts](src/research_lib/utils/stats.py): Per-unit paired contrasts (`paired_contrast`), the 2×2 interaction / difference-in-differences (`interaction_contrast`, `Factor`) and percentile `bootstrap_ci` of any statistic.

## Installation (Development)

Install the project in editable mode with development dependencies inside a managed virtual environment:

```bash
pip install -e ".[dev]"
```

Set up pre-commit hooks to ensure formatting consistency:
```bash
pre-commit install
```

Verify the installation by running the test suite:
```bash
pytest -v
```
