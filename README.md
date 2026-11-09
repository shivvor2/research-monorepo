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
- [MadLM](src/research_lib/architectures/mad_lm.py): MAD-style causal LM harness taking already-built sequence mixers ([Poli et al., 2024](https://arxiv.org/abs/2403.17844)): no position embedding, per-mixer `[mixer, SwiGLU]` pre-norm LayerNorm blocks, a final LayerNorm and a biased linear head; the MAD backbone init applies to the backbone only (mixers keep their own init).

**Attention Layers:**
- [Multi-Head Attention](src/research_lib/layers/attention/rotary_mha.py): With RoPE embeddings, QK-Norm, and [XSA](https://arxiv.org/pdf/2603.09078).
- [CausalSelfAttention](src/research_lib/layers/attention/causal_self_attention.py): A mixer-interface wrapper around `RotaryMultiheadAttention` — takes `(B, T, d)`, runs causal self-attention and returns `(B, T, d)`; the positive-control mixer of the structured-recall pilot.
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
- [SwiGLU](src/research_lib/layers/swiglu.py): a bias-free Swish-gated linear unit (`w3(silu(w1 x) * w2 x)`) with MAD's inner-width rounding (default 352 at `d_model=128`).
- [Squared ReLU](src/research_lib/layers/activations.py)
- [Tanh Soft Capping](src/research_lib/layers/logit_clipping.py)


### PyTorch Lightning Components

<!-- Separate into subsections (e.g., Modules, DataModules) when membership scales -->

- [DualOptimizerModule](src/research_lib/training/modules/dual_optimizer.py): LightningModule for training with two optimizers. Supports gradient accumulation scheduling.
    - *Reference:* Implementation details write-up on [r/MachineLearning](https://www.reddit.com/r/learnmachinelearning/comments/1qiw6p0/the_global_step_trap_when_using_multiple/).
- [FineWebDataModule](src/research_lib/data/fineweb_datamodule.py): DataModule handling [`kjj0/finewebedu10B-gpt2`](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2) from Hugging Face.
- [SyntheticRecallDataModule](src/research_lib/data/synthetic/datamodule.py): MQAR (Zoology format), repeated-key overwrite and entity-field with placement classes. Streams fresh seeded training batches and holds fixed validation/test slices; includes answer-position metrics (per-slice, per-class accuracy, old-value leakage). The overwrite generator's `reserved_filler` option restricts filler to a reserved token range (`[V/2-F, V/2)`) so no filler pair can mimic a write. [Synthetic task registry](src/research_lib/data/synthetic/registry.py) maps a task name to its config type, generator and slice convention, and fixes each task's training regime (streaming vs fixed set). *(Reference generators for the multi-state GDN-2 pilot.)*
- [entity_field](src/research_lib/data/synthetic/entity_field.py): the entity-field generator (`EntityFieldConfig`, `generate_entity_field`). Stores `E` entities of `M` fields each (`N = E * M` bindings), updates one field of a share of the entities (written twice: `v_old` then `v_new`), and labels the bindings by how they sit relative to an update: `same` / `adjacent` / `distant` for the updated pairs, `sibling` (written before the update) / `sibling_late` (written after) / `untouched` for the rest — the sibling being the class an erase gate must spare. Multi-query `[e, f, answer]` format after Zoology; `reserved_filler` is always on. *(Reference generator for the multi-state GDN-2 pilot.)*
- [FixedSetRecallDataModule](src/research_lib/data/synthetic/fixed_set.py): a general fixed-training-set DataModule (a task provides a dataset function + a row transform): the training set is materialised once and presented as a fresh permutation per epoch (per-epoch cosine regime), with worker-independent, resumable whole-batch streaming and a val/test leakage check.
- [MAD fuzzy recall generator](src/research_lib/data/synthetic/mad_fuzzy.py): re-implements MAD's fuzzy in-context recall generator (arXiv:2403.17844; `athms/mad-lab@0f49a45`) call for call on the numpy RNG, plus the unshifted-label batch mapping. [MacroTokenAccuracy / TokenAccuracy](src/research_lib/data/synthetic/mad_metrics.py): MAD's macro token accuracy via `torchmetrics` (default averaging) and the plain scored-token accuracy, as extra metrics of `MGDN2SyntheticRecallModule`.
- [Synthetic run control](src/research_lib/training/run_control.py): the general run-control helpers (`disable_tf32`, `assert_fit_finished`, `in_shard`, `should_skip`, `claim_run`, `RUN_DONE_STATUS`), free of any one project's conventions.
- [Synthetic run records](src/research_lib/training/run_records.py): general run-record artifacts and a convention-free pull (`artifact_name`, `run_record_metadata`, `pull_decision`, `ArtifactLayout`, `pull`, `GroupSummary`), plus [wandb_transport](src/research_lib/training/wandb_transport.py) for the upload side and [records](src/research_lib/training/records.py) (`load_records`) for the record table.
- [ExperimentStateCallback](src/research_lib/training/callbacks/experiment_state.py): Callback to track the latest checkpoint path for automated experiment resumption.
- [MGDN2SyntheticRecallModule](src/research_lib/training/modules/mgdn2_synthetic_recall.py): task-specific training module of the multi-state GDN-2 synthetic pilot (answer-only loss with a full-sequence switch, `RecallMetrics` logging).
- [RunSummaryCallback](src/research_lib/training/callbacks/run_summary.py): One JSON per run, written atomically at every stage: config + hash, provenance, timing, stopping, a lifecycle `status` and the outputs of the run's `Collector`s (`metrics_by_prefix`, `param_counts` factories).
- [LayerStatsCallback](src/research_lib/training/callbacks/layer_stats.py): Collects and logs per-layer statistics during validation/test epochs for any layer following the two-attribute `StatsProvider` convention.
- [RecallMetrics](src/research_lib/data/synthetic/metrics.py): takes a task's `class_names` (default the overwrite `OVERWRITE_CLASSES`) so per-class keys follow the task, and `MGDN2SyntheticRecallModule` exposes a configurable monitor key and pluggable extra metrics that receive the same predictions as `RecallMetrics`.

### Utilities
- [Optimizer Scheduling Suite](src/research_lib/training/scheduling/README.md): Flexible parameter scheduling system.
    - *Reference:* [r/MachineLearning post](https://www.reddit.com/r/MachineLearning/comments/1rfer1y/p_implementing_better_pytorch_schedulers/) (#2 post of the day: 27-02-26)
- [Parameter Partitioning Helpers](src/research_lib/training/param_utils.py): Filtering and grouping via PEFT-style pattern matching.
- [Paired Contrasts](src/research_lib/utils/stats.py): Per-unit paired contrasts (`paired_contrast`), the 2×2 interaction / difference-in-differences (`interaction_contrast`, `Factor`) and percentile `bootstrap_ci` of any statistic.

## Installation (Development)

The project uses [uv](https://docs.astral.sh/uv/) and commits a `uv.lock` that pins every dependency for Linux x86_64 (Python 3.13, PyTorch's CUDA 13 wheels). Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For development, create the environment (including the `dev` dependency group) and install the project in editable mode:

```bash
uv sync
```

On a machine that only runs experiments, install exactly the lock. This fails instead of re-resolving if `pyproject.toml` and `uv.lock` disagree:

```bash
uv sync --frozen
```

Run commands inside the environment with `uv run <cmd>`, or activate `.venv`:

```bash
uv run pytest -v
```

Add a dependency with `uv add <package>` (this updates `uv.lock`); remove one with `uv remove <package>`.

Set up pre-commit hooks to ensure formatting consistency:
```bash
uv run pre-commit install
```

Verify the installation by running the test suite:
```bash
uv run pytest -v
```
