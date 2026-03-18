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

## Implemented Components

### PyTorch Components

**Architectures:**
- [Modded NanoGPT (Base)](src/research_lib/architectures/modded_nanogpt_base.py): Transformer blocks with MHA and pre-norm (RMSNorm).

**Attention Layers:**
- [Multi-Head Attention](src/research_lib/layers/attention/rotary_mha.py): With RoPE embeddings and QK-Norm.

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
- [Squared ReLU](src/research_lib/layers/activations.py)
- [Tanh Soft Capping](src/research_lib/layers/logit_clipping.py)


### PyTorch Lightning Components

<!-- Separate into subsections (e.g., Modules, DataModules) when membership scales -->

- [DualOptimizerModule](src/research_lib/training/modules/dual_optimizer.py): LightningModule for training with two optimizers. Supports gradient accumulation scheduling.
    - *Reference:* Implementation details write-up on [r/MachineLearning](https://www.reddit.com/r/learnmachinelearning/comments/1qiw6p0/the_global_step_trap_when_using_multiple/).
- [FineWebDataModule](src/research_lib/data/fineweb_datamodule.py): DataModule handling [`kjj0/finewebedu10B-gpt2`](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2) from Hugging Face.
- [ExperimentStateCallback](src/research_lib/training/callbacks/experiment_state.py): Callback to track the latest checkpoint path for automated experiment resumption.

### Utilities
- [Optimizer Scheduling Suite](src/research_lib/training/scheduling/README.md): Flexible parameter scheduling system.
    - *Reference:* [r/MachineLearning post](https://www.reddit.com/r/MachineLearning/comments/1rfer1y/p_implementing_better_pytorch_schedulers/) (#2 post of the day: 27-02-26)
- [Parameter Partitioning Helpers](src/research_lib/training/param_utils.py): Filtering and grouping via PEFT-style pattern matching.

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
