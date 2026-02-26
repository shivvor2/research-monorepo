# Scheduling Submodule

This submodule provides a flexible system for scheduling optimizer parameters (e.g., learning rate, momentum) during training. It supports custom schedules, presets like warmup-stable-decay, and wrappers for cyclic patterns.

This suite is designed for PyTorch optimizers, it should be stateless where possible, picklable for checkpointing, and follows a layered architecture for ease of use.

Everything here is well-tested via `tests/test_scheduling.py`.


## Directory Structure

```
└── ./
    └── src
        └── research_lib
            └── training
                └── scheduling
                    ├── tests
                    │   ├── __init__.py
                    │   └── test_scheduling.py
                    ├── __init__.py
                    ├── scheduler.py
                    ├── schedules.py
                    ├── utils.py
                    ├── validation.py
                    └── wrappers.py
```

## Key Features

- **Primitives**: `ParamSchedule` for custom functions.
- **Preset Schedule**: Subclasses of `ParamSchedule` provided for common schedule patterns e.g. `WarmupStableDecaySchedule`.
- **Runtime**: `ParamScheduler` binds schedules to an optimizer and handles stepping/checkpointing.
- **Utilities**: Low-level tools for applying schedules and querying optimizer state.
- **Validation**: Checks like `validate_schedule` to ensure schedules behave correctly.
- **Wrappers**: Cyclic, decaying, and warm restart patterns via `wrappers` submodule.

## Usage Examples

### Basic Custom Schedule
Define and use a simple linear decay for learning rate:

```python
from research_lib.training.scheduling import ParamSchedule

def linear_decay(step: int, total_steps: int) -> float:
    return 1.0 - (step / total_steps) * 0.9  # Decays from 1.0 to 0.1

lr_schedule = ParamSchedule(param_name="lr", schedule_fn=linear_decay)
value = lr_schedule(step=500, total_steps=1000)  # Returns 0.55
```

### Preset Schedules

Common patterns provided as `ParamSchedule` subclasses e.g. `WarmupStableDecaySchedule` for LR with cosine decay

```python
from research_lib.training.scheduling import WarmupStableDecaySchedule

lr_schedule = WarmupStableDecaySchedule(
    param_name="lr",
    warmup_steps=100,
    cooldown_frac=0.5,
    min_value=0.0,
    max_value=1.0,
    decay_type="cosine",
)
```

### Runtime Scheduling
Bind to an optimizer and step during training:

```python
import torch
from torch.optim import AdamW
from research_lib.training.scheduling import ParamScheduler, WarmupStableDecaySchedule

model = torch.nn.Linear(10, 10)
optimizer = AdamW(model.parameters(), lr=0.1)

lr_schedule = WarmupStableDecaySchedule(param_name="lr", max_value=0.01)
scheduler = ParamScheduler(
    optimizer=optimizer,
    global_schedules=[lr_schedule],
    total_steps=10000,
)

# In training loop
for step in range(10000):
    # ... compute loss, optimizer.step()
    scheduler.step()  # Updates optimizer params

# Checkpointing
state = scheduler.state_dict()  # {'step_count': ...}
scheduler.load_state_dict(state)
```

### Per-Group Overrides
Different schedules for optimizer param groups:

```python
optimizer = AdamW([
    {"params": [model.weight], "lr": 0.1},
    {"params": [model.bias], "lr": 0.01},
])

slow_lr = WarmupStableDecaySchedule(param_name="lr", max_value=0.001)
scheduler = ParamScheduler(
    optimizer=optimizer,
    global_schedules=[lr_schedule],
    group_overrides={1: [slow_lr]},
    total_steps=10000,
)
```

### Cyclic Wrapper
Repeat a base schedule cyclically:

```python
from research_lib.training.scheduling import wrappers as sw

base_schedule = WarmupStableDecaySchedule(max_value=1.0, warmup_frac=0.1, cooldown_frac=0.0)
cyclic_fn = sw.Cyclic(base_schedule.schedule_fn, cycle_steps=1000, skip_on_restart=100)

lr_schedule = ParamSchedule(param_name="lr", schedule_fn=cyclic_fn)
```

### Validation
Check a schedule for correctness:

```python
from research_lib.training.scheduling import validate_schedule, check_non_negative, check_monotonic_non_increasing

validate_schedule(
    lr_schedule,
    total_steps=10000,
    single_checks=[check_non_negative],
    sequence_checks=[check_monotonic_non_increasing],
)
```

### Checkpointing

When loading checkpoints containing schedule objects, use `weights_only=False` in `torch.load()` or `trainer.fit()`.

---

For more details, see docstrings in `__init__.py`, `schedules.py`, etc.
