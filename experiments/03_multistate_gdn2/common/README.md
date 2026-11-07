# common: the project package shared by the pilots

A small package that both pilots of the multi-state GDN-2 project import,
next to the general `research_lib` and in front of each pilot's own
protocol. It is **shared by this project's pilots only** — general
machinery lives in `research_lib`.

What it holds:

- `launch.py` — the Hydra launch helpers (`job_num`, `stop_requested`,
  `begin_run`) on top of the general run control.
- `run_id.py` — the run-ID builder (with the task-key option) and the
  identity exclusion set `HASH_EXCLUDE`.
- `layout.py` — the `runs/<task>/<run_id>.json` layout and the project's
  artifact contract (an `ArtifactLayout` for the general pull), plus the
  config → W&B mapping.
- `pull.py` — a pilot's thin `03_pull.py` entry point over the general
  pull.
- `records.py` — the project's `flatten` for `load_records`.
- `analysis.py` — the shared report pieces (gap column, paired gap,
  Δ_int tables) over the general statistics.
- `protocol.py` — the grid constants both pilots share.
- `registries.py` — the task config-type and harness registries.

A pilot's scripts and its `conftest.py` add this package's parent (the
project folder) to `sys.path`, the same way they add the library's `src`.
