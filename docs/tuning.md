# Categorical hyperparameter optimization

`glucoalg.tuning` plans and runs experiments, scores complete training-seed
groups, and selects the highest-return configuration that satisfies a cost
limit. It supports exhaustive grids and reproducible random subsets of explicit
categorical levels. It needs no separate optimization service or Optuna install.
The workflow uses POSIX advisory file locks; run it on Linux or another POSIX
host with working locks on the study filesystem.

## Create and inspect a study

```bash
python -m glucoalg.tuning validate \
  --spec configs/tuning/ppolag_actor_critic_grid.json
python -m glucoalg.tuning plan \
  --spec configs/tuning/ppolag_actor_critic_grid.json \
  --simulator-root /absolute/path/GlucoSim --output ./studies/ppolag-grid
python -m glucoalg.tuning run --plan ./studies/ppolag-grid --all --dry-run
```

Validation resolves configurations through the training CLI without initializing
PyTorch, JAX or GlucoSim. Planning writes the normalized spec, `plan.json`, its
SHA256 fingerprint, a `PREREG.md` protocol record, and an attempt ledger. Nothing
is trained or submitted at this point. Changing factors, budgets, seed lists or
runtime source requires a new study directory.

The example grid has six configurations, each trained with seeds 100–102:
actor learning rates `{3e-4, 3e-5}` × critic learning rates
`{1e-4, 5e-4, 5e-5}`. Its 18 runs each request 2M steps on CPU with one
environment. The training CLI defaults to actor learning rate `1e-5`; the spec overrides it
with the listed factors.

## Spec format

```json
{
  "study_name": "ppolag-small-grid",
  "seeds": [100, 101, 102],
  "cost_limit": 100.0,
  "search": {"mode": "grid"},
  "factors": {"actor-lr": [0.0003, 0.00003]},
  "train": {
    "algo": "PPOLag",
    "env-id": "t1d-v0",
    "cohort": "adolescent",
    "total-steps": 2000000,
    "steps-per-epoch": 2048,
    "device": "cpu",
    "vector-env-nums": 1,
    "cost-limit": 100.0,
    "use-wandb": false
  }
}
```

| Field | Meaning |
| --- | --- |
| `study_name` | Descriptive nonempty name |
| `factors` | CLI option names mapped to nonempty categorical lists |
| `train` | Fixed CLI options; cannot overlap factors |
| `seeds` | Unique training seeds; default `[100, 101, 102]` |
| `cost_limit` | Finite nonnegative selection threshold |
| `search` | `grid`, or `random_subset` with positive `size` and integer `seed` |
| `simulator_root` | Optional checkout path; otherwise supplied at planning |
| `runner` | `auto` (default), `glucoalg.train`, or compatible `run.py` |

Example random subset: `{"mode":"random_subset","size":3,"seed":7}`.
The sampler selects Cartesian indices without materializing the full grid.
Exhaustive grids over 100,000 configurations are rejected; use a subset or a
smaller space. See `configs/tuning/ppolag_random_subset.json` for another example.

Use JSON numbers for numeric flags and JSON booleans for boolean flags.
Unknown keys, duplicate levels/seeds, continuous distributions such as
`{"min": 0.00001, "max": 0.001}`, and invalid training configurations are rejected.
`total-steps` and `steps-per-epoch` must be explicit. Budget division follows
training: 2M/2048 gives 976 full epochs, with no partial final epoch.

The runner manages `seed`, `log-dir`, `simulator-root` and `dry-run`; these cannot
be overridden through factors or fixed options. The tuning runner currently
uses CPU execution; its default device and vector count are CPU and one.
Use multiple training seeds to measure variability. A one-seed debug study
provides no evidence of robustness across training seeds.

## Execute, resume and retry

```bash
python -m glucoalg.tuning run --plan ./studies/ppolag-grid --all
python -m glucoalg.tuning status --plan ./studies/ppolag-grid
python -m glucoalg.tuning run --plan ./studies/ppolag-grid --all --resume
python -m glucoalg.tuning run --plan ./studies/ppolag-grid \
  --all --resume --retry-failed
```

Local `--all` runs jobs sequentially. `--job cfg00-s100` runs one planned seed.
Each attempt gets its own output directory and log under
`jobs/<job-id>/attempt<N>/`. The subprocess uses a list of arguments and pins
`PYTHONPATH=<algorithm-source>:<simulator-source>`, JAX to CPU, and CUDA visibility
to empty. No shell expansion is applied to parameter values.

An attempt is recorded before the subprocess starts and again when it finishes.
A job lock prevents duplicate execution; different seed jobs can run in parallel.
An interrupted attempt needs `--retry-failed` and receives a new attempt number.
The old log and directory remain available. Exit status is nonzero if selected
jobs failed, including failed jobs skipped without an explicit retry.

Resume verifies recorded output hashes, complete epoch counts, finite objective
metrics and matching study/source provenance before skipping a successful job.
Finite edits to a CSV invalidate it too. Runtime source fingerprints include
uncommitted Python/configuration files. Source changes before or during a run
invalidate continuation under that plan; create a fresh study after a code fix.
This is experiment bookkeeping, not an external scientific preregistration or
an automatic approval of a hypothesis.

## Optional Slurm execution

Local execution needs no scheduler. Slurm export is a separate adapter over the
same `run --job` interface, with one planned seed job per array task. Resource
settings belong in execution profiles, separate from the scientific tuning spec.
The generic profile is empty (`configs/execution/generic.json`): omitting
`--profile` has the same effect. Partition, account, QoS, time, memory, CPU count
and exclusive allocation are left to scheduler defaults unless requested.

```bash
python -m glucoalg.tuning export-slurm \
  --plan ./studies/ppolag-grid --out ./studies/ppolag-grid/tune.sbatch
```

An optional shared-CPU example requests four CPUs, 8 GiB of memory, a 24-hour
limit and at most two concurrent tasks:

```bash
python -m glucoalg.tuning export-slurm \
  --plan ./studies/ppolag-grid --out ./studies/ppolag-grid/cpu.sbatch \
  --profile configs/execution/slurm_cpu.json
```

These JSON files are examples in the source checkout; choose a profile path
explicitly when using an installed package. Copy and adjust the profile to your
cluster limits and workload. No partition, account, QoS or exclusive allocation
is assumed.

| Profile key | JSON value / CLI override |
| --- | --- |
| `partition` | Name or comma-separated names / `--partition` |
| `account` | Accounting allocation name / `--account` |
| `qos` | Quality-of-service name / `--qos` |
| `time` | Positive numeric Slurm duration, e.g. `"24:00:00"` / `--time` |
| `cpus_per_task` | Positive integer / `--cpus-per-task` |
| `memory` | Positive integer string, optional K/M/G/T suffix, e.g. `"8G"` / `--memory` |
| `exclusive` | Boolean / `--exclusive` or `--no-exclusive` |
| `max_concurrent` | Positive integer / `--max-concurrent` |

Every key is optional. Explicit flags override the selected profile; omitted
flags preserve it. For example:

```bash
python -m glucoalg.tuning export-slurm \
  --plan ./studies/ppolag-grid --out ./studies/ppolag-grid/shared.sbatch \
  --profile configs/execution/slurm_cpu.json \
  --time 08:00:00 --cpus-per-task 2 --memory 4G --max-concurrent 4
```

`max_concurrent` limits simultaneous array tasks using Slurm's `%N` array
syntax. Memory is requested per node; a suffix-free value is in MiB. Time accepts
minutes, minutes:seconds, hours:minutes:seconds or days-hours[:minutes[:seconds]].
The adapter requires positive time and memory instead of Slurm's special zero
values. See the [Slurm options reference](https://slurm.schedmd.com/sbatch.html)
for scheduler semantics. Profiles reject unknown/duplicate keys, invalid types,
control characters and arbitrary directives. Names use letters, digits,
underscores, dots and hyphens; only partition lists permit commas.

Export embeds the effective resource settings, profile path, plan hash and
exporting Python interpreter in a script comment. It does not alter the sealed
scientific plan. Selecting a different profile does not change the experiment
definition. Source paths come from that plan, and the interpreter comes from the
Python environment used to export; these paths must also exist on worker nodes.
Activate the intended environment before exporting. Profiles do not contain
paths or shell setup commands. The script starts in the sealed source checkout
so the submission directory cannot shadow its Python package, and resets both
JAX platform settings to CPU even if the submission environment selected a GPU.

Inspect the generated script and submit it using the site's usual `sbatch`
workflow. Export never submits anything. CPU execution, source checks, resume
behavior and result validation are shared with local execution. Requesting more
CPUs changes the allocation; it does not change training parameters. Failed
attempts still require explicit `run --job JOB_ID --retry-failed` recovery.

## Score results

```bash
python -m glucoalg.tuning summarize --plan ./studies/ppolag-grid --plot
```

For each seed, the runner requires exactly one `progress.csv` with exactly the
planned number of epoch rows. Reward/cost columns must be present and finite;
cost must be nonnegative. When `Train/Epoch` is present, it must count
contiguously from zero. Missing/extra epochs, duplicate logs or changed hashes
invalidate the seed.

The score is the mean of the final 20% of epochs (floor, at least one row).
Configurations are complete only when every declared seed has valid evidence.
Each seed gets equal weight; error bars show sample standard deviations over
seed scores. Missing metrics are never replaced with zero. Synthetic test-runner
outputs are ineligible for scientific selection.

Among complete configurations whose mean cost is at most `cost_limit`, select
the highest mean return; ties break by lower mean cost, then configuration ID.
If none is feasible or complete, the report explicitly has **no winner**.

The output directory, by default `<study>/summary/`, contains:

- `summary.json`: seed-level evidence, coverage, scores, feasibility, selected
  parameters, provenance and required confirmations.
- `summary.csv`: one configuration per row, with reward/cost means and sample
  standard deviations. Blank values indicate unavailable scores.
- `tradeoff.png` with `--plot`: complete configurations with uncertainty bars
  and the cost-limit line. Incomplete configurations are excluded.

All selections remain provisional. Training return and cost are different from
TIR/Risk Index, and a winning configuration still needs fresh training-seed
confirmation and unseen-patient evaluation. Record horizon, stochastic versus
deterministic action mode, glucose alignment, cost and termination outcomes
with those evaluations.
