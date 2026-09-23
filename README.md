# GlucoAlg

Safe reinforcement learning training, checkpoint evaluation, and categorical
hyperparameter optimization for [GlucoSim](https://github.com/safe-autonomy-lab/GlucoSim).
The repository includes modified OmniSafe algorithms with categorical actors,
FunctionEncoder/BA-NODE dynamics predictors, and optional evaluation shields.

## Installation

Python 3.10 or newer is required. Install GlucoSim first, then this repository
in a fresh environment:

```bash
git clone https://github.com/safe-autonomy-lab/GlucoSim.git
git clone https://github.com/safe-autonomy-lab/GlucoAlg.git
python -m pip install -e ./GlucoSim
python -m pip install -e './GlucoAlg[test]'
cd GlucoAlg
export GLUCOSIM_ROOT="$(realpath ../GlucoSim)"
```

Do not separately install upstream OmniSafe or FunctionEncoder: their modified
sources are included here. Install the base GlucoSim package, without its
`[omnisafe]` extra, to avoid installing another copy of OmniSafe.

The compatibility layer supports GlucoSim revision
`ddd674e35a32a3a6406a0c2d5e7bf3c72f56dfec`. To use that revision, run
`git -C ../GlucoSim checkout ddd674e35a32a3a6406a0c2d5e7bf3c72f56dfec`.
Record the simulator revision with each experiment.

`--simulator-root /absolute/path/GlucoSim` or `GLUCOSIM_ROOT` selects a source
checkout explicitly. Otherwise, the installed simulator is used. Training and
evaluation record the actual simulator path and commit. This prevents a stale
editable install from silently standing in for the requested simulator.
The packaged `glucobench` compatibility alias handles legacy self-imports inside
that GlucoSim revision; application code uses `glucosim`.

## Simulator contract

The [GlucoSim README](https://github.com/safe-autonomy-lab/GlucoSim#quickstart)
and `glucosim.diabetes_cmdp.DiabetesEnvs` are the source of truth:

| Property | Supported behavior |
| --- | --- |
| Environment IDs | `t1d-v0`, `t2d-v0`, `t2d_no_pump-v0` |
| Patients | `adolescent`, `adult`, or `child`, each with IDs `#001`–`#010` |
| Observations | 14 values; categorical policy checkpoints must match this width |
| Actions | `MultiDiscrete([5, 5])`: bolus level and meal level |
| Step output | `(observation, reward, cost, terminated, truncated, info)` |
| CMDP tensors | Batched even for one environment |
| Controller interval | Five minutes in the workflows below |
| Episode duration | At least one day; training defaults to one, evaluation to seven |

Exercise is not an action in this public interface. Reward, safety cost and
glucose metrics are different measurements. GlucoSim is a research simulator;
its outputs are not validated clinical treatment recommendations.

## Train a policy

```bash
python -m glucoalg.train \
  --algo CPO --env-id t1d-v0 --cohort adolescent --seed 100 \
  --device cpu --vector-env-nums 1 \
  --total-steps 2000000 --steps-per-epoch 2048 \
  --cost-limit 100 --log-dir ./runs
```

`python run.py ...` remains supported. The installed command is
`glucoalg-train`. Supported diabetes baselines are `PPOLag`, `TRPOLag`, `CUP`,
`CPO`, `FOCOPS`, `RCPO`, `PCPO`, and `OnCRPO`. The vendored off-policy algorithms
need a discrete-action adaptation before use with GlucoSim and are not exposed
by this CLI.

Training uses `<cohort>#001`. Defaults are actor learning rate `1e-5`, critic
learning rate `5e-5`, device `cuda:0`, and four environments. The example selects
CPU and one environment explicitly. JAX defaults to CPU unless a backend is
explicitly selected. Learning-rate and constraint settings should be tuned for
the selected algorithm.

Useful options include `--actor-lr`, `--critic-lr`, `--entropy-coef`,
`--lambda-lr`, `--lagrangian-multiplier-init`, `--batch-size`, and `--target-kl`.
The older `--cost_limit` and `--target_kl` spellings remain accepted.
W&B is disabled by default; `--use-wandb`, `--use-wandb True`, and
`--use-wandb False` are parsed explicitly.

Preview the resolved configuration without initializing the simulator:

```bash
python -m glucoalg.train --algo PPOLag --actor-lr 3e-4 \
  --device cpu --vector-env-nums 1 --dry-run
```

Use `--set PATH=JSON` for validated nested overrides, for example
`--set algo_cfgs.update_iters=10`. Unknown flags and configuration paths fail
instead of being ignored. Nondefault values for the old no-op `--safety-bonus`
and `--penalty-type` flags now fail explicitly.

OmniSafe writes `config.json`, `runtime.json`, `progress.csv`, and `torch_save/`
under a timestamped run directory inside `--log-dir`. The trainer executes only
full epochs: 2,000,000 requested steps with 2,048 steps per epoch produces
976 epochs and 1,998,848 steps. The progress log counts epochs from zero;
checkpoint filenames count completed epochs, ending at `epoch-976.pt`.

The supported GlucoSim CMDP constructor ignores its `seed` argument; actual
simulator reseeding happens through `reset(seed=...)`. Training preserves the
existing adapter behavior: its seed controls policy and training randomness,
but is not a guarantee of a separately seeded JAX environment trajectory.
Evaluation explicitly reseeds each episode. Record both policy and environment
seeding when comparing runs.

## Hyperparameter optimization

The repository-owned workflow supports exhaustive categorical grids and
reproducible random subsets, with complete multi-seed scoring and explicit
failed-job retries. Planning does not start training.

```bash
python -m glucoalg.tuning plan \
  --spec configs/tuning/ppolag_actor_critic_grid.json \
  --simulator-root "$GLUCOSIM_ROOT" --output ./studies/ppolag-grid
python -m glucoalg.tuning run --plan ./studies/ppolag-grid --all --dry-run
python -m glucoalg.tuning run --plan ./studies/ppolag-grid --all
python -m glucoalg.tuning summarize --plan ./studies/ppolag-grid --plot
```

The example crosses actor learning rates `{3e-4, 3e-5}` with critic learning
rates `{1e-4, 5e-4, 5e-5}`, using seeds 100–102. This is **18 full training jobs**;
change the spec before planning if a smaller budget is intended.

Selection maximizes the mean training return among configurations whose mean
training cost meets the fixed limit. Each seed contributes equally, using its
last 20% of completed epochs. Missing, incomplete, failed or non-finite results
cannot win; if nothing is feasible, there is no winner. A selected configuration
is provisional until confirmed with fresh training seeds and a separate patient
evaluation protocol. Training return/cost selection is not a TIR comparison.

See [the tuning guide](docs/tuning.md) for the spec schema, protocol record,
resume/retry behavior, source provenance, outputs and optional Slurm script export.
Study specifications contain the experiment settings; separate execution profiles
contain cluster resource requests. Local execution requires no scheduler.

## Evaluate checkpoints

Use explicit checkpoint/configuration paths and a new output directory:

```bash
python -m glucoalg.eval_grid \
  --checkpoint /absolute/run/torch_save/epoch-976.pt \
  --config /absolute/run/config.json \
  --output-dir ./evaluations/unseen-stochastic \
  --patient-type t1d --algorithm CPO --train-seed 100 \
  --patients 'adolescent#002' 'adolescent#003' \
  --episodes 10 --horizon-days 7 --eval-seed-base 22 \
  --action-mode stochastic --cost-limit 100
```

`glucoalg-eval-grid` provides the same command. Use patients `#002`–`#010` for
nine unseen same-cohort patients when training used only `#001`. Keep model selection patients
and final reporting patients separate. An output manifest records checkpoint
and configuration hashes, patient coverage, episode seeds, horizon, action
mode, simulator provenance, TIR, Risk Index, costs and termination outcomes.
Missing/non-finite metrics or incomplete patient coverage fail result validation.

The default action mode is **stochastic**. Use `--action-mode deterministic` in
a separate output directory for an explicit deterministic comparison. Episode seeds are `22 + episode_index`
by default and are shared across patients for a paired protocol.

Glucose metrics describe the observed trace. Read them with episode length,
horizon coverage, termination/truncation and cost, especially when an episode
ends early; a high TIR alone does not establish safety. The evaluator restores
saved observation normalization, including standard deviation and clipping.
Checkpoints with `obs_normalize: false` do not use that transform. Check the
saved configuration before comparing checkpoints.

`--glucose-source legacy` preserves the historical pre-action observation
alignment. Use `--glucose-source post-step` to score the observation returned by
each action instead. The manifest records this choice; these are separate
evaluation protocols.

The legacy interface is retained:

```bash
python eval_run.py t1d CPO 'adolescent#002' 100 --epoch 976 --num-episodes 10
```

It resolves checkpoints from `saved_models/<type>/<algorithm>/seed<seed>/` or
the older cohort subdirectory layout. Prefer explicit paths for new work.
Use `--help` for trace/plot controls and optional rule-based shielding.
The general checkpoint evaluator supports unshielded and rule-based modes.
The general evaluator accepts `--shield-type none` or `rule_based`. The legacy
`--shield` shorthand fails with instructions for using predictive rollouts.
`--logit-penalty` is a nonnegative magnitude (default `10`) in both interfaces;
negative values are rejected because they would boost penalized actions.
Predictive shielding has a separate experimental workflow with explicit causal
data, an explicit predictor artifact, and recorded policy/simulator provenance. Read the
[predictive-shield guide](docs/predictive-shield.md) before using its rollout
command. The implementation and its software checks do not establish predictive
benefit, calibrated uncertainty, or improved safety.

## Dynamics and development

The causal dynamics workflow collects whole episodes and restored-state action
branches, trains direct, context-only BA-NODE, or residual predictors,
measures held-out forecast error against persistence and linear-trend baselines,
and runs explicit experimental intervention rollouts:

```bash
python -m glucoalg.dynamics collect --help
python -m glucoalg.dynamics branches --help
python -m glucoalg.dynamics train --help
python -m glucoalg.dynamics forecast --help
python -m glucoalg.dynamics response --help
python -m glucoalg.dynamics rollout --help
python -m glucoalg.dynamics tune --help
```

`glucoalg-dynamics` provides the same subcommands after installation. The
[dynamics guide](docs/dynamics.md) describes commands, episode splits, artifact
hashes, causal windows, and the distinction between recommendations and accepted
controller levels. Legacy post-action datasets are incompatible with the new
schema and must be recollected for this workflow. No predictor weights are
bundled. Deterministic/stochastic comparisons and different horizons remain
separate protocols.

For a fixed predictor and policy, `glucoalg-dynamics tune` plans a categorical
forecast-penalty study, runs individual validation cases, and selects a scale
using glucose and horizon-coverage gates. It keeps static and rescue rules
fixed. See [shield tuning](docs/shield-tuning.md) for the scale definition,
absolute-magnitude conversion, validation-only selection, and fresh confirmation
requirements. Absolute incremental magnitudes `1, 5, 10, 15, 20` correspond to
scales `0.1, 0.5, 1, 1.5, 2` with the fixed legacy magnitude `10`. This workflow
is separate from policy-training hyperparameter optimization.

```bash
python -m pytest -q
python -m pip wheel --no-deps --wheel-dir dist .
```

The tests cover configuration handling, simulator import compatibility, batched
CMDP wrappers, tuning validation/aggregation/retry behavior, and evaluation
gates, causal data timing, BA-NODE gradients, and predictor artifact boundaries.
Simulator integration tests require the selected GlucoSim installation.
The test suite does not run full policy training.

Credit to [OmniSafe](https://github.com/PKU-Alignment/omnisafe) and
[FunctionEncoder](https://github.com/tyler-ingebrand/FunctionEncoder) for the
underlying frameworks. GlucoAlg's modifications include categorical actors and
related discrete-action support, BA-NODE, and diabetes evaluation workflows.
Vendored source notices remain in place.

## Citation

```bibtex
@inproceedings{kwon2026safetygeneralizationdistributionshift,
  title={Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed},
  author={Minjae Kwon and Josephine Lamp and Lu Feng},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=kSUGLBHd0T}
}
```
