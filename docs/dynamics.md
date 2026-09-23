# Causal dynamics and action-response experiments

The `glucoalg-dynamics` command, also available as `python -m glucoalg.dynamics`,
provides `collect`, `branches`, `train`, `forecast`, `response`, and experimental
`rollout` subcommands.
Use an installed GlucoSim checkout and a compatible 14-observation categorical
policy checkpoint with its original `config.json`. No policy or predictor
weights are bundled with this repository.

The examples create new data and artifacts. Choose the patient, seeds, episode
budget, output paths, and split protocol before running them. They are a workflow
example, not evidence that a predictor improves forecasting or shielding. The
[predictive-shield guide](predictive-shield.md) explains the intervention limits.

## Collect independent whole episodes

Assign train, validation, and test episodes explicitly. A patient may occur in
multiple splits with fresh reset seeds when testing within-patient prediction;
reserve separate patients when testing patient transfer. Do not randomly split
overlapping windows from the same episode.

This example collects six one-day training episodes with a stochastic policy and
10% uniform recommendation exploration:

```bash
python -m glucoalg.dynamics collect \
  --checkpoint /absolute/policy/torch_save/epoch-976.pt \
  --config /absolute/policy/config.json \
  --simulator-root "$GLUCOSIM_ROOT" \
  --output-dir ./causal-data/train-adolescent001 \
  --patient-type t1d --patient-name 'adolescent#001' --split train \
  --episodes 6 --horizon-days 1 \
  --env-seed-base 700 --action-seed-base 700 --exploration-seed-base 10700 \
  --action-mode stochastic --exploration-probability 0.1
```

Repeat with `--split validation`, a new output directory, `--episodes 2`, and
fresh seed bases such as 800/800/10800. Reserve test episodes similarly with
`--split test` and seeds such as 900/900/10900. Keep policy hashes, action mode,
exploration probability, controller interval, and simulator version consistent
across the forecast experiment. Full patient names are required, such as
`adolescent#001`; the supported diabetes types are `t1d`, `t2d`, and
`t2d_no_pump` (`t2dnp` is accepted as an alias).

`--env-seed-base` controls explicit `env.reset(seed=...)` calls. Policy sampling
is reseeded separately by `--action-seed-base`; exploration uses an independent
`numpy.PCG64` generator seeded by `--exploration-seed-base`. Each seed increments
by episode index. The collector draws a policy proposal every step, then replaces
it with the declared probability by independent uniform bolus and meal indices.
`--action-mode deterministic --exploration-probability 0` disables policy sampling
and random exploration; the simulator still has its own randomness.

The interval is five minutes, and the requested horizon must be at least one day
in whole controller steps. Early termination is preserved and reported. The
collector closes the environment on failure and refuses to overwrite outputs.

## Episode format and causal timing

A collection directory contains `collection.json` and `episode-000000/`,
`episode-000001/`, and so on. Each episode has compressed `episode.npz`,
`metadata.json`, and a SHA256 `manifest.json` written last. The loader requires
all three and uses `allow_pickle=False`.

| Array | Shape | Meaning |
| --- | --- | --- |
| `observations` | `[N+1,14]` | Reset observation, then each returned outcome |
| `recommended_actions` | `[N,2]` | Proposed bolus and meal indices, each 0–4 |
| `executed_actions` | `[N,2]` | Accepted controller indices; rejected components become zero |
| `accepted` | `[N,2]`, boolean | Authoritative bolus/meal acceptance flags |
| `rewards`, `costs` | `[N]` | Raw simulator outputs, including terminal penalties |
| `terminated`, `truncated` | `[N]`, boolean | Completion flags; no later rows are allowed |

Executed indices are accepted recommendation levels, not delivered physical
amounts. GlucoSim scales accepted levels by patient parameters, adds dose/meal
noise, and separately handles basal insulin and autonomous meals. Missing or
nonboolean acceptance evidence is an error. The recommendation buffer is never
mutated into an execution buffer.

Metadata records schema version, patient identity, all three seeds, split,
controller interval, requested horizon, behavior and continuation descriptions,
checkpoint/config hashes, simulator commit and content hash, runtime/source
provenance, cost, termination, and coverage. Safety summaries use **post-step
measured CGM**, excluding reset and including the final outcome. This is distinct
from the general evaluator's explicit historical `--glucose-source legacy` mode.

Both training and online inference use the same raw window helper. With model
history length L and forecast horizon H:

```text
input row t = (observation[t], recommendation[t])
input window = rows t-L+1 through t
target[t,h] = CGM[t+h] - CGM[t], h=1..H
```

CGM is observation feature zero; it is not latent simulator blood glucose.
Inputs contain 14 raw features and two five-way recommendation one-hots. Future
acceptance and executed actions are diagnostics, not model inputs. Factual
training drops warmup and incomplete horizons without crossing episode boundaries.
Branch training retains each observed partial horizon; missing outcomes never
become zero targets.

BA-NODE fits its per-query representation from E context windows with anchors
`t-H-E+1` through `t-H`. Each entire context target is already observed at query
time t. This requires **R=L+H+E-2 completed transitions**, which is the shield's
raw-history requirement; it differs from the model input history L.

## Train and load an artifact

Pass episode directories explicitly; shell globs below expand whole episodes:

```bash
python -m glucoalg.dynamics train \
  --train ./causal-data/train-adolescent001/episode-* \
  --validation ./causal-data/validation-adolescent001/episode-* \
  --simulator-root "$GLUCOSIM_ROOT" --output-dir ./predictors/ba-node-seed1101 \
  --seed 1101 --epochs 20 --batch-size 16 --learning-rate 0.001 \
  --history-length 12 --horizon-steps 12 --context-size 5 \
  --n-basis 3 --hidden-size 32 --torch-threads 4 \
  --transfer-policy same-cohort
```

Add further patient episode directories to `--train` and `--validation` as
needed. Training rejects shared episode IDs, patient/reset-seed identities, or
data hashes across the two splits, and mismatched collection protocols. It
requires the simulator content recorded by the episodes. Keep outputs outside
source-package directories and leave the recorded source unchanged during a run.

Choose the prediction path explicitly when comparing models:

| `--prediction-mode` | Prediction path |
| --- | --- |
| `context_only` (default) | Existing BA-NODE basis expansion fitted to observed context deltas |
| `direct` | Flattened causal input → two SiLU hidden layers → cumulative deltas; no function encoder |
| `residual` | Direct prediction plus BA-NODE fitted to observed context residuals |

`--direct-hidden-size` defaults to 64. Residual training differentiates through
the context subtraction and adds direct-head factual MSE with
`--residual-direct-weight 1`. The original context-only artifact/state layout is
preserved. All modes use the same causal query set and conservative warmup R.
A context-only fit with exactly zero context targets necessarily returns zero
deltas; a direct path removes that algebraic restriction. This does not establish
action sensitivity or accuracy on nonflat real contexts.

Training uses CPU PyTorch; BA-NODE modes use eager RK4 through registered dynamics
modules. Gradients reach the optimizer's dynamics parameters. Observation means
and scales are fitted once from training pre-observations; one-hot channels are
unchanged. Constant or near-constant observation channels use scale one. Targets
are divided by the training CGM scale without mean subtraction. No validation or
test episode fits normalization, and inference does not load datasets.

Without branch data, the first checkpoint with the lowest validation MAE is saved.
With branch data, selection uses the response-aware score below. `--max-train-windows N`
optionally caps training windows using deterministic evenly spaced selection;
it is a debugging/budget option and is recorded. `--transfer-policy
seen-patients-only` restricts inference to fitted patient identities.
`same-cohort` explicitly permits the same diabetes type and trained cohorts,
using pooled training statistics and only the new episode's observed past for
representation fitting. Permission to transfer is not evidence of accuracy on
an unseen patient.

The output contains `weights.pt`, `artifact.json`, `artifact.sha256`,
`metrics.json`, `epoch_metrics.json`, `split_check.json`, and `run.json`.
Artifacts retain model configuration, training statistics, encoding/units,
patient scope, continuation protocol, data identities, report hashes, and
source provenance. Loading checks dimensions, encodings, statistics, and hashes:

```python
from glucoalg.dynamics.model import load_predictor

adapter = load_predictor("./predictors/ba-node-seed1101", device="cpu")
```

The adapter returns absolute point CGM forecasts in mg/dL. It does not expose
calibrated uncertainty or load a dataset at inference. See the predictive guide
for the explicit `ForecastRequest`/`Shield` sequence.

## Collect and learn paired recommendation responses

Factual episodes can contain few accepted actions. The separate `branches`
collector creates a fixed no-action prefix and restores the complete simulator,
vector buffers, and random streams at each predeclared anchor. It tries all 25
first recommendations, then follows the declared policy/exploration continuation.
A duplicate `[0,0]` branch must reproduce its trace and final state exactly.
Branches never feed back into the prefix. The simulator's basal insulin, autonomous
meals, dose noise, and acceptance gates remain active.

```bash
python -m glucoalg.dynamics branches \
  --checkpoint /absolute/policy/torch_save/epoch-976.pt \
  --config /absolute/policy/config.json --simulator-root "$GLUCOSIM_ROOT" \
  --output-dir ./branch-data/train-2000 --split train \
  --patient-type t1d --patient-name 'adolescent#001' \
  --env-seed 2000 --action-seed 2000 --exploration-seed 12000 \
  --anchors 27 51 75 99 123 147 171 195 219 243 267 \
  --history-length 12 --horizon-steps 12 --context-size 5
```

Choose separate reset seeds for validation and test. Every anchor and sibling
branch from a patient/reset episode belongs to one partition. Rejected actions,
no-ops, early endings, and unreachable anchors remain recorded; future acceptance
is a diagnostic label, never an input or a reason to select training examples.
Each `anchor-*` directory contains ragged `data.npz`, `metadata.json`, `audit.json`,
and a hash manifest. The collection records its entire requested anchor schedule
and final prefix. Loading verifies restoration evidence, duplicate controls,
prefix consistency, schedule completeness, and coverage counts.

To add branch supervision, extend a training command with:

```bash
  --prediction-mode direct --direct-hidden-size 64 --batch-size 32 \
  --train-branches ./branch-data/train-* \
  --validation-branches ./branch-data/validation-* \
  --protocol-file ./experiment/PROTOCOL.md \
  --protocol-clarifications ./experiment/IMPLEMENTATION_CLARIFICATIONS.md
```

The clarification file is optional. Training records and rechecks protocol and
input hashes. One training prefix group is sampled uniformly per factual batch;
all its candidates contribute absolute and paired-response MSE, each with weight
one and the same factual TRAIN-only CGM scale. Branch rows do not fit normalization.
Losses average observed horizons within each candidate, candidates within each
anchor, then anchors. Paired responses subtract the `[0,0]` branch and use only
horizons observed in both branches. The validation selection score is
`factual MAE / max(1, persistence MAE) + response MAE / max(1, zero-response MAE)`.
Save the first minimum; select architecture/seeds using validation only.

After sealing model selection, score complete test collections:

```bash
python -m glucoalg.dynamics response \
  --artifact ./predictors/direct-seed3101 \
  --collections ./branch-data/test-* --output-dir ./response-results/direct3101
```

The response evaluator reports absolute and paired errors, accepted pure-bolus
and pure-meal strata, rejected/no-op coverage, per-patient results, and latency.
Coverage includes all planned anchors, even those the prefix never reached.
`responses.npz` retains predictions, observed targets/masks, acceptance labels,
and candidate controls. Reversing candidate order must preserve forecasts;
replacing every candidate with `[0,0]` and shuffling nonzero candidate labels are
also reported. These descriptive controls have no automatic scientific pass rule.
Training's `observed_group_*coverage_fraction` describes loaded groups only;
use complete collection/response reports for planned-budget coverage.

These are forced-first-recommendation effects under the fixed continuation,
not isolated delivered-dose effects: later policy actions and acceptance can
change. Overlapping anchors are not independent experimental replicates. Evaluate
both natural factual episodes and the enriched branch distribution before using
the predictor for interventions.

## Measure held-out forecast error

Keep the final test episodes separate from fitting and checkpoint selection:

```bash
python -m glucoalg.dynamics forecast \
  --artifact ./predictors/ba-node-seed1101 \
  --episodes ./causal-data/test-adolescent001/episode-* \
  --output-dir ./forecast-results/test-seed1101
```

The validator rejects overlap with training/validation identities and mismatched
policy, behavior, continuation, or simulator provenance. It scores complete
causal queries after warmup and compares the model with fixed persistence and
OLS linear-trend baselines on the same observed history. Output includes raw
`forecasts.npz`, per-horizon MAE/RMSE by episode and patient, current-CGM and
acceptance strata, query coverage, episode safety/termination coverage, and
inference latency in `report.json`, plus `metrics.json` and `split_check.json`.

Errors are descriptive and query weighted. Overlapping windows are not
independent statistical replicates. Early termination and discarded warmup or
incomplete horizons limit which states are scored. Observational forecasting
under the recorded policy/exploration mixture does not identify the effect of
changing an action. Inspect each horizon and patient rather than treating an
aggregate error as evidence that a shield improves outcomes.

## Run experimental interventions

Use `python -m glucoalg.dynamics rollout --help` and the
[predictive-shield guide](predictive-shield.md#experimental-rollouts) for paired
unshielded, rule-based, and predictive conditions. This is a separate diagnostic
path; `glucoalg-eval-grid` retains its unshielded/rule-based interface.

`python -m glucoalg.dynamics tune` provides a portable validation study for
forecast-penalty scales. It reuses explicit predictor artifacts and their bound
policy checkpoint, without retraining or selecting a policy. Read the
[shield-tuning guide](shield-tuning.md) before planning the grid or interpreting
a selected scale. Fresh confirmation remains a separate experiment.

## Legacy scripts and tests

The numbered scripts `1.collect_transition.py`, `2.train_dynamics_predictor.py`,
and `3.evaluate_dynamics_predictor.py` remain historical research entry points.
Their old `saved_files/env_transitions/...` and `saved_files/dynamics_predictor/...`
layouts do not use this causal schema. In particular, legacy post-action inputs
already contain the first target outcome. Do not relabel those files or use old
weights as a new artifact; recollect and retrain with explicit provenance.
The new workflow supports context-only BA-NODE, a direct MLP, and their residual
combination. It does not import legacy ITF/NODE artifacts.

```bash
python -m glucoalg.dynamics train --selftest
python -m pytest -q tests/test_dynamics_data.py tests/test_dynamics_collect.py \
  tests/test_dynamics_model.py
```

The self-test and unit tests check timing, split/normalization failures, real
BA-NODE parameter gradients, artifact integrity, and reset/RNG behavior. They do
not demonstrate forecast or intervention benefit. Real simulator experiments
need separate protocol, artifacts, and outcome reporting.
