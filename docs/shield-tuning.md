# Tune forecast penalties with a fixed policy

`glucoalg-dynamics tune` creates a categorical validation study for the
predictive shield. Each study fixes its policy, predictor artifacts, patients,
reset seeds, action modes and exploration probability. Execution runs locally;
an external scheduler can launch individual planned tasks through the same CLI.
Cluster names and Slurm resources are not part of the scientific specification.

## What the scale changes

Let `S` be the original static mask and `L` the original full predictive mask.
The applied mask is `S + scale * (L - S)`:

| Scale | Action effect |
| --- | --- |
| `0` | Exactly the static rule; forecasts remain available as telemetry |
| `1` | Original predictive behavior |
| Between `0` and `1` | Weaker incremental forecast penalties |
| Greater than `1` | Stronger incremental forecast penalties |

The static cap, critical rescue, thresholds, candidate set and cooldown remain
fixed. A forecast-triggered cap restoration belongs to the scaled contribution.
A forecast flag already covered by the same static penalty has zero incremental
contribution, so increasing the scale does not strengthen that component.
`predictive_static` is a separate control that skips forecast calls entirely.

All scales must be finite and nonnegative. Run a single explicit setting with
`glucoalg-dynamics rollout ... --condition predictive --forecast-penalty-scale 0.3`.
The report records the scale, while decision records contain the mask actually
applied. Finite penalties still permit sampling penalized actions, and later
exploration can replace the recommendation.

To sweep **absolute incremental forecast-penalty magnitudes** with the fixed
legacy magnitude of `10`, divide each requested magnitude by `10`:

| Absolute magnitude | `forecast_penalty_scale` |
| --- | --- |
| `1` | `0.1` |
| `5` | `0.5` |
| `10` (default) | `1` |
| `15` | `1.5` |
| `20` | `2` |

For example, absolute magnitude `15` uses
`--condition predictive --forecast-penalty-scale 1.5`. This changes an
incremental forecast contribution of `-10` to `-15`; static and rescue
contributions remain fixed. A component already covered by the static rule
still has no additional forecast contribution. The scale is therefore not
the absolute magnitude of the entire applied mask. Keep a separate zero-scale
control when comparing positive magnitudes, and report whether a positive
candidate passes the safety gates even if it has the highest TIR.

## Plan and execute

Copy `configs/tuning/shield_penalty_grid.json` and replace its absolute placeholder
paths with your policy, predictor, simulator and protocol files. The example
contains three predictor seeds, three one-day validation cases on seen patient
`#001`, two action modes and scales `[0, 0.1, 0.5, 1, 1.5, 2]`: **96 validation
tasks**. The positive scales correspond to absolute incremental magnitudes
`1, 5, 10, 15, 20`. The zero baseline
uses one explicitly declared predictor artifact and is shared across predictor
seeds for action-effect comparison. It is not three independent episodes.
Reserve unseen patients `#002`–`#010` and seven-day episodes for the final
generalization comparison; do not select penalties using that test set.

```bash
python -m glucoalg.dynamics tune plan \
  --spec /absolute/experiment/shield_penalty_grid.json \
  --output ./studies/cpo-shield
python -m glucoalg.dynamics tune run \
  --plan ./studies/cpo-shield --task-id TASK_ID_FROM_PLAN
python -m glucoalg.dynamics tune summarize --plan ./studies/cpo-shield
```

Planning records the complete task grid and input/source hashes before running
anything. Use the task IDs in `plan.json`. Each task has its own output directory;
changing code, artifacts or the scientific specification requires a new plan.
All required results must be complete and match the plan before selection.
A physiological early termination is a retained outcome, even when the command
itself completed successfully. Failed or missing tasks cannot be silently dropped.

An external evidence runner can supply `summarize --results-manifest FILE` with
a JSON object mapping every task ID to its verified rollout artifact directory.
The same identity, hash, configuration and trace checks apply to these artifacts.
This supports independent capture/reconstruction without embedding scheduler
behavior in the package.

The spec declares `models` as `{seed, artifact_dir}` entries and `cases` as
`{id, patient_type, patient_name, env_seed, action_seed, exploration_seed}` entries.
`zero_model_seed` identifies the shared zero baseline. `protocol_files` binds the
written method and any pre-run clarification. Optional `excluded_families` names
reset families that cannot enter the study. Predictor fitting/selection families
are checked from artifact provenance as well. Reserve confirmation cases outside
this validation plan.

## Selection rule

Each action mode selects one scale shared across all listed predictor seeds.
For every seed, an eligible setting must satisfy these comparisons with its
matched zero-scale baseline:

- No new physiological termination and no shorter observed horizon.
- No increase in observed below-54 or below-70 CGM counts on any case.
- No increase in each patient's mean observed fraction above 250.
- No decrease in mean observed TIR across cases.
- At least 95% overall observed/requested horizon coverage.

These are conservative descriptive gates. Counting every additional low sample
can disadvantage a setting that survives longer; it is not a statistical
noninferiority test. A newly recorded physiological termination at the last
requested step still fails the termination comparison.

The score is in-range samples divided by **requested** steps, averaged equally
across cases within each model and then across models. It is separate from TIR
on observed samples. Missing tails do not enter its numerator and are not filled
with invented glucose values. Among eligible settings within 0.1 percentage
points of the maximum score, select the lowest scale. Floating arithmetic uses
`tolerance=1e-12`; glucose-event counts are compared exactly.

Zero remains the fallback when no positive setting wins. A fallback that itself
fails coverage is reported without an eligible improvement; it is not promoted
as a passed safety screen. Simulator reward and cost are reported separately
and do not rank settings: both can depend on requests that were rejected.

## Policy quality and fresh confirmation

Predictor artifacts bind their policy checkpoint and configuration hashes.
Evaluate each policy checkpoint independently, with a corresponding trained
and validated predictor artifact. Swapping policies under
an existing artifact would change the declared continuation law.

Profile the unshielded policy with the current normalizer before interpreting
shield effects. A useful comparison crosses categorical versus deterministic
actions with exploration enabled versus disabled, holding reset cases fixed.
Changing both sampling and exploration at once cannot identify either effect.
Predictor/rollout behavior mismatches remain explicit in reports; penalty tuning
does not repair or revalidate that forecast continuation mismatch.

Seal the selected scale before evaluating fresh patient/reset cases. Compare
true no-shield behavior, zero/static, the default scale and the selected scale,
retaining every predictor seed and the observed horizon. The portable command
performs validation selection; the confirmation protocol and its benefit
threshold must be declared separately. Read TIR alongside glucose extremes,
low/high counts, termination, coverage, accepted actions and intervention
attribution. A changed logit or rejected request alone is not a physiological
improvement.
