# Predictive shielding: explicit artifacts and experimental rollouts

The causal workflow provides fresh episode and paired-branch collection, direct
or context-adapted prediction, held-out forecast/response evaluation, and a
separate experimental intervention runner.
It uses explicit policy/predictor artifacts and patient identities. The general
`glucoalg-eval-grid` command continues to support unshielded and rule-based modes;
use `glucoalg-dynamics rollout` for predictive diagnostics.

**No predictive benefit or safety improvement has been demonstrated by the
implementation checks.** Point-forecast accuracy, action effects, intervention
behavior, and long-horizon outcomes require separate evidence. A successful
collection or training run does not establish those claims.

The [dynamics guide](dynamics.md) gives collection/training/forecast commands and
the complete artifact contract. This guide explains the predictor boundary,
intervention semantics, and interpretation of rollout results.

## Explicit predictor interface

`load_predictor(artifact_dir, device="cpu")` loads the declared model, training-fitted
normalization, feature/target encoding, patient scope, and continuation metadata.
It checks manifest and weight hashes and uses weights-only loading. Inference
never reads episode datasets or chooses an artifact from a hard-coded directory.
The adapter supports CPU inference and recomputes its representation from the
request's fully observed past.

- `PatientIdentity` contains diabetes type and a full name such as `adolescent#003`.
  Training artifacts explicitly declare seen-only or same-cohort transfer scope.
- `PredictorMetadata` declares required raw history, forecast horizon, feature
  order, action dimensions, controller interval, and continuation policy.
- `ForecastRequest` separates completed past transitions, the current raw
  observation, and candidate **recommendations**. Past recommendations are
  required by this model; executed actions cannot replace them.
- The result is `PointForecast` in mg/dL with shape `[candidates,horizon]`, or
  `ForecastUnavailable(reason)`. Column zero predicts the next five-minute
  measured-CGM observation. The adapter validates the final absolute forecast.
- `Shield` owns bounded immutable raw history and validates the call sequence.
  Reset clears history, pending transitions, cooldown state, and adapter state.

This explicit API sketch uses an already loaded artifact:

```python
import torch
from glucoalg.dynamics.model import load_predictor
from shield.predictive_shield import Shield
from shield.predictor import PatientIdentity

adapter = load_predictor("/absolute/predictor-artifact", device="cpu")
shield = Shield(
    predictor=adapter,
    patient=PatientIdentity("t1d", "adolescent#003"),
    device="cpu",
)
shield.reset()
# One environment: finite raw observation [14] and policy logits [10].
obs = torch.as_tensor(obs, dtype=torch.float32, device="cpu")
policy_logits = torch.as_tensor(policy_logits, dtype=torch.float32, device="cpu")
adjusted_logits = shield.apply(obs, policy_logits, [5, 5])
# Draw a two-index NumPy recommendation from adjusted_logits.
# The CMDP step stays batched: env.step(recommendation[None].copy()).
# Decode executed controller indices from the simulator's boolean acceptance flags.
shield.record_action(
    executed_indices,
    next_observation=next_obs,
    recommended_action=recommendation,
    accepted=(bool(bolus_accepted), bool(meal_accepted)),
)
# Next apply uses exactly next_obs, after this completed transition.
```

`apply` cannot be called twice before recording the actual outcome.
`record_action` needs the next observation as well as accepted controller indices;
a recommendation alone is not a completed transition. GlucoSim adds physical
dose/meal noise after acceptance, so these indices are not exact delivered amounts.

## Causal model and continuation meaning

The common timing contract is implemented in `glucoalg/dynamics/data.py`:

```text
input[t] = (raw observation before action t, recommendation t)
target[t,h] = CGM[t+h] - CGM[t], h=1..H
normalized target = target[t,h] / training_CGM_std
absolute forecast[t,h] = CGM[t] + prediction[t,h] * training_CGM_std
```

Measured CGM is observation feature zero, distinct from latent simulator blood
glucose. Training fits observation statistics only from training pre-observations,
and uses the same cumulative target definition online. Inputs use 14 observation
features and bolus/meal recommendation one-hots. All input rows precede their
target outcomes; there is no padding or window crossing between episodes.

For input history L, horizon H, and E representation examples, the adapter needs
R=L+H+E-2 completed transitions. Context anchors end at t-H, so their entire
outcomes are available at the current time t. Both paths call the same helper.
The adapter's declared raw-history requirement is R, while its model input
history remains L.

Collection records a fixed categorical policy and its stochastic/deterministic
mode, followed by optional independent uniform recommendation exploration.
The model learns observational trajectories under that continuation mixture,
including simulator acceptance and noise. Supplying another candidate to the
network does not establish the causal effect of changing that action. No
counterfactual outcome label is invented from the logged trajectory.

## Experimental intervention semantics

The current predictive rule retains finite component-logit penalties and a
critical-low-CGM rescue rule. It examines the top two bolus levels by default
against all five meal levels, flags point trajectories with predicted low CGM,
and penalizes affected positive bolus levels. The optional predicted-high-CGM
meal check is disabled by default. Programmatic configuration is explicit through
`PredictiveShieldConfig` and `ShieldParams`.

There is also a nonpredictive branch: measured CGM in [70, 180] mg/dL penalizes
bolus levels 2–4. With no forecast risk flags, CGM strictly inside (90, 160)
returns logits unchanged instead. Thus warmup interventions in [70, 90] or
[160, 180] can arise from this branch without a predictive forecast; they must
not be attributed to successful prediction.

This is a soft intervention, not a hard joint-action filter. Finite penalties
leave sampling probability on penalized choices. Untested bolus levels and
independent component sampling also limit what the candidate checks imply.
No calibrated confidence bound is computed; horizon variation or basis variation
is not predictive uncertainty.

Warmup or an unavailable forecast preserves the rule's nonpredictive behavior.
The critical rescue path can also bypass forecasting. Those cases are reported
and are not fail-closed safety guarantees. Their coverage must accompany outcome
metrics. Rule-based mode is a distinct condition and also needs empirical
outcome evaluation.

## Experimental rollouts

Each invocation runs one episode with an explicit artifact, policy, condition,
patient, seeds, horizon, and output directory:

```bash
python -m glucoalg.dynamics rollout \
  --artifact ./predictors/ba-node-seed1101 \
  --checkpoint /absolute/policy/torch_save/epoch-976.pt \
  --config /absolute/policy/config.json --simulator-root "$GLUCOSIM_ROOT" \
  --output-dir ./rollouts/predictive-seed1000 \
  --patient-type t1d --patient-name 'adolescent#001' --condition predictive \
  --env-seed 1000 --action-seed 1000 --exploration-seed 11000 \
  --horizon-days 1 --action-mode stochastic \
  --exclude-episodes ./causal-data/test-adolescent001/episode-*
```

Run `--condition none`, `--condition rule_based`, and `--condition predictive_static`
into separate new directories
with the same seed/horizon protocol for comparison. The artifact is required in
all conditions to bind policy, simulator, patient scope, and excluded fit/selection
seeds, including fitting/selection branch reset families. `--exclude-episodes`
additionally checks explicitly listed test episodes;
the runner cannot discover every earlier experiment automatically.
Rollout requires the canonical spelling `--patient-type t2d_no_pump`; the
`t2dnp` alias is accepted by collection and the general evaluator.

Exploration defaults to the artifact's collection probability. It occurs after
the intervention and can replace the shield-adjusted recommendation; this is
recorded. For a separate deterministic diagnostic, specify **both**
`--action-mode deterministic --exploration-probability 0`. The report labels
behavior mismatches with the collection protocol. Even with matched sampling
settings, an intervention changes continuation from the behavior policy used to
collect training data.

`predictive_static` sets `PredictiveShieldConfig(use_forecast=False)`: it runs
the predictive shield's static rules and rescue behavior without calling the
predictor. This differs from the separate `rule_based` implementation. Compare
it with `predictive` to isolate what the forecast adds to those same static rules.

`--forecast-penalty-scale` controls the incremental forecast contribution in
`predictive` mode. If `S` is the static mask and `L` the original full mask, the
applied mask is `S + scale * (L - S)`. The default `1` preserves the original
behavior; `0` applies exactly the static action rule while still collecting
forecasts. Static caps and critical rescue keep their original strength.
Forecast-triggered cap restoration is part of `L - S`; a risk flag already
covered by the same static mask has zero incremental contribution. Increasing
the scale does not strengthen that already-covered component.

The scale must be finite and nonnegative. Nondefault scales are rejected for
`none`, `rule_based`, and `predictive_static`, where they would be ignored.
Use `predictive` with scale zero for the shadow-forecast control, and
`predictive_static` for a control that makes no forecast calls. Reports record
the scale and decisions record the actual scaled mask. See
[shield tuning](shield-tuning.md) for a complete validation-only sweep.

At each visited state, diagnostic base, static-only, and full adjusted policy
proposals use the same saved Torch RNG state; comparison does not consume an
extra policy draw.
After exploration the final recommendation can differ from that proposal.
Reports distinguish static and prediction-driven logit changes, proposal changes,
and final recommendation changes after exploration, as well as simulator
acceptance/rejection. Equal initial seeds do not keep complete
trajectories identical once actions differ.

After each successful `Shield.apply`, immutable `shield.last_decision` records
the forecast status/reason, candidate minima/maxima and threshold flags, static
mask, prediction contribution, final mask, and rule reasons. Reset clears it.
The prediction contribution is `final_mask - static_mask`; it can include a
static cap activated by a forecast, so it is not always a simple risk penalty.
An unavailable forecast or a critical-CGM rescue is distinguishable from an
available forecast that did not change the sampled action.

Outputs include `trace.npz`, hash-bound `decisions.json`, `report.json`, numeric `metrics.json`,
`split_check.json`, and `run.json`. Inspect costs, TIR, glucose extremes,
termination/truncation, observed horizon coverage, intervention frequency,
forecast availability/warmup reasons, and latency together. Metrics use post-step
measured CGM. Different action modes, exploration settings, and one-day/seven-day
horizons are distinct protocols. Repeatedly tuning against these same diagnostic
seeds would require a new held-out confirmation set.

## Validation and artifact compatibility

Predictor artifacts must use pre-action inputs, cumulative measured-CGM targets,
training-only normalization, and explicit patient and continuation-policy scope.
Older post-action datasets and predictor checkpoints are not compatible with
this contract; recollect and retrain them. The numbered legacy scripts retain
their original formats.

Software tests cover timing, leakage failures, normalization boundaries, actual
BA-NODE parameter updates, artifact corruption, reset/RNG behavior, and invalid
forecast values. Further evidence must compare held-out per-horizon forecasts
with fixed baselines, test candidate interventions in the simulator, and assess
paired safety/cost/termination outcomes with adequate independent episode and
model seeds. Observational forecast scores alone cannot validate alternate
candidate actions or a hard-filter safety claim.

Use the [paired-response workflow](dynamics.md#collect-and-learn-paired-recommendation-responses)
to test action dependence against the zero-response baseline on both accepted
pure-bolus and pure-meal branches. Report stratum coverage and effect magnitude;
an empty or weak-response stratum is inconclusive. Keep ordinary forecast error
and branch-response error separate. A static rescue that changes a recommendation
while forecasting is unavailable is evidence of the static mechanism only.

Policy-training hyperparameter optimization remains in `glucoalg-tune`. It does
not optimize shield parameters or certify predictor quality. A shield study should fix the artifact and validation protocol first, then
reserve fresh seeds and patients for confirmation. For unseen-patient and
longer-horizon evaluation, select penalties using separate seen-patient episodes
and freeze the selection before inspecting the final test results.
