# Patients #009 and #010 trajectory fields

For each patient, three conditions: `none`, `static`, `predictive`. Each has 2,016 recorded
five-minute transitions and 2,017 observations including reset. The seven-day
horizon is 10,080 minutes. Every original NPZ field is preserved.

| NPZ array | Shape | Meaning |
|---|---|---|
| `observations` | (2017,14) | Raw simulator observations; reset at row 0. CGM is column 0 in mg/dL. |
| `recommended_actions` | (2016,2) | Requested bolus and meal levels, in that order, each 0–4. |
| `executed_actions` | (2016,2) | Simulator-accepted recommendation levels, not physical doses. |
| `accepted` | (2016,2) | Acceptance flags for the two requested dimensions. |
| `rewards`, `costs` | (2016,) | Original native transition outcomes. |
| `terminated`, `truncated` | (2016,) | Native termination/timeout flags. All six episodes reach the horizon. |
| `base_proposals`, `static_proposals`, `adjusted_proposals` | (2016,2) | Base, static and final action proposals before exploration override. |
| `base_actions`, `static_actions` | (2016,2) | Coupled base/static recommendations after exploration override. |
| `static_logit_changed`, `prediction_logit_changed` | (2016,) | Which component changed logits; this alone does not establish an action change. |
| `static_proposal_changed`, `prediction_proposal_changed` | (2016,) | Stage-specific changes to proposed recommendations. |
| `static_recommendation_changed`, `prediction_recommendation_changed` | (2016,) | Stage-specific changes to final recommendations. |
| `proposal_changed`, `recommendation_changed`, `logit_changed` | (2016,) | Overall base-to-final changes. |
| `explored` | (2016,) | Exploration override flag; exploration probability is zero here. |
| `forecast_available` | (2016,) | Whether an actual forecast was available on that step. |
| `shield_latency_seconds` | (2016,) | Host execution time spent applying the shield; not simulated time. |

The 14 observation features, in column order, are `cgm`, `iob`, `cob`,
`cgm_trend`, `time_sin`, `time_cos`, `time_since_meal`, `time_since_bolus`,
`planned_meal_left`, `meal_count_norm`, `bolus_count_norm`,
`time_until_meal_norm`, `next_meal_size_norm`, `is_pre_bolus_window`.
CGM is mg/dL, bolus IOB is U and COB is g. CGM trend is the raw change between
successive five-minute average CGM values, not a per-minute rate. Other features
are normalized times/counts/meal sizes or indicators; original feature names
and values are available in metadata and the observation CSV.

`output/patient009.observations.csv` contains all 14 features for each arm,
with `observation_minute` and `is_reset`. Its 6,051 rows include resets. Patient #010 has the same files prefixed `patient010`.
`output/patient009.actions.csv` contains 6,048 transitions, with decision and
post-step observation timestamps, requested/executed levels, acceptance,
rewards/costs, terminal flags and forecast availability. Additional diagnostic
arrays remain in the original NPZ files. `output/patient009.metrics.csv` has one
row per arm. CSV is a convenience format; NPZ retains canonical dtypes/bytes.

`rollout.json` contains original per-step pre/post observations and action
records. `source/decisions.json` contains candidate action levels, forecast
minimum/maximum CGM, masks, threshold flags and reasons where retained. Forecast
extrema are model predictions, not measured glucose. Full forecast-horizon
vectors were not saved and are not reconstructed.

No scheduled meal or delivered dose history is inferred from accepted requests.
Autonomous meals, basal insulin, acceptance rules and absorption delays can
also affect glucose. This selected illustration is separate from aggregate
cohort evidence.
