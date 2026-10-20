# Reading the result files

The CSVs preserve the frozen export's original values. Empty fields represent
missing/not-applicable values; `no_eligible_penalty` is a scientific selection
outcome and must not be replaced with zero.

## Summary tables

`algorithm`, `patient_type`, `cohort`, `action_mode`, `source`, `cell` and `arm`
identify a comparison. `cell` is `seen1`, `unseen1`, `seen7` or `unseen7`.
`fixed10` applies absolute penalty 10; `selected` uses seen-one-day validation.
`selected_aliases_fixed10` means both labels refer to the same physical runs.

The six-algorithm CSVs encode `metrics`, `none_metrics`, `static_metrics`,
`delta_vs_none`, `delta_vs_static`, `policy_means`, `batch_keys`, `counts` and
`issues` as JSON where present. Use `json.loads` to decode these columns.
Metric objects contain `mean`, `policy_sd`, `policy_n`, `policy_min` and
`policy_max`. Historical CPO/PPO-Lag summaries use flat metric columns.

| Metric | Meaning / unit |
| --- | --- |
| `requested_tir_pct` | CGM 70–180 mg/dL samples divided by the requested horizon, % |
| `observed_tir_pct` | CGM 70–180 mg/dL samples divided by recorded samples, % |
| `coverage_pct` | Recorded versus requested horizon, % |
| `early_end_pct` | Episodes ending before the requested horizon, % |
| `physiological_end_pct` | Episodes with physiological termination, % |
| `below54_*`, `below70_*`, `above250_*` | Glucose exposure at the named mg/dL threshold; `observed` and `requested` specify denominator |
| `delta_vs_none`, `delta_vs_static` | Differences of the stated metric; percentage-valued metric differences are percentage points |
| `native_cost`, `native_reward`, `risk_index` | Recorded definitions from the corresponding runtime; do not assume cross-runtime cost alignment |

`policy_sd` is the sample SD of three equally weighted policy means. Episode
counts are support counts, not the number of independent training seeds.
The diagnostic `pooled_observed_tir_pct_diagnostic` in the six-algorithm
`all-panels.csv` and `pooled_observed_tir_pct` in the historical table use a
different weighting and are not the primary requested-horizon result.

## Episode files on Hugging Face

Each `episodes/<algorithm>.csv.gz` contains one row per retained physical
episode and is readable with `pandas.read_csv`. `crpo.csv.gz` contains the
implementation label `OnCRPO`. Shared columns include:

| Columns | Meaning |
| --- | --- |
| `patient_name`, `patient_type`, `cohort` | Virtual patient and diabetes configuration |
| `policy_seed`, `model_seed`, `env_seed`, `action_seed` | Distinct policy, predictor, reset and action seeds; model seed is blank for controls |
| `condition`, `absolute_penalty`, `action_mode` | Shield condition, actual penalty, and action-selection mode |
| `horizon_days`, `horizon_steps`, `length` | Requested duration and recorded controller-step count |
| `in_range_requested_fraction`, `requested_time_in_range` | Requested-horizon TIR as a fraction, not a percentage |
| `in_range_fraction`, `observed_tir` | Observed-horizon TIR as a fraction |
| `coverage_fraction`, `early_termination`, `terminated`, `truncated`, `termination_cause` | Coverage and termination bookkeeping |
| `*_samples`, `*_steps` | Recorded integer counts |
| `*_fraction` | Recorded fractions; multiply by 100 only when reporting percentages |
| `glucose_min`, `glucose_mean`, `glucose_max` | Recorded CGM summary, mg/dL |
| `cost`, `reward`, `risk_index` | Existing scalar values; not freshly recomputed by this export |
| `shield_latency_*_seconds` | Runtime latency in seconds, dependent on the original hardware |
| `policy_checkpoint_sha256`, `policy_config_sha256`, `model_artifact_sha256`, `trace_sha256` | Artifact identities; trace bytes are not in this download |
| `source_runtime`, `source_task`, `source_wave`, `case_id` | Provenance and physical row identity |
| `source_proof_sha256`, `source_task_sha256` | Accepted six-algorithm proof/task hashes; historical rows use their separately pinned source CSV instead |

The frozen simulator's `termination_cause` values are `0` (no termination),
`1` (interval-average modeled blood glucose below 10 mg/dL), `2` (above
600 mg/dL), and `3` (simulator time limit). A physiological cause takes
precedence over the time limit. These are simulator definitions, not clinical
threshold recommendations.

Use `(source_runtime, source_task, case_id)` as the physical row key. Preserve
predictor identities, match controls within the same policy/patient/reset, and
never count a reused control or selected-10 alias twice. For the main
generalization comparison, filter seven days and patients #002–#010, while
keeping deterministic and stochastic modes separate.

The episode files contain inherited native metrics. Their inclusion does not
claim a new archive-wide native-metric recomputation. Read the release README
and source summary before changing the aggregation or comparing code versions.
