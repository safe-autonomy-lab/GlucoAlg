"""Finite synthetic traces test selection rules; these are not simulator results."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glucoalg.dynamics import tuning as t


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def study(tmp_path, monkeypatch):
    sources = {"repo": {"head": "commit", "content_sha256": "a" * 64},
               "simulator": {"head": "sim", "content_sha256": "b" * 64}}
    monkeypatch.setattr(t, "probe_sources", lambda *_: deepcopy(sources))
    checkpoint, config, protocol = (tmp_path / name for name in ("checkpoint.pt", "config.json", "protocol.md"))
    checkpoint.write_bytes(b"fixture policy")
    config.write_text("{}")
    protocol.write_text("fixture protocol before data")
    sim = tmp_path / "simulator"
    sim.mkdir()
    models = []
    for seed in (3101, 3102, 3103):
        directory = tmp_path / str(seed)
        directory.mkdir()
        (directory / "weights.pt").write_bytes(str(seed).encode())
        artifact = dict(schema="glucoalg.ba_node_predictor", schema_version=1,
                        weights_sha256=t.sha256_file(directory / "weights.pt"), training={"seed": seed},
                        continuation_policy="fixed stochastic policy with epsilon0.1",
                        patient_scope={"supported_patients": [dict(diabetes_type="t1d", patient_name=f"adolescent#{index:03}") for index in (1, 2)]},
                        provenance=dict(policy_checkpoint_sha256=t.sha256_file(checkpoint),
                                        policy_config_sha256=t.sha256_file(config), simulator_commit="sim",
                                        simulator_content_sha256="b" * 64,
                                        behavior=dict(action_mode="stochastic", exploration_probability=.1),
                                        train_episodes=[dict(patient_type="t1d", patient_name="adolescent#001", seed=700)],
                                        validation_episodes=[dict(patient_type="t1d", patient_name="adolescent#002", seed=800)]))
        write(directory / "artifact.json", artifact)
        write(directory / "artifact.sha256", {"sha256": t.sha256_file(directory / "artifact.json")})
        models.append(dict(seed=seed, artifact_dir=str(directory)))
    cases = [dict(id=f"p{patient}-s{seed}", patient_type="t1d", patient_name=f"adolescent#{patient:03}",
                  env_seed=seed, action_seed=seed, exploration_seed=seed + 10000)
             for patient in (1, 2) for seed in (5100, 5101)]
    spec = dict(schema_version=1, study_name="fixture", checkpoint=str(checkpoint), config=str(config),
                simulator_root=str(sim), models=models, zero_model_seed=3101, cases=cases,
                action_modes=["stochastic", "deterministic"], scales=[0, .1, .3, 1, 3],
                horizon_days=7, exploration_probability=0, protocol_files=[str(protocol)])
    return spec, sources, tmp_path


def plan(study, *, compact=False):
    spec, sources, tmp_path = study
    spec = deepcopy(spec)
    if compact:
        spec.update(cases=spec["cases"][:1], action_modes=["stochastic"], scales=[0, 1], horizon_days=1)
    return t.create_plan(spec, tmp_path / "study")


def rows_for(plan, *, zero=1000, gains=None, n=2016):
    gains = gains or {}
    rows = []
    for task in plan["tasks"]:
        h = round(task["config"]["horizon_days"] * 288)
        length = min(n, h)
        count = zero + gains.get(task["scale"], 0)
        count = min(count, length)
        rows.append(dict(task_id=task["id"], case_id=task["case_id"], model_seed=task["model_seed"],
                         scale=task["scale"], action_mode=task["action_mode"],
                         patient_type=task["config"]["patient_type"], patient_name=task["config"]["patient_name"],
                         length=length, horizon_steps=h, terminated=length < h, early_termination=length < h,
                         severe_hypo_samples=0, below_70_samples=0, above_250_fraction=0.,
                         observed_tir=count / length, requested_time_in_range=count / h, cost=100., reward=-100.))
    return rows


def result_fixture(plan, task, directory, *, cgm=None):
    directory.mkdir()
    cgm = np.asarray(cgm if cgm is not None else [120.] * 288, dtype=np.float32)
    n, h = len(cgm), round(task["config"]["horizon_days"] * 288)
    arrays = dict(observations=np.zeros((n + 1, 14), dtype=np.float32),
                  recommended_actions=np.zeros((n, 2), dtype=np.int64), executed_actions=np.zeros((n, 2), dtype=np.int64),
                  accepted=np.zeros((n, 2), dtype=bool), rewards=np.ones(n), costs=np.zeros(n),
                  terminated=np.zeros(n, dtype=bool), truncated=np.zeros(n, dtype=bool),
                  shield_latency_seconds=np.ones(n) * .001)
    arrays["observations"][:, 0] = np.r_[120., cgm]
    arrays["terminated"][-1] = n < h
    arrays["truncated"][-1] = n == h
    for name in ("base_proposals", "static_proposals", "adjusted_proposals", "base_actions", "static_actions"):
        arrays[name] = np.zeros((n, 2), dtype=np.int64)
    flags = ("static_logit_changed", "prediction_logit_changed", "static_proposal_changed", "prediction_proposal_changed",
             "static_recommendation_changed", "prediction_recommendation_changed", "proposal_changed", "recommendation_changed",
             "logit_changed", "explored", "forecast_available")
    for name in flags:
        arrays[name] = np.zeros(n, dtype=bool)
    decisions = {"schema": "glucoalg.shield_decisions/1", "steps": [dict(step_index=i,
                  current_cgm=float(arrays["observations"][i, 0]), use_forecast=True, static_mask=[0.] * 10,
                  prediction_mask=[0.] * 10, final_mask=[0.] * 10, static_changed=False,
                  prediction_changed=False, final_changed=False) for i in range(n)]}
    metrics = dict(length=n, horizon_steps=h, coverage_fraction=n / h, terminated=n < h, truncated=n == h,
                   early_termination=n < h, severe_hypo_samples=int(np.sum(cgm < 54)), hyper_250_samples=int(np.sum(cgm > 250)),
                   reward=float(n), cost=0., termination_cause=2 if n < h else 0,
                   time_in_range_pct=100 * float(np.mean((cgm >= 70) & (cgm <= 180))),
                   time_below_range_pct=100 * float(np.mean(cgm < 70)), time_above_range_pct=100 * float(np.mean(cgm > 180)))
    for name in flags:
        metric = {"logit_changed": "logit_intervention", "explored": "exploration", "forecast_available": "forecast_available"}.get(name, name) + "_steps"
        metrics[metric] = 0
    model = plan["inputs"]["models"][str(task["model_seed"]) ]
    report = {key: value for key, value in task["config"].items() if key not in ("artifact_dir", "checkpoint", "config", "simulator_root")}
    report.update(schema="glucoalg.predictive_rollout/1", status="diagnostic", model_seed=task["model_seed"],
                  artifact_sha256=model["hashes"]["artifact.json"], artifact_hashes=model["hashes"],
                  continuation_policy=model["continuation_policy"], behavior_matches_collection=False,
                  shield_settings={"config": dict(t.DEFAULT_CONFIG, forecast_penalty_scale=task["scale"]), "params": t.DEFAULT_PARAMS},
                  sources={"before": plan["sources"], "after": plan["sources"]}, metrics=metrics)
    def publish():
        np.savez_compressed(directory / "trace.npz", **arrays)
        write(directory / "decisions.json", decisions)
        report.update(trace_sha256=t.sha256_file(directory / "trace.npz"), decisions_sha256=t.sha256_file(directory / "decisions.json"))
        write(directory / "report.json", report)
        write(directory / "metrics.json", {key: value for key, value in metrics.items() if type(value) in (int, float, bool)})
        write(directory / "split_check.json", {"overlap_count": 0})
        write(directory / "run.json", {"status": "complete", "report_sha256": t.sha256_file(directory / "report.json")})
    publish()
    return arrays, decisions, report, publish


def test_plan_has_104_tasks_deduplicates_only_zero_and_exposes_portable_argv(study):
    sealed = plan(study)
    assert len(sealed["tasks"]) == 104
    zero = [task for task in sealed["tasks"] if task["scale"] == 0]
    assert len(zero) == 8 and {task["model_seed"] for task in zero} == {3101}
    task = zero[0]
    argv = t.task_argv(sealed, task, "{outdir}/artifact", python="/pinned/python")
    assert argv[:3] == ["/pinned/python", "-m", "glucoalg.dynamics.rollout"]
    assert argv[-2:] == ["--output-dir", "{outdir}/artifact"]
    assert argv[argv.index("--forecast-penalty-scale") + 1] == "0.0"
    assert t.load_plan(study[2] / "study") == sealed
    with pytest.raises(FileExistsError):
        t.create_plan(study[0], study[2] / "study")


@pytest.mark.parametrize("key,value", [("scales", [0, True]), ("scales", [0, float("nan")]),
    ("scales", [0, 1, 1]), ("scales", [1]), ("horizon_days", .9), ("horizon_days", 10**400),
    ("exploration_probability", True), ("exploration_probability", 1.1), ("action_modes", ["stochastic", "stochastic"]),
    ("zero_model_seed", True), ("schema_version", True)])
def test_rejects_ambiguous_or_nonfinite_spec(study, key, value):
    spec = deepcopy(study[0])
    spec[key] = value
    with pytest.raises(ValueError):
        t.normalize_spec(spec)


def test_duplicate_and_excluded_patient_reset_families_fail(study):
    spec = deepcopy(study[0])
    spec["cases"][1] = dict(spec["cases"][0], id="another")
    with pytest.raises(ValueError, match="reset family"):
        t.normalize_spec(spec)
    spec = deepcopy(study[0])
    spec["excluded_families"] = [{key: spec["cases"][0][key] for key in ("patient_type", "patient_name", "env_seed")}]
    with pytest.raises(ValueError, match="excluded"):
        t.normalize_spec(spec)


def test_fit_overlap_and_policy_binding_fail_before_plan_publication(study):
    spec = deepcopy(study[0])
    spec["cases"][0]["env_seed"] = 700
    with pytest.raises(ValueError, match="overlaps"):
        t.create_plan(spec, study[2] / "bad")
    assert not (study[2] / "bad").exists()
    Path(spec["checkpoint"]).write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="policy hashes"):
        t.create_plan(study[0], study[2] / "bad")


@pytest.mark.parametrize("target", ["protocol", "weights", "source", "plan"])
def test_sealed_plan_refuses_drift(study, target):
    sealed = plan(study)
    if target == "protocol":
        Path(sealed["spec"]["protocol_files"][0]).write_text("changed")
    elif target == "weights":
        (Path(sealed["spec"]["models"][0]["artifact_dir"]) / "weights.pt").write_bytes(b"changed")
    elif target == "source":
        study[1]["repo"]["content_sha256"] = "c" * 64
    else:
        (study[2] / "study" / "plan.json").write_text("{}")
    with pytest.raises(ValueError):
        t.load_plan(study[2] / "study")


def test_tie_band_chooses_smallest_and_never_uses_cost_or_reward(study):
    sealed = plan(study)
    rows = rows_for(sealed, gains={.1: 20, .3: 22, 1.: 21, 3.: 22})
    # Difference 2/2016 < .001, so .1 wins despite .3's larger score.
    selected = t.select_scales(sealed, rows)
    assert all(item["selected_scale"] == .1 for item in selected.values())
    for row in rows:
        row.update(cost=1e100 if row["scale"] == .1 else 0, reward=-1e100)
    assert t.select_scales(sealed, rows) == selected


def test_zero_wins_small_gain_and_no_eligible_fallback_is_not_success(study):
    sealed = plan(study)
    selected = t.select_scales(sealed, rows_for(sealed, gains={.1: 2}))
    assert all(item["selected_scale"] == 0 and item["status"] == "no_eligible_improvement" for item in selected.values())
    failed = t.select_scales(sealed, rows_for(sealed, n=1500))
    assert all(not item["zero_eligible"] and item["maximum_eligible_score"] is None and item["selected_scale"] == 0 for item in failed.values())


def test_one_model_seed_veto_and_final_step_termination(study):
    sealed = plan(study)
    rows = rows_for(sealed, gains={.1: 50})
    row = next(row for row in rows if row["scale"] == .1 and row["model_seed"] == 3103 and row["action_mode"] == "stochastic")
    row["terminated"] = True  # full horizon, still a new physiological termination
    selection = t.select_scales(sealed, rows)
    assert selection["stochastic"]["selected_scale"] == 0
    assert selection["deterministic"]["selected_scale"] == .1
    candidate = next(item for item in selection["stochastic"]["candidates"] if item["scale"] == .1)
    assert any("physiological" in reason for reason in candidate["by_model_seed"]["3103"]["reasons"])


@pytest.mark.parametrize("change,reason", [({"length": 2015}, "shorter"),
    ({"severe_hypo_samples": 1}, "severe_hypo"), ({"below_70_samples": 1}, "below_70"),
    ({"above_250_fraction": .01}, "above_250"), ({"observed_tir": 0}, "observed_tir")])
def test_each_registered_gate_is_effective(study, change, reason):
    sealed = plan(study)
    baseline = [row for row in rows_for(sealed) if row["scale"] == 0 and row["action_mode"] == "stochastic"]
    candidate = deepcopy(baseline)
    candidate[0].update(change)
    checked = t.eligibility(candidate, baseline)
    assert not checked["eligible"] and any(reason in item for item in checked["reasons"])


def test_patient_high_fraction_cannot_cancel_between_patients(study):
    sealed = plan(study)
    baseline = [row for row in rows_for(sealed) if row["scale"] == 0 and row["action_mode"] == "stochastic"]
    for row in baseline:
        row["above_250_fraction"] = .2
    candidate = deepcopy(baseline)
    for row in candidate:
        row["above_250_fraction"] = .3 if row["patient_name"].endswith("001") else 0
    assert not t.eligibility(candidate, baseline)["eligible"]


def test_missing_duplicate_or_mislabeled_rows_block_selection(study):
    sealed = plan(study)
    rows = rows_for(sealed)
    with pytest.raises(ValueError, match="every planned task"):
        t.select_scales(sealed, rows[:-1])
    with pytest.raises(ValueError, match="every planned task"):
        t.select_scales(sealed, rows[:-1] + [rows[0]])
    rows[0]["model_seed"] = 3103
    with pytest.raises(ValueError, match="identity"):
        t.select_scales(sealed, rows)


def test_raw_trace_summary_and_exclusive_selection_publication(study):
    sealed = plan(study, compact=True)
    locations = {}
    for task in sealed["tasks"]:
        directory = study[2] / task["id"]
        result_fixture(sealed, task, directory, cgm=[120.] * 144 + [300.] * 144)
        locations[task["id"]] = str(directory)
    result = t.summarize(study[2] / "study", locations)
    assert result["planned_tasks"] == 4
    assert result["rows"][0]["requested_time_in_range"] == .5
    assert result["rows"][0]["hyper_250_samples"] == 144
    assert result["selection"]["stochastic"]["selected_scale"] == 0
    with pytest.raises(FileExistsError):
        t.summarize(study[2] / "study", locations)


@pytest.mark.parametrize("mutation", ["nan", "early_row", "wrong_shape", "acceptance", "wrong_model", "wrong_source",
    "wrong_scale", "wrong_static", "bool_setting", "metric", "zero_proposal", "none_decision", "zero_mask", "cause"])
def test_corrupt_or_mismatched_result_blocks_selector(study, mutation):
    sealed = plan(study, compact=True)
    task, directory = sealed["tasks"][0], study[2] / "result"
    arrays, decisions, report, publish = result_fixture(sealed, task, directory)
    if mutation == "nan":
        arrays["observations"][1, 0] = np.nan
    elif mutation == "early_row":
        arrays["terminated"][0] = True
    elif mutation == "wrong_shape":
        arrays["rewards"] = arrays["rewards"][:, None]
    elif mutation == "acceptance":
        arrays["executed_actions"][0, 0] = 1
    elif mutation == "wrong_model":
        report["model_seed"] = 3102
    elif mutation == "wrong_source":
        report["sources"] = deepcopy(report["sources"])
        report["sources"]["after"]["repo"]["content_sha256"] = "other"
    elif mutation == "wrong_scale":
        report["forecast_penalty_scale"] = .1
    elif mutation == "wrong_static":
        report["shield_settings"]["config"]["logit_penalty"] = 5.
    elif mutation == "bool_setting":
        report["shield_settings"]["config"]["use_forecast"] = 1
    elif mutation == "metric":
        report["metrics"]["time_in_range_pct"] = 99.
    elif mutation == "zero_proposal":
        arrays["adjusted_proposals"][0, 0] = 1
    elif mutation == "none_decision":
        decisions["steps"][0] = None
    elif mutation == "zero_mask":
        decisions["steps"][0]["prediction_mask"][0] = -10.
        decisions["steps"][0]["final_mask"][0] = -10.
    else:
        report["metrics"]["termination_cause"] = 99
    publish()
    with pytest.raises(ValueError):
        t.verify_result(sealed, task, directory)


def test_failed_task_and_hash_changed_trace_do_not_pass(study):
    sealed = plan(study, compact=True)
    task, directory = sealed["tasks"][0], study[2] / "result"
    _, _, _, publish = result_fixture(sealed, task, directory)
    write(directory / "run.json", {"status": "failed"})
    with pytest.raises(ValueError, match="incomplete"):
        t.verify_result(sealed, task, directory)
    publish()
    with (directory / "trace.npz").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="hash mismatch"):
        t.verify_result(sealed, task, directory)


def test_cli_help_and_import_do_not_load_heavy_runtime():
    code = "import sys; import glucoalg.dynamics.tuning; assert not any(m in sys.modules for m in ('torch','jax','glucosim')); print('lightweight')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "glucoalg.dynamics.tuning", "plan", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--output" in result.stdout


@pytest.mark.parametrize("scale", [0., .1, .3, 1., 3.])
@pytest.mark.parametrize("mode", ["stochastic", "deterministic"])
@pytest.mark.parametrize("epsilon", [0., 1.])
def test_actual_rollout_decision_output_satisfies_verifier(scale, mode, epsilon):
    """Real shield/rollout plumbing with a fake predictor/environment, no simulation."""
    from types import SimpleNamespace
    import torch
    from glucoalg.dynamics.rollout import rollout_episode
    from shield.predictive_shield import PredictiveShieldConfig, Shield
    from shield.predictor import PatientIdentity, PointForecast, PredictorMetadata
    from test_dynamics_collect import FakeEnv
    from test_dynamics_rollout import Actor
    predictor = SimpleNamespace(metadata=PredictorMetadata(1, 2, "fixture"), reset=lambda: None,
                                forecast=lambda request: PointForecast(torch.full((len(request.candidate_actions), 2), 65.)))
    shield = Shield(predictor=predictor, patient=PatientIdentity("t1d", "adolescent#001"),
                    config=PredictiveShieldConfig(forecast_penalty_scale=scale))
    records = []
    arrays, metrics, _ = rollout_episode(FakeEnv(), Actor(), None, shield=shield, seed=1, action_seed=2,
                                        exploration_seed=3, horizon_steps=288, action_mode=mode,
                                        exploration_probability=epsilon, decision_records=records)
    t._verify_decisions(arrays, {"steps": records}, metrics, scale, len(arrays["rewards"]))
