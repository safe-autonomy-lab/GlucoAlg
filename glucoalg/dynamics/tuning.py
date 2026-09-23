"""Portable, sealed validation sweep for incremental forecast penalties.

This is a descriptive selector, not a clinical safety test. Confirmation uses
separate reset families and is deliberately outside this validation command.
Planning and summarizing do not import Torch, JAX or the simulator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

from glucoalg.tuning.plan import probe_sources
from glucoalg.tuning.spec import load_json_nodupes

SCHEMA = "glucoalg.shield_tuning/1"
TOLERANCE = 1e-12
TIE_BAND = .001
MIN_COVERAGE = .95
ARTIFACT_FILES = ("artifact.json", "artifact.sha256", "weights.pt")
DEFAULT_CONFIG = dict(hypo_check_threshold=80., hyper_check_threshold=250.,
                      top_k_bolus_levels=2, logit_penalty=10., use_meal_hyper_check=False,
                      use_forecast=True, forecast_penalty_scale=1.)
DEFAULT_PARAMS = dict(BG_CRITICAL_HYPO=60., MIN_INTERVENTION_BG_LOW=90.,
                      MIN_INTERVENTION_BG_HIGH=160., RESCUE_MEAL_LEVEL=1,
                      RESCUE_COOLDOWN_MIN=60.)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    result = load_json_nodupes(Path(path))
    if not _finite_tree(result):
        raise ValueError(f"nonfinite JSON: {path}")
    return result


def _write_json(path, value):
    blob = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(blob)


def _number(value, name, *, minimum=0):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _seed(value, name):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an integer in [0,2**32)")
    return value


def _identity(case):
    kind, name = case["patient_type"], case["patient_name"]
    if kind not in ("t1d", "t2d", "t2d_no_pump") or not isinstance(name, str) or not re.fullmatch(
            r"(?:adolescent|adult|child)#00[1-9]|(?:adolescent|adult|child)#010", name):
        raise ValueError("invalid simulator patient identity")
    return kind, name, _seed(case["env_seed"], "env_seed")


def _path(value, *, directory=False):
    if not isinstance(value, str) or not value:
        raise ValueError("input paths must be nonempty strings")
    result = Path(value).expanduser().resolve()
    if not (result.is_dir() if directory else result.is_file()):
        raise ValueError(f"input {'directory' if directory else 'file'} missing: {result}")
    return str(result)


def normalize_spec(spec):
    """Validate an explicit validation grid; no hidden patient/seed defaults."""
    spec = deepcopy(spec)
    required = {"schema_version", "study_name", "checkpoint", "config", "simulator_root",
                "models", "zero_model_seed", "cases", "action_modes", "scales",
                "horizon_days", "exploration_probability", "protocol_files"}
    if not isinstance(spec, dict) or set(spec) - required - {"excluded_families"} or required - set(spec):
        raise ValueError("study spec has missing or unknown keys")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("unsupported study schema_version")
    if not isinstance(spec["study_name"], str) or not spec["study_name"].strip():
        raise ValueError("study_name must be nonempty")
    for key in ("checkpoint", "config", "simulator_root"):
        spec[key] = _path(spec[key], directory=key == "simulator_root")
    horizon = _number(spec["horizon_days"], "horizon_days", minimum=1)
    if not math.isfinite(horizon * 288) or not (horizon * 288).is_integer():
        raise ValueError("horizon_days must represent complete five-minute steps")
    spec["horizon_days"] = horizon
    epsilon = _number(spec["exploration_probability"], "exploration_probability")
    if epsilon > 1:
        raise ValueError("exploration_probability must be <=1")
    spec["exploration_probability"] = epsilon
    modes = spec["action_modes"]
    if not isinstance(modes, list) or not modes or any(mode not in ("stochastic", "deterministic") for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("action_modes must be unique stochastic/deterministic values")
    scales = spec["scales"]
    if not isinstance(scales, list) or not scales:
        raise ValueError("scales must be a nonempty list")
    scales = [_number(value, "scale") for value in scales]
    if 0 not in scales or not any(value > 0 for value in scales) or len(set(scales)) != len(scales):
        raise ValueError("scales require unique zero and positive values")
    spec["scales"] = sorted(scales)
    models = spec["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be nonempty")
    for model in models:
        if not isinstance(model, dict) or set(model) != {"seed", "artifact_dir"}:
            raise ValueError("model entries require seed and artifact_dir")
        _seed(model["seed"], "model seed")
        model["artifact_dir"] = _path(model["artifact_dir"], directory=True)
    if len({model["seed"] for model in models}) != len(models):
        raise ValueError("model seeds must be unique")
    if _seed(spec["zero_model_seed"], "zero_model_seed") not in {model["seed"] for model in models}:
        raise ValueError("zero_model_seed must identify a planned model")
    spec["models"] = sorted(models, key=lambda item: item["seed"])
    cases = spec["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    seen_ids, seen_families = set(), set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "patient_type", "patient_name", "env_seed", "action_seed", "exploration_seed"}:
            raise ValueError("case requires explicit id, patient and all three seeds")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
            raise ValueError("case id must contain letters, digits, underscores or hyphens")
        family = _identity(case)
        for key in ("action_seed", "exploration_seed"):
            _seed(case[key], key)
        if case["id"] in seen_ids or family in seen_families:
            raise ValueError("duplicate case identity or reset family")
        seen_ids.add(case["id"])
        seen_families.add(family)
    excluded = spec.setdefault("excluded_families", [])
    if not isinstance(excluded, list):
        raise ValueError("excluded_families must be a list")
    for family in excluded:
        if not isinstance(family, dict) or set(family) != {"patient_type", "patient_name", "env_seed"}:
            raise ValueError("excluded family requires patient_type/name and env_seed")
        if _identity(family) in seen_families:
            raise ValueError("planned case overlaps excluded reset family")
    if not isinstance(spec["protocol_files"], list) or not spec["protocol_files"]:
        raise ValueError("protocol_files must bind at least one frozen file")
    spec["protocol_files"] = [_path(path) for path in spec["protocol_files"]]
    if len(set(spec["protocol_files"])) != len(spec["protocol_files"]):
        raise ValueError("protocol_files must be unique")
    return spec


def _input_bindings(spec, sources):
    bindings = {key: {"path": spec[key], "sha256": sha256_file(spec[key])} for key in ("checkpoint", "config")}
    bindings["protocol_files"] = [{"path": path, "sha256": sha256_file(path)} for path in spec["protocol_files"]]
    models = {}
    for model in spec["models"]:
        directory = Path(model["artifact_dir"])
        hashes = {name: sha256_file(directory / name) for name in ARTIFACT_FILES}
        artifact = _json(directory / "artifact.json")
        if _json(directory / "artifact.sha256") != {"sha256": hashes["artifact.json"]}:
            raise ValueError("artifact seal mismatch")
        if (artifact.get("schema") != "glucoalg.ba_node_predictor" or artifact.get("schema_version") != 1
                or type(artifact["training"]["seed"]) is not int
                or artifact["weights_sha256"] != hashes["weights.pt"] or artifact["training"]["seed"] != model["seed"]):
            raise ValueError("model weight/seed mismatch")
        provenance = artifact["provenance"]
        for key in ("checkpoint", "config"):
            if provenance["policy_" + key + "_sha256"] != bindings[key]["sha256"]:
                raise ValueError("predictor and policy hashes disagree")
        if (provenance["simulator_commit"] != sources["simulator"]["head"] or
                provenance["simulator_content_sha256"] != sources["simulator"]["content_sha256"]):
            raise ValueError("predictor and simulator provenance disagree")
        known = []
        for key in ("train_episodes", "validation_episodes", "train_branches", "validation_branches"):
            entries = provenance.get(key, [])
            if not isinstance(entries, list) or (key.endswith("episodes") and not entries):
                raise ValueError("missing fitting/validation episode provenance")
            known.extend(entries)
        families = {(item["patient_type"], item["patient_name"], item["seed"]) for item in known}
        scope = {(item["diabetes_type"], item["patient_name"]) for item in artifact["patient_scope"]["supported_patients"]}
        for case in spec["cases"]:
            if _identity(case) in families:
                raise ValueError("validation case overlaps predictor fitting/model selection")
            if _identity(case)[:2] not in scope:
                raise ValueError("validation patient outside artifact transfer scope")
        behavior = provenance["behavior"]
        _number(behavior["exploration_probability"], "model exploration probability")
        if behavior["action_mode"] not in ("stochastic", "deterministic"):
            raise ValueError("invalid model behavior provenance")
        models[str(model["seed"])] = {"artifact_dir": str(directory), "hashes": hashes,
                                      "behavior": behavior, "continuation_policy": artifact["continuation_policy"]}
    bindings["models"] = models
    return bindings


def _tasks(spec, output):
    tasks = []
    for mode in spec["action_modes"]:
        for case in spec["cases"]:
            for scale in spec["scales"]:
                for model in spec["models"]:
                    if scale == 0 and model["seed"] != spec["zero_model_seed"]:
                        continue
                    index = len(tasks)
                    config = {key: spec[key] for key in ("checkpoint", "config", "simulator_root", "horizon_days", "exploration_probability")}
                    config.update({key: value for key, value in case.items() if key != "id"})
                    config.update(artifact_dir=model["artifact_dir"], action_mode=mode,
                                  condition="predictive", forecast_penalty_scale=scale)
                    tasks.append({"id": f"task-{index:05d}", "index": index, "model_seed": model["seed"],
                                  "scale": scale, "case_id": case["id"], "action_mode": mode, "config": config,
                                  "output_dir": str(Path(output) / "runs" / f"task-{index:05d}")})
    return tasks


def create_plan(spec, output_dir):
    """Seal the explicit validation grid and all scientific inputs; never overwrite."""
    spec = normalize_spec(_json(spec) if isinstance(spec, (str, Path)) else spec)
    sources = probe_sources(spec["simulator_root"])
    inputs = _input_bindings(spec, sources)
    output = Path(output_dir).resolve()
    plan = {"schema": SCHEMA, "phase": "validation", "spec": spec, "sources": sources,
            "inputs": inputs, "output_dir": str(output), "tasks": _tasks(spec, output),
            "selection_rule": {"tolerance": TOLERANCE, "tie_band_fraction": TIE_BAND,
                               "minimum_coverage": MIN_COVERAGE,
                               "score": "equal-model/equal-case in-range samples/requested steps",
                               "tie": "lowest scale within tie band of maximum eligible score"}}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256_file(output / "plan.json") + "\n")
    return plan


def load_plan(path, *, verify_inputs=True):
    path = Path(path)
    if path.is_dir():
        path = path / "plan.json"
    if path.with_suffix(".sha256").read_text().strip() != sha256_file(path):
        raise ValueError("plan seal mismatch")
    plan = _json(path)
    if plan["schema"] != SCHEMA or plan["phase"] != "validation":
        raise ValueError("unsupported plan schema/phase")
    if plan["tasks"] != _tasks(plan["spec"], plan["output_dir"]):
        raise ValueError("plan task grid mismatch")
    if verify_inputs:
        current = probe_sources(plan["spec"]["simulator_root"])
        for kind in ("repo", "simulator"):
            if current[kind]["content_sha256"] != plan["sources"][kind]["content_sha256"]:
                raise ValueError(f"{kind} source content drift")
        if _input_bindings(plan["spec"], current) != plan["inputs"]:
            raise ValueError("plan input/protocol/artifact drift")
    return plan


def task_argv(plan, task, output_dir=None, python=None):
    """Explicit argv for local or external runners; no shell or scheduler syntax."""
    if task not in plan["tasks"]:
        raise ValueError("task is not in the sealed plan")
    argv = [python or sys.executable, "-m", "glucoalg.dynamics.rollout"]
    for key, value in task["config"].items():
        argv.extend(["--" + ("artifact" if key == "artifact_dir" else key.replace("_", "-")), str(value)])
    return argv + ["--output-dir", str(output_dir or task["output_dir"])]


def run_task(plan_path, task_id, output_dir=None):
    plan = load_plan(plan_path)
    task = next((task for task in plan["tasks"] if task["id"] == task_id or task["index"] == task_id), None)
    if task is None:
        raise ValueError("unknown task id/index")
    from .rollout import run_rollout
    report = run_rollout(**task["config"], output_dir=output_dir or task["output_dir"])
    load_plan(plan_path)  # A successful return also requires unchanged inputs/source.
    return report


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if type(value) in (int, float):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return True


def _same_number(actual, expected, name, *, tolerance=1e-9):
    if type(expected) is bool:
        valid = type(actual) is bool and actual == expected
    else:
        valid = type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
    if not valid:
        raise ValueError(f"reported {name} disagrees with trace")


def _same_config(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(_same_config(actual[key], item) for key, item in expected.items())
    if type(expected) is bool:
        return type(actual) is bool and actual == expected
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return actual == expected


def _verify_decisions(arrays, decisions, metrics, scale, n):
    """Check recorded attribution, including exact static-effect zero control."""
    flags = ("static_logit_changed", "prediction_logit_changed", "static_proposal_changed",
             "prediction_proposal_changed", "static_recommendation_changed", "prediction_recommendation_changed",
             "proposal_changed", "recommendation_changed", "logit_changed", "explored", "forecast_available")
    for name in flags:
        if name not in arrays or arrays[name].shape != (n,) or arrays[name].dtype != np.bool_:
            raise ValueError("invalid attribution flag array: " + name)
    for name in ("base_proposals", "static_proposals", "adjusted_proposals", "base_actions", "static_actions"):
        values = arrays.get(name)
        if values is None or values.shape != (n, 2) or values.dtype.kind not in "iu" or (values < 0).any() or (values > 4).any():
            raise ValueError("invalid paired action array: " + name)
    for flag, left, right in (("static_proposal_changed", "base_proposals", "static_proposals"),
                              ("prediction_proposal_changed", "static_proposals", "adjusted_proposals"),
                              ("proposal_changed", "base_proposals", "adjusted_proposals"),
                              ("static_recommendation_changed", "base_actions", "static_actions"),
                              ("prediction_recommendation_changed", "static_actions", "recommended_actions"),
                              ("recommendation_changed", "base_actions", "recommended_actions")):
        if not np.array_equal(arrays[flag], np.any(arrays[left] != arrays[right], axis=1)):
            raise ValueError("attribution flag disagrees with paired actions: " + flag)
    explored = arrays["explored"]
    for proposal, action in (("base_proposals", "base_actions"), ("static_proposals", "static_actions"), ("adjusted_proposals", "recommended_actions")):
        if not np.array_equal(arrays[proposal][~explored], arrays[action][~explored]):
            raise ValueError("nonexploratory proposal differs from recommendation")
    if not np.array_equal(arrays["base_actions"][explored], arrays["static_actions"][explored]) or not np.array_equal(arrays["base_actions"][explored], arrays["recommended_actions"][explored]):
        raise ValueError("exploration did not override all paired actions equally")
    for index, decision in enumerate(decisions["steps"]):
        if (not isinstance(decision, dict) or type(decision.get("step_index")) is not int
                or decision["step_index"] != index or decision.get("use_forecast") is not True
                or decision.get("current_cgm") != float(arrays["observations"][index, 0])):
            raise ValueError("decision step/patient observation/forecast mismatch")
        masks = [np.asarray(decision.get(key), dtype=np.float64) for key in ("static_mask", "prediction_mask", "final_mask")]
        if any(mask.shape != (10,) or not np.isfinite(mask).all() for mask in masks):
            raise ValueError("invalid decision masks")
        static, prediction, final = masks
        if not np.allclose(static + prediction, final, rtol=1e-6, atol=1e-6):
            raise ValueError("decision masks do not sum")
        if not np.isin(static, (-10., 0., 10.)).all() or not np.all(np.isclose(prediction, 0, rtol=0, atol=1e-6) | np.isclose(prediction, -10 * scale, rtol=1e-6, atol=1e-6)):
            raise ValueError("decision mask violates the fixed penalty operator")
        for key, array_name in (("static_changed", "static_logit_changed"), ("prediction_changed", "prediction_logit_changed"), ("final_changed", "logit_changed")):
            if type(decision.get(key)) is not bool or decision[key] != bool(arrays[array_name][index]):
                raise ValueError("decision/logit attribution mismatch")
        if scale == 0 and (np.any(prediction) or not np.array_equal(static, final)):
            raise ValueError("zero penalty differs from the static mask")
    if scale == 0 and any(arrays[name].any() for name in ("prediction_logit_changed", "prediction_proposal_changed", "prediction_recommendation_changed")):
        raise ValueError("zero penalty changed static proposals/recommendations")
    metric_names = {name: name + "_steps" for name in flags if name not in ("logit_changed", "explored", "forecast_available")}
    metric_names.update(logit_changed="logit_intervention_steps", explored="exploration_steps", forecast_available="forecast_available_steps")
    for name, metric in metric_names.items():
        _same_number(metrics.get(metric), int(arrays[name].sum()), metric)
    latency = arrays.get("shield_latency_seconds")
    if latency is None or latency.shape != (n,) or (latency < 0).any():
        raise ValueError("invalid shield latency vector")


def verify_result(plan, task, directory):
    """Bind a complete artifact to its planned identity and recompute selector data."""
    directory = Path(directory)
    run, report = _json(directory / "run.json"), _json(directory / "report.json")
    if run.get("status") != "complete" or run.get("report_sha256") != sha256_file(directory / "report.json"):
        raise ValueError("incomplete or unsealed rollout")
    if report.get("schema") != "glucoalg.predictive_rollout/1" or report.get("status") != "diagnostic" or not _finite_tree(report):
        raise ValueError("invalid/nonfinite rollout report")
    config = task["config"]
    for key in ("patient_type", "patient_name", "env_seed", "action_seed", "exploration_seed", "horizon_days",
                "action_mode", "exploration_probability", "condition", "forecast_penalty_scale"):
        if not _same_config(report.get(key), config[key]) or (type(config[key]) is int and type(report.get(key)) is not int):
            raise ValueError(f"rollout protocol mismatch: {key}")
    expected_settings = {"config": dict(DEFAULT_CONFIG, forecast_penalty_scale=task["scale"]), "params": DEFAULT_PARAMS}
    if not _same_config(report.get("shield_settings"), expected_settings):
        raise ValueError("shield settings differ from the fixed static/rescue protocol")
    model = plan["inputs"]["models"][str(task["model_seed"])]
    if (report.get("model_seed") != task["model_seed"] or report.get("artifact_hashes") != model["hashes"]
            or report.get("artifact_sha256") != model["hashes"]["artifact.json"]
            or report.get("continuation_policy") != model["continuation_policy"]):
        raise ValueError("rollout model identity/hash mismatch")
    behavior_match = (config["action_mode"] == model["behavior"]["action_mode"] and
                      config["exploration_probability"] == model["behavior"]["exploration_probability"])
    if type(report.get("behavior_matches_collection")) is not bool or report["behavior_matches_collection"] != behavior_match:
        raise ValueError("behavior match label disagrees with the artifact")
    for phase in ("before", "after"):
        for kind in ("repo", "simulator"):
            if report["sources"][phase][kind]["content_sha256"] != plan["sources"][kind]["content_sha256"]:
                raise ValueError("rollout source content mismatch")
    overlap = _json(directory / "split_check.json").get("overlap_count")
    if type(overlap) is not int or overlap != 0:
        raise ValueError("rollout split check failed")
    for name, key in (("trace.npz", "trace_sha256"), ("decisions.json", "decisions_sha256")):
        if sha256_file(directory / name) != report.get(key):
            raise ValueError("rollout " + name + " hash mismatch")
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name, values in arrays.items():
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("nonfinite/non-numeric trace array: " + name)
    n = len(arrays["rewards"])
    horizon = round(config["horizon_days"] * 288)
    if not 0 < n <= horizon or arrays["observations"].shape != (n + 1, 14):
        raise ValueError("invalid trace observation/episode shape")
    for name in ("rewards", "costs", "terminated", "truncated"):
        if arrays[name].shape != (n,):
            raise ValueError("invalid trace vector shape: " + name)
    for name in ("terminated", "truncated", "accepted"):
        if arrays[name].dtype != np.bool_:
            raise ValueError("trace flags must be Boolean")
    done = arrays["terminated"] | arrays["truncated"]
    if done[:-1].any() or not done[-1] or (arrays["costs"] < 0).any():
        raise ValueError("invalid terminal ordering or negative cost")
    for name in ("recommended_actions", "executed_actions"):
        values = arrays[name]
        if values.shape != (n, 2) or values.dtype.kind not in "iu" or (values < 0).any() or (values > 4).any():
            raise ValueError("invalid categorical action array")
    if arrays["accepted"].shape != (n, 2) or not np.array_equal(arrays["executed_actions"], np.where(arrays["accepted"], arrays["recommended_actions"], 0)):
        raise ValueError("executed actions disagree with simulator acceptance flags")
    decisions = _json(directory / "decisions.json")
    if decisions.get("schema") != "glucoalg.shield_decisions/1" or len(decisions.get("steps", [])) != n or not _finite_tree(decisions):
        raise ValueError("invalid/nonfinite decision log")
    _verify_decisions(arrays, decisions, report["metrics"], task["scale"], n)
    cgm = arrays["observations"][1:, 0].astype(np.float64)
    low54, low70, high250 = (int(np.count_nonzero(cgm < 54)), int(np.count_nonzero(cgm < 70)), int(np.count_nonzero(cgm > 250)))
    in_range = int(np.count_nonzero((cgm >= 70) & (cgm <= 180)))
    cost, reward = math.fsum(map(float, arrays["costs"])), math.fsum(map(float, arrays["rewards"]))
    if not math.isfinite(cost) or not math.isfinite(reward):
        raise ValueError("nonfinite total reward/cost")
    recomputed = dict(length=n, horizon_steps=horizon, coverage_fraction=n / horizon,
                      terminated=bool(arrays["terminated"][-1]), truncated=bool(arrays["truncated"][-1]),
                      early_termination=n < horizon, severe_hypo_samples=low54, hyper_250_samples=high250,
                      reward=reward, cost=cost)
    for name, expected in recomputed.items():
        _same_number(report["metrics"].get(name), expected, name)
    # Glucose summaries also pass through the legacy float32/JAX implementation.
    for name, expected in (("time_in_range_pct", 100 * in_range / n),
                           ("time_below_range_pct", 100 * low70 / n),
                           ("time_above_range_pct", 100 * int(np.count_nonzero(cgm > 180)) / n)):
        _same_number(report["metrics"].get(name), expected, name, tolerance=1e-5)
    cause = report["metrics"].get("termination_cause")
    if type(cause) is not int or cause not in (0, 1, 2, 3):
        raise ValueError("invalid termination cause metadata")
    projection = {key: value for key, value in report["metrics"].items() if type(value) in (int, float, bool)}
    if _json(directory / "metrics.json") != projection:
        raise ValueError("metrics.json differs from the sealed report")
    return dict(task_id=task["id"], case_id=task["case_id"], model_seed=task["model_seed"],
                scale=task["scale"], action_mode=task["action_mode"], patient_type=config["patient_type"],
                patient_name=config["patient_name"], env_seed=config["env_seed"], **recomputed,
                below_70_samples=low70, in_range_samples=in_range, observed_tir=in_range / n,
                requested_time_in_range=in_range / horizon, above_250_fraction=high250 / n,
                termination_cause=cause, behavior_matches_collection=behavior_match,
                artifact_path=str(directory.resolve()), report_sha256=sha256_file(directory / "report.json"),
                trace_sha256=report["trace_sha256"], decisions_sha256=report["decisions_sha256"])


def eligibility(rows, baseline):
    """Registered per-model safety/coverage gates, including final-step termination."""
    reasons = []
    base = {row["case_id"]: row for row in baseline}
    if len(base) != len(baseline) or {row["case_id"] for row in rows} != set(base) or len(rows) != len(base):
        raise ValueError("eligibility requires exact paired case coverage")
    for row in rows:
        control = base[row["case_id"]]
        if row["terminated"] and not control["terminated"]:
            reasons.append(row["case_id"] + ":new_physiological_termination")
        if row["early_termination"] and not control["early_termination"]:
            reasons.append(row["case_id"] + ":new_early_termination")
        if row["length"] < control["length"]:
            reasons.append(row["case_id"] + ":shorter_coverage")
        for key in ("severe_hypo_samples", "below_70_samples"):
            if row[key] > control[key]:
                reasons.append(row["case_id"] + ":increased_" + key)
    for patient in {(row["patient_type"], row["patient_name"]) for row in rows}:
        chosen = [row for row in rows if (row["patient_type"], row["patient_name"]) == patient]
        controls = [base[row["case_id"]] for row in chosen]
        if statistics.mean(row["above_250_fraction"] for row in chosen) > statistics.mean(row["above_250_fraction"] for row in controls) + TOLERANCE:
            reasons.append("/".join(patient) + ":increased_above_250_fraction")
    if statistics.mean(row["observed_tir"] for row in rows) + TOLERANCE < statistics.mean(row["observed_tir"] for row in baseline):
        reasons.append("decreased_observed_tir")
    coverage = sum(row["length"] for row in rows) / sum(row["horizon_steps"] for row in rows)
    if coverage + TOLERANCE < MIN_COVERAGE:
        reasons.append("coverage_below_95_percent")
    return {"eligible": not reasons, "reasons": reasons, "coverage_fraction": coverage,
            "score": statistics.mean(row["requested_time_in_range"] for row in rows),
            "observed_tir": statistics.mean(row["observed_tir"] for row in rows)}


def select_scales(plan, rows):
    """Select only after the complete physical task grid has verified results."""
    if len(rows) != len(plan["tasks"]) or {row["task_id"] for row in rows} != {task["id"] for task in plan["tasks"]}:
        raise ValueError("selection requires every planned task exactly once")
    indexed = {row["task_id"]: row for row in rows}
    for task in plan["tasks"]:
        for key in ("case_id", "scale", "model_seed", "action_mode"):
            if indexed[task["id"]][key] != task[key]:
                raise ValueError("selection row identity mismatch")
    selection = {}
    for mode in plan["spec"]["action_modes"]:
        baseline = [row for row in rows if row["action_mode"] == mode and row["scale"] == 0]
        candidates = []
        for scale in plan["spec"]["scales"]:
            by_seed = {}
            for model in plan["spec"]["models"]:
                chosen = baseline if scale == 0 else [row for row in rows if row["action_mode"] == mode and row["scale"] == scale and row["model_seed"] == model["seed"]]
                by_seed[str(model["seed"])] = eligibility(chosen, baseline)
            candidates.append({"scale": scale, "eligible": all(item["eligible"] for item in by_seed.values()),
                               "score": statistics.mean(item["score"] for item in by_seed.values()), "by_model_seed": by_seed})
        eligible = [item for item in candidates if item["eligible"]]
        zero = next(item for item in candidates if item["scale"] == 0)
        best_score = max((item["score"] for item in eligible), default=None)
        chosen = min((item["scale"] for item in eligible if item["score"] + TIE_BAND + TOLERANCE >= best_score), default=0.)
        # No positive promotion without an actual improvement over the shared zero.
        chosen_result = next(item for item in candidates if item["scale"] == chosen)
        if chosen == 0 or chosen_result["score"] <= zero["score"] + TOLERANCE:
            chosen, status = 0., "no_eligible_improvement"
        else:
            status = "selected_positive_scale"
        selection[mode] = {"selected_scale": chosen, "status": status, "maximum_eligible_score": best_score,
                           "zero_score": zero["score"], "zero_eligible": zero["eligible"], "candidates": candidates}
    return selection


def summarize(plan_path, results_manifest=None, output_path=None):
    """Verify every artifact, then exclusively publish the first complete selection.

    results_manifest is a JSON object mapping task IDs to reconstructed artifact
    directory paths. Verification of an external capture/signature is the
    caller's responsibility; this command verifies content and scientific identity.
    """
    plan = load_plan(plan_path)
    locations = _json(results_manifest) if isinstance(results_manifest, (str, Path)) else results_manifest
    if locations is None:
        locations = {task["id"]: task["output_dir"] for task in plan["tasks"]}
    if not isinstance(locations, dict) or set(locations) != {task["id"] for task in plan["tasks"]}:
        raise ValueError("results manifest must map every planned task exactly once")
    if any(not isinstance(value, str) or not value for value in locations.values()) or len({str(Path(value).resolve()) for value in locations.values()}) != len(locations):
        raise ValueError("result directories must be distinct explicit paths")
    rows = [verify_result(plan, task, locations[task["id"]]) for task in plan["tasks"]]
    result = {"schema": SCHEMA, "status": "complete", "phase": "validation", "planned_tasks": len(plan["tasks"]),
              "plan_sha256": sha256_file(Path(plan_path) / "plan.json" if Path(plan_path).is_dir() else plan_path),
              "inputs": plan["inputs"], "sources": plan["sources"], "selection_rule": plan["selection_rule"],
              "rows": rows, "selection": select_scales(plan, rows),
              "scope": "Descriptive validation selector; no held-out confirmation or safety guarantee."}
    load_plan(plan_path)
    destination = Path(output_path) if output_path else Path(plan["output_dir"]) / "selection.json"
    _write_json(destination, result)
    with destination.with_suffix(".sha256").open("x") as stream:
        stream.write(sha256_file(destination) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("plan", help="seal an explicit validation study")
    build.add_argument("--spec", required=True)
    build.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    run = commands.add_parser("run", help="run one local planned rollout")
    run.add_argument("--plan", required=True)
    choice = run.add_mutually_exclusive_group(required=True)
    choice.add_argument("--task-id")
    choice.add_argument("--index", type=int)
    run.add_argument("--output-dir")
    summary = commands.add_parser("summarize", help="verify all outputs and seal selection")
    summary.add_argument("--plan", required=True)
    summary.add_argument("--results-manifest")
    summary.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = create_plan(args.spec, args.output_dir)
            print(json.dumps({"tasks": len(result["tasks"]), "plan": str(Path(args.output_dir) / "plan.json")}))
        elif args.command == "run":
            run_task(args.plan, args.task_id if args.task_id is not None else args.index, args.output_dir)
        else:
            result = summarize(args.plan, args.results_manifest, args.output)
            print(json.dumps(result["selection"], allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        parser.exit(1, f"shield tuning failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
