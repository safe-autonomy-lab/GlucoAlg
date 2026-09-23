"""Held-out causal forecast evaluation against fixed persistence/trend baselines."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from shield.predictor import ForecastRequest, ObservedTransition, PatientIdentity, PointForecast
from .data import claim_directory, load_episodes
from .model import atomic_json, load_predictor, sha256_file


def request_at(episode, t, required, candidates=None):
    """Snapshot only completed transitions; outcome targets never enter this request."""
    if t < required or t > len(episode.recommended_actions):
        raise ValueError("query does not have the required completed history")
    past = tuple(ObservedTransition(
        tuple(map(float, episode.observations[index])),
        tuple(map(int, episode.executed_actions[index])),
        tuple(map(float, episode.observations[index + 1])),
        tuple(map(int, episode.recommended_actions[index])),
        tuple(map(bool, episode.accepted[index])),
    ) for index in range(t - required, t))
    if candidates is None:
        if t >= len(episode.recommended_actions):
            raise ValueError("current recommendation is unavailable")
        candidates = (tuple(map(int, episode.recommended_actions[t])),)
    return ForecastRequest(
        PatientIdentity(episode.metadata["patient_type"], episode.metadata["patient_name"]),
        past, tuple(map(float, episode.observations[t])), tuple(candidates), torch.device("cpu"),
    )


def audit_holdout(artifact, episodes):
    """Refuse reused fit/selection episodes and a different continuation protocol."""
    provenance = artifact["provenance"]
    if not provenance["train_episodes"] or not provenance["validation_episodes"]:
        raise ValueError("artifact requires both fitting and validation episode identities")
    known = provenance["train_episodes"] + provenance["validation_episodes"]
    ids = {item["episode_id"] for item in known}
    seeds = {(item["patient_type"], item["patient_name"], item["seed"]) for item in known}
    hashes = {item["data_sha256"] for item in known}
    branches = fitting_branches(artifact)
    ids.update(item["group_id"] for item in branches)
    seeds.update((item["patient_type"], item["patient_name"], item["seed"]) for item in branches)
    hashes.update(item["prefix_data_sha256"] for item in branches)
    if not known or not ids or not hashes:
        raise ValueError("artifact does not record fit and validation episode identities")
    for episode in episodes:
        meta = episode.metadata
        if (meta["episode_id"] in ids or meta["data_sha256"] in hashes
                or (meta["patient_type"], meta["patient_name"], meta["seed"]) in seeds):
            raise ValueError("held-out episode overlaps fitting or model selection data")
        if meta["split"] != "test":
            raise ValueError("held-out forecast evaluation requires test episodes")
        for key in ("checkpoint", "config"):
            if meta["policy"][key + "_sha256"] != provenance["policy_" + key + "_sha256"]:
                raise ValueError("held-out policy checkpoint/configuration mismatch")
        if meta["continuation_policy"] != artifact["continuation_policy"]:
            raise ValueError("held-out continuation protocol mismatch")
        for key in ("action_mode", "exploration_probability", "exploration_rng"):
            if meta["behavior"][key] != provenance["behavior"][key]:
                raise ValueError("held-out behavior protocol mismatch")
        if meta["runtime"]["glucosim_commit"] != provenance["simulator_commit"]:
            raise ValueError("held-out simulator revision mismatch")
        if meta["runtime"]["glucosim_content_sha256"] != provenance["simulator_content_sha256"]:
            raise ValueError("held-out simulator content mismatch")
    return {"overlap_count": 0, "checked_episodes": len(episodes), "fit_selection_episodes": len(known),
            "fit_selection_branch_groups": len(branches)}


def fitting_branches(artifact):
    """Read optional branch provenance; every anchor binds its entire reset family."""
    provenance = artifact["provenance"]
    records = []
    for key in ("train_branches", "validation_branches"):
        entries = provenance.get(key, [])
        if not isinstance(entries, list):
            raise ValueError("branch fitting provenance must be a list")
        for item in entries:
            if not isinstance(item, dict) or any(not isinstance(item.get(name), str) or not item[name]
                                                 for name in ("group_id", "patient_type", "patient_name", "prefix_data_sha256")):
                raise ValueError("branch fitting provenance requires explicit group/patient/prefix identities")
            PatientIdentity(item["patient_type"], item["patient_name"])
            if type(item.get("seed")) is not int or not 0 <= item["seed"] < 2**32:
                raise ValueError("branch fitting provenance requires an explicit reset seed")
            records.append(item)
    return records


def fixed_baselines(observed_cgm, horizon):
    """Persistence and OLS slope on the same L CGMs, anchored at current CGM."""
    values = np.asarray(observed_cgm, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("baseline history must contain finite observed CGM")
    centered_time = np.arange(len(values), dtype=np.float64) - (len(values) - 1) / 2
    denominator = float(centered_time @ centered_time)
    slope = float(centered_time @ values) / denominator if denominator else 0.0
    return np.full(horizon, values[-1]), values[-1] + slope * np.arange(1, horizon + 1)


def error_metrics(predictions, targets):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.ndim != 2 or predictions.shape != targets.shape or not len(targets):
        raise ValueError("forecast metrics require matching nonempty [queries,horizon] arrays")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("forecast metrics require finite predictions and targets")
    difference = predictions.astype(np.float64) - targets
    return {"queries": len(targets), "mae": np.abs(difference).mean(axis=0).tolist(),
            "rmse": np.sqrt(np.square(difference).mean(axis=0)).tolist()}


def artifact_hashes(directory):
    return {name: sha256_file(Path(directory) / name) for name in ("artifact.json", "artifact.sha256", "weights.pt")}


def validation_source_hash():
    from glucoalg.tuning.plan import _content_hash
    return _content_hash(Path(__file__).resolve().parents[2], ("glucoalg", "shield", "FunctionEncoder"))


def _evaluate_forecasts(artifact_dir, episode_dirs, output):
    before, source_before = artifact_hashes(artifact_dir), validation_source_hash()
    adapter = load_predictor(artifact_dir, device="cpu")
    episodes = load_episodes(episode_dirs, expected_split="test")
    split_check = audit_holdout(adapter.artifact, episodes)
    horizon, length = adapter.config.horizon_steps, adapter.config.history_length
    required = adapter.metadata.history_length
    predictions, targets, persistence, trend, latency = [], [], [], [], []
    episode_indices, anchor_indices, current_cgm, acceptance, rejected, inactive = [], [], [], [], [], []
    rows = []
    for episode_index, episode in enumerate(episodes):
        adapter.reset()
        first = len(predictions)
        for t in range(required, len(episode.observations) - horizon):
            request = request_at(episode, t, required)
            start = time.perf_counter()
            forecast = adapter.forecast(request)
            latency.append(time.perf_counter() - start)
            if not isinstance(forecast, PointForecast):
                raise ValueError("trained adapter unavailable after its declared warm-up")
            values = forecast.glucose_mg_dl.detach().cpu().numpy()
            if values.shape != (1, horizon) or not np.isfinite(values).all():
                raise ValueError("adapter returned an invalid absolute point forecast")
            predictions.append(values[0].copy())
            targets.append(episode.observations[t + 1:t + horizon + 1, 0].copy())
            p, linear = fixed_baselines(episode.observations[t - length + 1:t + 1, 0], horizon)
            persistence.append(p)
            trend.append(linear)
            episode_indices.append(episode_index)
            anchor_indices.append(t)
            current_cgm.append(float(episode.observations[t, 0]))
            acceptance.append(bool(episode.accepted[t].all()))
            active = episode.recommended_actions[t] > 0
            rejected.append(bool(np.any(active & ~episode.accepted[t])))
            inactive.append(not bool(active.any()))
        count = len(predictions) - first
        def episode_errors(values):
            return error_metrics(values[first:], targets[first:]) if count else {
                "queries": 0, "reason": "episode ended before a complete context/query target was available"}
        rows.append({"episode_id": episode.metadata["episode_id"], "patient_type": episode.metadata["patient_type"],
                     "patient_name": episode.metadata["patient_name"], "seed": episode.metadata["seed"],
                     "data_sha256": episode.metadata["data_sha256"], "safety_coverage": episode.metadata["metrics"],
                     "query_start": first, "query_count": count,
                     "model": episode_errors(predictions), "persistence": episode_errors(persistence),
                     "linear_trend": episode_errors(trend)})
    if not predictions:
        raise ValueError("no supplied test episode has an evaluable forecast window")
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    persistence, trend = np.asarray(persistence), np.asarray(trend)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["patient_type"] + "/" + row["patient_name"]].append(index)
    by_patient = {}
    episode_indices = np.asarray(episode_indices)
    for patient, indices in grouped.items():
        mask = np.isin(episode_indices, indices)
        by_patient[patient] = {name: error_metrics(array[mask], targets[mask]) if mask.any() else {"queries": 0}
                               for name, array in (("model", predictions), ("persistence", persistence), ("linear_trend", trend))}
    regions = {}
    current_cgm, acceptance = np.asarray(current_cgm), np.asarray(acceptance)
    rejected, inactive = np.asarray(rejected), np.asarray(inactive)
    for name, mask in (("cgm_below_70", current_cgm < 70),
                       ("cgm_70_to_180", (current_cgm >= 70) & (current_cgm <= 180)),
                       ("cgm_180_to_250", (current_cgm > 180) & (current_cgm <= 250)),
                       ("cgm_above_250", current_cgm > 250),
                       ("both_acceptance_flags_true", acceptance),
                       ("active_recommendation_rejected", rejected),
                       ("all_active_recommendations_accepted", ~rejected & ~inactive),
                       ("no_active_recommendation", inactive)):
        regions[name] = {"queries": int(mask.sum())}
        if mask.any():
            regions[name].update({key: error_metrics(array[mask], targets[mask])
                                  for key, array in (("model", predictions), ("persistence", persistence), ("linear_trend", trend))})
    arrays = output / "forecasts.npz"
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, predictions=predictions, targets=targets, persistence=persistence,
                            linear_trend=trend, episode_index=episode_indices,
                            anchor_index=np.asarray(anchor_indices), current_cgm=current_cgm,
                            both_acceptance_flags_true=acceptance, active_recommendation_rejected=rejected,
                            no_active_recommendation=inactive, latency_seconds=np.asarray(latency))
    summary = {name: error_metrics(array, targets) for name, array in (
        ("model", predictions), ("persistence", persistence), ("linear_trend", trend))}
    if before != artifact_hashes(artifact_dir) or source_before != validation_source_hash():
        raise ValueError("artifact or evaluation source changed during forecast evaluation")
    report = {"schema": "glucoalg.forecast_validation/1", "artifact_dir": str(Path(artifact_dir).resolve()),
              "artifact_sha256": before["artifact.json"], "artifact_hashes": before,
              "evaluation_source_sha256": source_before,
              "weights_sha256": adapter.artifact["weights_sha256"],
              "continuation_policy": adapter.metadata.continuation_policy,
              "forecast_semantics": "observed behavior-policy continuation; point forecasts, not causal bounds",
              "controller_interval_minutes": 5, "horizon_steps": horizon, "history_length": length,
              "required_completed_transitions": required, "episodes": rows, "summary": summary,
              "by_patient": by_patient, "by_region_and_acceptance": regions,
              "latency_seconds": {"p50": float(np.quantile(latency, .5)), "p95": float(np.quantile(latency, .95)),
                                  "p99": float(np.quantile(latency, .99)), "max": max(latency)},
              "forecasts_sha256": sha256_file(arrays),
              "aggregation": "query-weighted descriptive errors; overlapping windows are not independent samples"}
    atomic_json(output / "report.json", report)
    atomic_json(output / "metrics.json", {"queries": len(predictions), "episodes": len(episodes),
                "model_mae": summary["model"]["mae"], "persistence_mae": summary["persistence"]["mae"],
                "linear_trend_mae": summary["linear_trend"]["mae"], "latency_p99_seconds": report["latency_seconds"]["p99"]})
    atomic_json(output / "split_check.json", split_check)
    return report


def evaluate_forecasts(artifact_dir, episode_dirs, output_dir):
    output = claim_directory(output_dir)
    atomic_json(output / "run.json", {"status": "started"})
    try:
        report = _evaluate_forecasts(artifact_dir, episode_dirs, output)
        atomic_json(output / "run.json", {"status": "complete", "report_sha256": sha256_file(output / "report.json")})
        return report
    except BaseException as exc:
        atomic_json(output / "run.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, help="explicit test episode directories")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = evaluate_forecasts(args.artifact, args.episodes, args.output_dir)
    print(json.dumps({"episodes": len(report["episodes"]), "model_mae": report["summary"]["model"]["mae"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
