"""Experimental, paired simulator diagnostics for a supplied point predictor.

This command records intervention behavior; it does not promote the predictor
to the supported evaluation path or establish a safety guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from numbers import Real
from pathlib import Path
import time

import numpy as np
import torch

from glucoalg.evaluation import MultiCategoricalDistribution, set_seed
from glucoalg.runtime import configure_runtime, initialize_simulator
from shield.predictive_shield import PredictiveShieldConfig, Shield
from shield.predictor import ForecastUnavailable, PatientIdentity, PointForecast
from shield.rule_based_shield import RuleBasedShield
from .collect import _flag, _observation, _scalar, accepted_execution
from .data import Episode, claim_directory, episode_metrics, load_episodes
from .model import atomic_json, load_predictor, sha256_file
from .validate import artifact_hashes, fitting_branches


def paired_proposals(base_logits, adjusted_logits, mode):
    """Couple diagnostic proposals using the same Torch RNG state.

    Only one proposal's random draws advance the stream. A second proposal is
    evaluated at that saved state, so counting action changes does not itself
    alter subsequent policy randomness.
    """
    return coupled_proposals(mode, base_logits, adjusted_logits)


def coupled_proposals(mode, *logit_sets):
    """Compare base/static/full proposals while advancing only the base RNG draw."""
    if mode not in ("stochastic", "deterministic"):
        raise ValueError("action_mode must be stochastic or deterministic")
    if not logit_sets:
        raise ValueError("at least one logit set is required")
    for logits in logit_sets:
        if logits.shape != (1, 10) or logits.device.type != "cpu" or not torch.isfinite(logits).all():
            raise ValueError("paired proposals require finite CPU logits of shape (1,10)")

    def draw(logits):
        distribution = MultiCategoricalDistribution([
            torch.distributions.Categorical(logits=part) for part in logits.split((5, 5), dim=-1)])
        action = distribution.sample() if mode == "stochastic" else distribution.mode
        return action.detach().numpy().reshape(2).copy()

    before = torch.random.get_rng_state()
    proposals = [draw(logit_sets[0])]
    after = torch.random.get_rng_state()
    try:
        for logits in logit_sets[1:]:
            torch.random.set_rng_state(before)
            proposals.append(draw(logits))
    finally:
        torch.random.set_rng_state(after)
    return tuple(proposals)


def rollout_episode(env, actor, normalizer, *, shield, seed, action_seed, exploration_seed,
                    horizon_steps, action_mode, exploration_probability, decision_records=None):
    """Run one episode; expose the arrays needed to recompute all diagnostics."""
    if getattr(actor, "shield", None) is not None:
        raise ValueError("diagnostics require an unshielded base actor")
    if tuple(np.asarray(env.action_space.nvec)) != (5, 5):
        raise ValueError("diagnostics require two five-level categorical actions")
    if _scalar(env.sample_time, "interval") != 5 or _scalar(env.simulation_minutes, "horizon") != horizon_steps * 5:
        raise ValueError("simulator interval/horizon mismatch")
    if action_mode not in ("stochastic", "deterministic") or not 0 <= exploration_probability <= 1:
        raise ValueError("invalid action mode or exploration probability")
    set_seed(action_seed)
    exploration = np.random.Generator(np.random.PCG64(exploration_seed))
    if shield is not None:
        shield.reset()
    observations, recommendations, executed, acceptances, rewards, costs, terminals, timeouts = ([] for _ in range(8))
    base_actions, proposal_changes, recommendation_changes, logit_changes, explored, elapsed = ([] for _ in range(6))
    attribution = {name: [] for name in (
        "base_proposals", "static_proposals", "adjusted_proposals", "static_actions",
        "static_logit_changed", "prediction_logit_changed", "static_proposal_changed",
        "prediction_proposal_changed", "static_recommendation_changed", "prediction_recommendation_changed",
    )}
    reasons, available = Counter(), []
    observation, _ = env.reset(seed=seed)
    observations.append(_observation(observation))
    for _ in range(horizon_steps):
        raw = observations[-1]
        normalized = normalizer.normalize(raw.copy()) if normalizer is not None else raw
        if not np.isfinite(normalized).all():
            raise ValueError("non-finite normalized policy observation")
        with torch.inference_mode():
            logits = actor.logits_net(torch.as_tensor(normalized, dtype=torch.float32)[None])
            start = time.perf_counter()
            adjusted = logits if shield is None else shield.apply(
                torch.as_tensor(raw, dtype=torch.float32)[None], logits, [5, 5])
            elapsed.append(time.perf_counter() - start)
            decision = shield.last_decision if isinstance(shield, Shield) else None
            static_logits = adjusted if decision is None else logits + logits.new_tensor(decision.static_mask)
            base, static, action = coupled_proposals(action_mode, logits, static_logits, adjusted)
        if decision_records is not None:
            decision_records.append(None if decision is None else asdict(decision))
        attribution["base_proposals"].append(base.copy())
        attribution["static_proposals"].append(static.copy())
        attribution["adjusted_proposals"].append(action.copy())
        attribution["static_logit_changed"].append(not torch.equal(logits, static_logits))
        attribution["prediction_logit_changed"].append(not torch.equal(static_logits, adjusted))
        attribution["static_proposal_changed"].append(not np.array_equal(base, static))
        attribution["prediction_proposal_changed"].append(not np.array_equal(static, action))
        logit_changes.append(not torch.equal(logits, adjusted))
        proposal_changes.append(not np.array_equal(base, action))
        exploring = exploration.random() < exploration_probability
        if exploring:
            # Deliberately the same continuation mixture used for collection.
            # Exploration occurs after the soft intervention and can bypass it.
            base = exploration.integers(0, 5, size=2, dtype=np.int64)
            static = base.copy()
            action = base.copy()
        explored.append(exploring)
        base_actions.append(base.copy())
        attribution["static_actions"].append(static.copy())
        attribution["static_recommendation_changed"].append(not np.array_equal(base, static))
        attribution["prediction_recommendation_changed"].append(not np.array_equal(static, action))
        recommendation_changes.append(not np.array_equal(base, action))
        prediction = getattr(shield, "last_forecast", None)
        available.append(isinstance(prediction, PointForecast))
        if isinstance(prediction, ForecastUnavailable):
            reason = "warmup" if prediction.reason.startswith("warmup:") else prediction.reason
            reasons[reason] += 1
        recommendation = action.copy()
        observation, reward, cost, terminal, timeout, info = env.step(action[None].copy())
        outcome, flags = accepted_execution(recommendation, info)
        next_observation = _observation(observation)
        if isinstance(shield, Shield):
            shield.record_action(outcome, next_observation=torch.as_tensor(next_observation),
                                 recommended_action=recommendation, accepted=tuple(map(bool, flags)))
        observations.append(next_observation)
        recommendations.append(recommendation)
        executed.append(outcome)
        acceptances.append(flags)
        rewards.append(_scalar(reward, "reward"))
        costs.append(_scalar(cost, "cost"))
        if costs[-1] < 0:
            raise ValueError("simulator cost must be nonnegative")
        terminals.append(_flag(terminal, "terminated"))
        timeouts.append(_flag(timeout, "truncated"))
        if terminals[-1] or timeouts[-1]:
            break
    else:
        raise ValueError("simulator did not end at the requested horizon")
    cause = _scalar(info["termination_cause"], "termination_cause")
    if cause < 0 or cause != int(cause):
        raise ValueError("invalid termination cause")
    arrays = dict(observations=np.asarray(observations), recommended_actions=np.asarray(recommendations),
                  executed_actions=np.asarray(executed), accepted=np.asarray(acceptances), rewards=np.asarray(rewards),
                  costs=np.asarray(costs), terminated=np.asarray(terminals), truncated=np.asarray(timeouts))
    episode = Episode(**arrays, metadata={"horizon_steps": horizon_steps})
    metrics = episode_metrics(episode, termination_cause=int(cause))
    cgm = arrays["observations"][1:, 0]
    metrics.update(glucose_min=float(cgm.min()), glucose_max=float(cgm.max()), glucose_mean=float(cgm.mean()),
                   logit_intervention_steps=sum(logit_changes), proposal_changed_steps=sum(proposal_changes),
                   recommendation_changed_steps=sum(recommendation_changes), exploration_steps=sum(explored),
                   forecast_available_steps=sum(available),
                   active_recommendation_rejected_steps=int(np.any(
                       (arrays["recommended_actions"] > 0) & ~arrays["accepted"], axis=1).sum()))
    for name in ("logit_intervention", "proposal_changed", "recommendation_changed"):
        metrics[name + "_fraction"] = metrics[name + "_steps"] / len(rewards)
    for name, values in attribution.items():
        if name.endswith("_changed"):
            metrics[name + "_steps"] = sum(values)
            metrics[name + "_fraction"] = sum(values) / len(rewards)
    arrays.update({name: np.asarray(values) for name, values in attribution.items()})
    arrays.update(base_actions=np.asarray(base_actions), proposal_changed=np.asarray(proposal_changes),
                  recommendation_changed=np.asarray(recommendation_changes), logit_changed=np.asarray(logit_changes),
                  explored=np.asarray(explored), forecast_available=np.asarray(available),
                  shield_latency_seconds=np.asarray(elapsed))
    return arrays, metrics, dict(reasons)


def audit_rollout(artifact, patient, seed, checkpoint, config, runtime, excluded):
    provenance = artifact["provenance"]
    if vars(patient) not in artifact["patient_scope"]["supported_patients"]:
        raise ValueError("patient is outside artifact transfer scope")
    for key, path in (("checkpoint", checkpoint), ("config", config)):
        if sha256_file(path) != provenance["policy_" + key + "_sha256"]:
            raise ValueError("rollout policy " + key + " mismatch")
    if (runtime["glucosim_commit"] != provenance["simulator_commit"]
            or runtime["glucosim_content_sha256"] != provenance["simulator_content_sha256"]):
        raise ValueError("rollout simulator revision/content mismatch")
    known = provenance["train_episodes"] + provenance["validation_episodes"]
    known += [episode.metadata for episode in excluded]
    branches = fitting_branches(artifact)
    for item in known + branches:
        if (item["patient_type"], item["patient_name"], item["seed"]) == (patient.diabetes_type, patient.patient_name, seed):
            raise ValueError("rollout seed overlaps fitting, selection or excluded episodes")
    return {"overlap_count": 0, "fit_selection_episodes_checked": len(known) - len(excluded),
            "additional_episodes_checked": len(excluded), "fit_selection_branch_groups_checked": len(branches)}


def run_rollout(*, artifact_dir, checkpoint, config, simulator_root, output_dir, patient_type,
                patient_name, condition, env_seed, action_seed, exploration_seed, horizon_days=1,
                action_mode="stochastic", exploration_probability=None, exclude_episodes=(),
                forecast_penalty_scale=1.0):
    from glucoalg.evaluation import _glucose_metrics, create_diabetes_env, load_model
    from glucoalg.eval_grid import validate_protocol
    from glucoalg.tuning.plan import _content_hash, probe_sources

    validate_protocol(patient_type=patient_type, patients=[patient_name], episodes=1,
                      horizon_days=horizon_days, eval_seed_base=env_seed, action_mode=action_mode)
    patient = PatientIdentity(patient_type, patient_name)
    if condition not in ("none", "rule_based", "predictive", "predictive_static"):
        raise ValueError("unknown intervention condition")
    if (isinstance(forecast_penalty_scale, bool) or not isinstance(forecast_penalty_scale, Real)
            or not math.isfinite(forecast_penalty_scale) or forecast_penalty_scale < 0):
        raise ValueError("forecast_penalty_scale must be finite and nonnegative")
    if condition != "predictive" and forecast_penalty_scale != 1:
        raise ValueError("nondefault forecast_penalty_scale requires condition='predictive'; "
                         "use predictive with scale zero for the shadow control")
    forecast_penalty_scale = float(forecast_penalty_scale)
    for value in (action_seed, exploration_seed):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("RNG seeds must be integers in [0,2**32)")
    before = artifact_hashes(artifact_dir)
    adapter = load_predictor(artifact_dir)
    behavior = adapter.artifact["provenance"]["behavior"]
    epsilon = behavior["exploration_probability"] if exploration_probability is None else exploration_probability
    if type(epsilon) not in (int, float) or not 0 <= epsilon <= 1:
        raise ValueError("exploration probability must lie in [0,1]")
    output = claim_directory(output_dir)
    atomic_json(output / "run.json", {"status": "started"})
    env = None
    try:
        configure_runtime()
        runtime = initialize_simulator(simulator_root)
        sim_path = Path(runtime["glucosim_file"]).parent.parent
        runtime["glucosim_content_sha256"] = _content_hash(sim_path, ("glucosim",))
        sources_before = probe_sources(simulator_root)
        excluded = load_episodes(exclude_episodes, expected_split="test") if exclude_episodes else []
        split = audit_rollout(adapter.artifact, patient, env_seed, checkpoint, config, runtime, excluded)
        env = create_diabetes_env(patient_type, patient_name, seed=env_seed, horizon_days=horizon_days)
        actor, _, normalizer = load_model(checkpoint, json.loads(Path(config).read_text()), env)
        shield = Shield(predictor=adapter, patient=patient,
                        config=PredictiveShieldConfig(use_forecast=condition == "predictive",
                                                      forecast_penalty_scale=forecast_penalty_scale)) if condition in (
                            "predictive", "predictive_static") else (
            RuleBasedShield() if condition == "rule_based" else None)
        decisions = []
        arrays, metrics, reasons = rollout_episode(
            env, actor, normalizer, shield=shield, seed=env_seed, action_seed=action_seed,
            exploration_seed=exploration_seed, horizon_steps=round(horizon_days * 288),
            action_mode=action_mode, exploration_probability=epsilon, decision_records=decisions)
        metrics.update(_glucose_metrics(arrays["observations"][1:, 0], 5))
        sources_after = probe_sources(simulator_root)
        if any(sources_before[name]["content_sha256"] != sources_after[name]["content_sha256"] for name in ("repo", "simulator")):
            raise ValueError("runtime source content changed during rollout")
        if before != artifact_hashes(artifact_dir):
            raise ValueError("predictor artifact changed during rollout")
        if _content_hash(sim_path, ("glucosim",)) != runtime["glucosim_content_sha256"]:
            raise ValueError("simulator content changed during rollout")
        for key, path in (("checkpoint", checkpoint), ("config", config)):
            if sha256_file(path) != adapter.artifact["provenance"]["policy_" + key + "_sha256"]:
                raise ValueError("policy files changed during rollout")
        with (output / "trace.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        atomic_json(output / "decisions.json", {"schema": "glucoalg.shield_decisions/1", "steps": decisions})
        report = {"schema": "glucoalg.predictive_rollout/1", "status": "diagnostic",
                  "patient_type": patient_type, "patient_name": patient_name, "condition": condition,
                  "env_seed": env_seed, "action_seed": action_seed, "exploration_seed": exploration_seed,
                  "horizon_days": horizon_days, "action_mode": action_mode, "exploration_probability": epsilon,
                  "forecast_penalty_scale": forecast_penalty_scale,
                  "artifact_sha256": before["artifact.json"], "artifact_hashes": before,
                  "sources": {"before": sources_before, "after": sources_after},
                  "shield_settings": {"config": asdict(shield.pred_cfg), "params": asdict(shield.params)}
                  if isinstance(shield, Shield) else ({"config": asdict(shield.config)} if shield is not None else {}),
                  "model_seed": adapter.artifact["training"]["seed"],
                  "continuation_policy": adapter.metadata.continuation_policy,
                  "behavior_matches_collection": action_mode == behavior["action_mode"] and epsilon == behavior["exploration_probability"],
                  "intervention_changes_continuation": condition != "none",
                  "sampling": "Torch categorical with paired RNG state; independent PCG64 exploration overrides afterward",
                  "attribution": "base/static/full proposals share RNG; prediction contribution compares static versus full at the same visited state, before and after matched exploration",
                  "scope": "mechanism diagnostic; factual forecast accuracy does not validate counterfactual actions or safety",
                  "runtime": runtime, "metrics": metrics, "forecast_unavailable_reasons": reasons,
                  "shield_latency_seconds": {key: float(np.quantile(arrays["shield_latency_seconds"], quantile))
                                              for key, quantile in (("p50", .5), ("p95", .95), ("p99", .99))},
                  "trace_sha256": sha256_file(output / "trace.npz"),
                  "decisions_sha256": sha256_file(output / "decisions.json")}
        atomic_json(output / "report.json", report)
        atomic_json(output / "metrics.json", {key: value for key, value in metrics.items()
                                              if type(value) in (int, float, bool)})
        atomic_json(output / "split_check.json", split)
        atomic_json(output / "run.json", {"status": "complete", "report_sha256": sha256_file(output / "report.json")})
        return report
    except BaseException as exc:
        atomic_json(output / "run.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("artifact", "checkpoint", "config", "simulator-root", "output-dir", "patient-name"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--patient-type", choices=("t1d", "t2d", "t2d_no_pump"), default="t1d")
    parser.add_argument("--condition", choices=("none", "rule_based", "predictive", "predictive_static"), required=True)
    for flag in ("env-seed", "action-seed", "exploration-seed"):
        parser.add_argument("--" + flag, type=int, required=True)
    parser.add_argument("--horizon-days", type=float, default=1)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--exploration-probability", type=float)
    parser.add_argument("--forecast-penalty-scale", type=float, default=1.0,
                        help="scale the incremental forecast mask (predictive only): 0=shadow/static, 1=legacy")
    parser.add_argument("--exclude-episodes", nargs="*", default=[])
    args = vars(parser.parse_args(argv))
    args["artifact_dir"] = args.pop("artifact")
    print(json.dumps(run_rollout(**args)["metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
