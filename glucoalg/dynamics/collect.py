"""Collect fresh causal GlucoSim episodes from an explicit unshielded policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from glucoalg.dynamics.data import (
    Episode, EXECUTION_SEMANTICS, FEATURES, SCHEMA, SCHEMA_VERSION, SPLITS,
    claim_directory, episode_metrics, save_episode, sha256_file, validate_episode,
)
from glucoalg.runtime import configure_runtime, initialize_simulator


def _scalar(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be one real scalar")
    result = float(array.item())
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}")
    return result


def _flag(value, name, *, strict_boolean=False):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1 or (strict_boolean and array.dtype.kind != "b"):
        raise ValueError(f"{name} must be an authoritative boolean scalar")
    number = _scalar(array, name)
    if number not in (0, 1):
        raise ValueError(f"{name} must be boolean or an exact 0/1 terminal flag")
    return bool(number)


def _observation(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape not in ((14,), (1, 14)) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Expected one finite raw 14-feature observation")
    return array.reshape(14).copy()


def accepted_execution(recommendation, info):
    """Decode locked GlucoSim controller levels, never infer physical dose.

    step.py first zeros rejected recommendations, then converts levels into
    patient-specific units and adds dose/meal noise. These indices describe
    only that first discrete acceptance stage. Missing/numeric flags fail.
    """
    action = np.asarray(recommendation)
    if action.shape != (2,) or action.dtype.kind not in "iu" or np.any(action < 0) or np.any(action > 4):
        raise ValueError("Recommendation must contain two integer indices in [0, 4]")
    if not isinstance(info, dict) or any(name not in info for name in ("bolus_accepted", "meal_accepted")):
        raise ValueError("Simulator did not expose authoritative bolus/meal acceptance flags")
    accepted = np.array([_flag(info[name], name, strict_boolean=True)
                         for name in ("bolus_accepted", "meal_accepted")], dtype=bool)
    return np.where(accepted, action, 0).astype(np.int64), accepted


def _recommend(actor, normalizer, observation, action_mode):
    import torch

    normalized = normalizer.normalize(observation.copy()) if normalizer is not None else observation
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized policy observation")
    with torch.inference_mode():
        raw = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        distribution = actor(torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0), raw)
        action = distribution.sample() if action_mode == "stochastic" else distribution.mode
    array = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
    if array.shape not in ((2,), (1, 2)) or array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array > 4):
        raise ValueError("Policy must produce two categorical indices in [0, 4]")
    return array.reshape(2).astype(np.int64, copy=True)


def collect_episode(env, actor, normalizer, *, metadata: dict) -> Episode:
    """Collect one complete episode. The caller owns/always closes the env."""
    from glucoalg.evaluation import set_seed

    if getattr(actor, "shield", None) is not None:
        raise ValueError("Causal collection currently requires an unshielded actor")
    if tuple(np.asarray(env.action_space.nvec).tolist()) != (5, 5):
        raise ValueError("Collection requires GlucoSim's (5,5) categorical action space")
    interval = _scalar(env.sample_time, "controller interval")
    if interval != 5:
        raise ValueError("Actual simulator controller interval must be 5 minutes")
    horizon = metadata["horizon_steps"]
    if _scalar(env.simulation_minutes, "simulation_minutes") != 5 * horizon:
        raise ValueError("Actual simulator horizon disagrees with collection metadata")
    set_seed(metadata["action_seed"])
    exploration = np.random.Generator(np.random.PCG64(metadata["exploration_seed"]))
    probability = metadata["behavior"]["exploration_probability"]
    mode = metadata["behavior"]["action_mode"]
    observations, recommendations, executed, accepted, rewards, costs, terminated, truncated = ([] for _ in range(8))
    obs, _ = env.reset(seed=metadata["seed"])
    observations.append(_observation(obs))
    for _ in range(horizon):
        # Always draw the policy proposal first; exploration uses a separate RNG.
        action = _recommend(actor, normalizer, observations[-1], mode)
        if exploration.random() < probability:
            action = exploration.integers(0, 5, size=2, dtype=np.int64)
        recommendation = action.copy()
        # Pass a different buffer: an environment must never mutate the proposal log.
        obs, reward, cost, terminal, timeout, info = env.step(action[None].copy())
        outcome, flags = accepted_execution(recommendation, info)
        reward_value, cost_value = _scalar(reward, "reward"), _scalar(cost, "cost")
        if cost_value < 0:
            raise ValueError("Simulator cost must be nonnegative")
        terminal_flag, timeout_flag = _flag(terminal, "terminated"), _flag(timeout, "truncated")
        observations.append(_observation(obs))
        recommendations.append(recommendation)
        executed.append(outcome)
        accepted.append(flags)
        rewards.append(reward_value)
        costs.append(cost_value)
        terminated.append(terminal_flag)
        truncated.append(timeout_flag)
        if terminal_flag or timeout_flag:
            break
    else:
        raise ValueError("Simulator did not terminate or truncate at its declared horizon")
    if "termination_cause" not in info:
        raise ValueError("Simulator did not expose termination_cause")
    cause = _scalar(info["termination_cause"], "termination_cause")
    if cause < 0 or cause != int(cause):
        raise ValueError("termination_cause must be a nonnegative integer")
    episode = Episode(observations=np.asarray(observations), recommended_actions=np.asarray(recommendations),
                      executed_actions=np.asarray(executed), accepted=np.asarray(accepted, dtype=bool),
                      rewards=np.asarray(rewards), costs=np.asarray(costs),
                      terminated=np.asarray(terminated, dtype=bool), truncated=np.asarray(truncated, dtype=bool),
                      metadata=metadata)
    episode = replace(episode, metadata=dict(metadata, metrics=episode_metrics(episode, termination_cause=int(cause))))
    validate_episode(episode, expected_split=metadata["split"])
    return episode


def _load_policy(checkpoint, config, patient_type, patient_name, seed, horizon_days):
    from glucoalg.evaluation import create_diabetes_env, load_model

    env = create_diabetes_env(patient_type, patient_name, seed=seed, horizon_days=horizon_days)
    try:
        actor, loaded_env, normalizer = load_model(checkpoint, config, env, shield_type="none")
        if loaded_env is not env:
            raise ValueError("Unexpected replacement environment from policy loader")
        return env, actor, normalizer
    except BaseException:
        env.close()
        raise


def _source_hashes(runtime):
    """Hash the collection implementation and selected simulator Python sources."""
    import glucoalg.evaluation
    import glucoalg.runtime
    import glucoalg.dynamics.data
    from glucoalg.tuning.plan import _content_hash

    files = [Path(__file__), Path(glucoalg.evaluation.__file__), Path(glucoalg.runtime.__file__),
             Path(glucoalg.dynamics.data.__file__)]
    simulator_file = runtime.get("glucosim_file")
    if simulator_file:
        files.extend(sorted(Path(simulator_file).parent.rglob("*.py")))
    hashes = {str(path.resolve()): sha256_file(path) for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"files": hashes, "sha256": digest,
            "simulator_content_sha256": _content_hash(Path(simulator_file).parent.parent, ("glucosim",))
            if simulator_file else None}


def run_collection(*, checkpoint, config, simulator_root, output_dir, patient_type, patient_name,
                   split, episodes=1, env_seed_base=22, action_seed_base=22,
                   exploration_seed_base=1022, horizon_days=1, action_mode="stochastic",
                   exploration_probability=0.0, continuation_policy=None):
    """Collect explicit whole-episode splits and commit a fail-closed manifest."""
    from glucoalg.eval_grid import validate_protocol

    validate_protocol(patient_type=patient_type, patients=[patient_name], episodes=episodes,
                      horizon_days=horizon_days, eval_seed_base=env_seed_base, action_mode=action_mode)
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    for name, seed in (("env_seed_base", env_seed_base), ("action_seed_base", action_seed_base),
                       ("exploration_seed_base", exploration_seed_base)):
        if type(seed) is not int or seed < 0 or seed + episodes > 2**32:
            raise ValueError(f"{name} and derived episode seeds must lie in [0, 2**32)")
    if type(exploration_probability) not in (int, float) or not 0 <= exploration_probability <= 1:
        raise ValueError("exploration_probability must be finite and in [0, 1]")
    if continuation_policy is not None and (not isinstance(continuation_policy, str) or not continuation_policy.strip()):
        raise ValueError("continuation_policy must be nonempty when provided")
    checkpoint, config = Path(checkpoint).expanduser().resolve(), Path(config).expanduser().resolve()
    policy = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
              "config": str(config), "config_sha256": sha256_file(config)}
    with config.open(encoding="utf-8") as stream:
        configuration = json.load(stream)
    patient_type = {"t2dnp": "t2d_no_pump"}.get(patient_type, patient_type)
    description = (
        f"Unshielded fixed categorical checkpoint {policy['checkpoint_sha256']}, configuration "
        f"{policy['config_sha256']}; {action_mode} policy proposal drawn each step, then independently "
        f"replaced with probability {exploration_probability:g} by uniform independent bolus/meal "
        "indices in [0,4]. Simulator acceptance, dose noise and autonomous meals remain active."
    )
    # The factual continuation contract is always preserved even with a user label.
    continuation = description if continuation_policy is None else description + " User description: " + continuation_policy
    output = claim_directory(Path(output_dir).expanduser().resolve())
    manifest = {
        "schema": "glucoalg.causal_collection", "schema_version": 1, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "split": split,
        "patient_type": patient_type, "patient_name": patient_name, "policy": policy,
        "expected_episodes": episodes, "episodes": [],
    }

    def write_manifest():
        temporary = output / ".collection.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(output / "collection.json")

    env = None
    write_manifest()
    try:
        configure_runtime()
        runtime = initialize_simulator(simulator_root)
        runtime = dict(runtime, collection_sources=_source_hashes(runtime))
        runtime["glucosim_content_sha256"] = runtime["collection_sources"]["simulator_content_sha256"]
        env, actor, normalizer = _load_policy(checkpoint, configuration, patient_type, patient_name,
                                               env_seed_base, horizon_days)
        for index in range(episodes):
            metadata = {
                "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "episode_id": uuid.uuid4().hex,
                "patient_type": patient_type, "patient_name": patient_name, "split": split,
                "seed": env_seed_base + index, "action_seed": action_seed_base + index,
                "exploration_seed": exploration_seed_base + index,
                "controller_interval_minutes": 5, "horizon_steps": round(float(horizon_days) * 288),
                "observation_features": list(FEATURES), "execution_semantics": EXECUTION_SEMANTICS,
                "policy": policy, "behavior": {"action_mode": action_mode,
                    "exploration_probability": exploration_probability, "exploration_rng": "numpy.PCG64",
                    "description": description}, "continuation_policy": continuation,
                "runtime": runtime, "metrics": {},
            }
            episode = collect_episode(env, actor, normalizer, metadata=metadata)
            if sha256_file(checkpoint) != policy["checkpoint_sha256"] or sha256_file(config) != policy["config_sha256"]:
                raise ValueError("Policy checkpoint/config changed during collection")
            if _source_hashes(runtime) != runtime["collection_sources"]:
                raise ValueError("Collection or simulator sources changed during collection")
            path = save_episode(output / f"episode-{index:06d}", episode)
            manifest["episodes"].append({"path": path.name, "episode_id": metadata["episode_id"],
                                          "manifest_sha256": sha256_file(path / "manifest.json"),
                                          "metrics": episode.metadata["metrics"]})
            write_manifest()
        manifest["status"] = "complete"
        write_manifest()
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest()
        raise
    finally:
        if env is not None:
            env.close()
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "simulator-root", "output-dir", "patient-name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--patient-type", choices=("t1d", "t2d", "t2dnp", "t2d_no_pump"), required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--env-seed-base", type=int, default=22)
    parser.add_argument("--action-seed-base", type=int, default=22)
    parser.add_argument("--exploration-seed-base", type=int, default=1022)
    parser.add_argument("--horizon-days", type=float, default=1)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--exploration-probability", type=float, default=0.0)
    parser.add_argument("--continuation-policy", help="Optional label appended to the factual fixed-policy continuation contract")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = run_collection(**vars(args))
    print(json.dumps({"status": manifest["status"], "episodes": len(manifest["episodes"]),
                      "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
