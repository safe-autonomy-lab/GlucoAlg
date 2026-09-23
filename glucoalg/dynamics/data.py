"""Versioned causal episodes and shared offline/online recommendation windows.

The first observation precedes action zero. Targets are future measured CGM
minus current measured CGM; this module performs no learned normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np

SCHEMA = "glucoalg.causal_episode"
SCHEMA_VERSION = 1
FEATURES = (
    "cgm", "iob", "cob", "cgm_trend", "time_sin", "time_cos",
    "time_since_meal", "time_since_bolus", "planned_meal_left",
    "meal_count_norm", "bolus_count_norm", "time_until_meal_norm",
    "next_meal_size_norm", "is_pre_bolus_window",
)
SPLITS = ("train", "validation", "test")
EXECUTION_SEMANTICS = (
    "Accepted controller recommendation level indices: rejected components become zero. "
    "These are not delivered physical doses. GlucoSim subsequently adds bolus/meal "
    "amount noise; basal insulin and autonomous scenario meals are separate."
)


@dataclass(frozen=True)
class Episode:
    observations: np.ndarray
    recommended_actions: np.ndarray
    executed_actions: np.ndarray
    accepted: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict

    def __post_init__(self):
        # Detach all buffers and metadata from the collector/NPZ lifetime.
        for name in ARRAY_NAMES:
            snapshot = np.array(getattr(self, name), copy=True)
            snapshot.setflags(write=False)
            object.__setattr__(self, name, snapshot)
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))


ARRAY_NAMES = tuple(field.name for field in fields(Episode) if field.name != "metadata")


@dataclass(frozen=True)
class ContextQuery:
    context_x: np.ndarray
    context_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray | None


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers with shape {shape}")
    return array


def _actions(value, n, name):
    array = _finite_array(value, (n, 2), name)
    if array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array >= 5):
        raise ValueError(f"{name} must contain integer indices in [0, 4], in bolus/meal order")
    return array


def _observations(value):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 14 or len(array) < 1:
        raise ValueError("observations must have shape [N+1,14]")
    return _finite_array(array, array.shape, "observations")


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 hex digest")


def episode_metrics(episode: Episode, *, termination_cause: int) -> dict:
    """Recomputable coverage, cost and post-step measured-CGM safety counts."""
    n = len(episode.rewards)
    if n == 0:
        raise ValueError("Cannot score an empty episode")
    cgm = episode.observations[1:, 0].astype(np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            reward = math.fsum(float(value) for value in episode.rewards)
            cost = math.fsum(float(value) for value in episode.costs)
        except (OverflowError, FloatingPointError) as exc:
            raise ValueError("Non-finite episode reward/cost aggregate") from exc
    if not math.isfinite(reward) or not math.isfinite(cost):
        raise ValueError("Non-finite episode reward/cost aggregate")
    horizon = episode.metadata["horizon_steps"]
    return {
        "length": n, "horizon_steps": horizon, "coverage_fraction": n / horizon,
        "reward": reward, "cost": cost,
        "terminated": bool(episode.terminated[-1]), "truncated": bool(episode.truncated[-1]),
        "early_termination": n < horizon, "termination_cause": termination_cause,
        "glucose_source": "post-step measured CGM; reset excluded, final included",
        "time_in_range_pct": 100 * int(np.count_nonzero((cgm >= 70) & (cgm <= 180))) / n,
        "time_below_range_pct": 100 * int(np.count_nonzero(cgm < 70)) / n,
        "time_above_range_pct": 100 * int(np.count_nonzero(cgm > 180)) / n,
        "severe_hypo_samples": int(np.count_nonzero(cgm < 54)),
        "hyper_250_samples": int(np.count_nonzero(cgm > 250)),
    }


def validate_episode(episode: Episode, *, expected_split: str | None = None) -> None:
    if not isinstance(episode, Episode):
        raise ValueError("Expected an Episode")
    observations = _observations(episode.observations)
    n = len(observations) - 1
    if n < 1:
        raise ValueError("An episode needs at least one completed transition")
    recommendations = _actions(episode.recommended_actions, n, "recommended_actions")
    executed = _actions(episode.executed_actions, n, "executed_actions")
    for name, shape in (("accepted", (n, 2)), ("terminated", (n,)), ("truncated", (n,))):
        array = getattr(episode, name)
        if array.shape != shape or array.dtype.kind != "b":
            raise ValueError(f"{name} must be boolean with shape {shape}")
    if not np.array_equal(executed, np.where(episode.accepted, recommendations, 0)):
        raise ValueError("executed_actions disagree with authoritative acceptance flags")
    for name in ("rewards", "costs"):
        _finite_array(getattr(episode, name), (n,), name)
    if np.any(episode.costs < 0):
        raise ValueError("costs must be nonnegative")
    ended = episode.terminated | episode.truncated
    if np.any(ended[:-1]) or not ended[-1]:
        raise ValueError("Episode must end on its sole terminal/truncated row; no rows after terminal")
    metadata = episode.metadata
    required = {
        "schema", "schema_version", "episode_id", "patient_type", "patient_name", "seed",
        "action_seed", "exploration_seed", "split", "controller_interval_minutes", "horizon_steps",
        "continuation_policy", "policy", "behavior", "runtime", "metrics", "observation_features",
        "execution_semantics",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Episode metadata missing keys: {sorted(missing)}")
    if metadata["schema"] != SCHEMA or type(metadata["schema_version"]) is not int or metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported causal episode schema; legacy post-action datasets cannot be relabeled")
    if not isinstance(metadata["episode_id"], str) or not metadata["episode_id"].strip():
        raise ValueError("episode_id must be a nonempty string")
    if metadata["patient_type"] not in ("t1d", "t2d", "t2d_no_pump") or not isinstance(metadata["patient_name"], str) or re.fullmatch(r"(child|adolescent|adult)#(00[1-9]|010)", metadata["patient_name"]) is None:
        raise ValueError("Invalid complete simulator patient identity")
    for name in ("seed", "action_seed", "exploration_seed"):
        if type(metadata[name]) is not int or not 0 <= metadata[name] < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32)")
    if metadata["split"] not in SPLITS:
        raise ValueError("split must be train, validation or test")
    if expected_split is not None and (expected_split not in SPLITS or metadata["split"] != expected_split):
        raise ValueError(f"Episode split {metadata['split']!r} does not match expected split {expected_split!r}")
    interval = metadata["controller_interval_minutes"]
    if type(interval) not in (int, float) or interval != 5:
        raise ValueError("controller_interval_minutes must be 5 for this schema")
    horizon = _positive_int(metadata["horizon_steps"], "horizon_steps")
    if horizon < 288 or n > horizon:
        raise ValueError("horizon_steps must cover at least one day and all episode rows")
    if tuple(metadata["observation_features"]) != FEATURES:
        raise ValueError("Unexpected observation_features order")
    if metadata["execution_semantics"] != EXECUTION_SEMANTICS:
        raise ValueError("execution_semantics must identify accepted controller indices and physical-dose limits")
    if not isinstance(metadata["continuation_policy"], str) or not metadata["continuation_policy"].strip():
        raise ValueError("continuation_policy must describe future recommendations")
    policy = metadata["policy"]
    if not isinstance(policy, dict):
        raise ValueError("policy must contain checkpoint/config paths and hashes")
    for name in ("checkpoint", "config"):
        if not isinstance(policy.get(name), str) or not policy[name].strip():
            raise ValueError(f"policy.{name} path is required")
        _hash(policy.get(name + "_sha256"), "policy." + name + "_sha256")
    behavior = metadata["behavior"]
    if not isinstance(behavior, dict) or behavior.get("action_mode") not in ("stochastic", "deterministic"):
        raise ValueError("behavior.action_mode must be stochastic or deterministic")
    probability = behavior.get("exploration_probability")
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        raise ValueError("behavior.exploration_probability must be in [0, 1]")
    if behavior.get("exploration_rng") != "numpy.PCG64" or not isinstance(behavior.get("description"), str) or not behavior["description"].strip():
        raise ValueError("behavior must identify exploration RNG and describe collection")
    runtime = metadata["runtime"]
    if not isinstance(runtime, dict) or not isinstance(runtime.get("glucosim_commit"), str) or not runtime["glucosim_commit"].strip():
        raise ValueError("runtime provenance must identify the simulator commit")
    _hash(runtime.get("glucosim_content_sha256"), "runtime.glucosim_content_sha256")
    metrics = metadata["metrics"]
    if not isinstance(metrics, dict) or type(metrics.get("termination_cause")) is not int or metrics["termination_cause"] < 0:
        raise ValueError("metrics must include nonnegative integer termination_cause")
    expected = episode_metrics(episode, termination_cause=metrics["termination_cause"])
    if metrics != expected:
        raise ValueError("Episode safety/coverage metrics disagree with recorded arrays")
    if "data_sha256" in metadata:
        _hash(metadata["data_sha256"], "data_sha256")
    # Reject nested NaN/Infinity and non-JSON provenance as well.
    json.dumps(metadata, allow_nan=False)


def claim_directory(path: str | Path) -> Path:
    """Atomically reserve a new or empty directory; failures remain claimed."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".claim").open("x", encoding="utf-8") as stream:
        stream.write(SCHEMA + "\n")
    if any(child.name != ".claim" for child in path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    return path


def save_episode(path: str | Path, episode: Episode) -> Path:
    """Write one new artifact; manifest is published last and is required to load."""
    validate_episode(episode)
    path = claim_directory(path)
    with (path / "episode.npz").open("xb") as stream:
        np.savez_compressed(stream, **{name: getattr(episode, name) for name in ARRAY_NAMES})
    metadata = dict(episode.metadata, data_sha256=sha256_file(path / "episode.npz"))
    with (path / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    manifest = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "episode_id": metadata["episode_id"],
                "files": {name: sha256_file(path / name) for name in ("episode.npz", "metadata.json")}}
    with (path / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def load_episode(path: str | Path, *, expected_split: str | None = None) -> Episode:
    path = Path(path)
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != SCHEMA or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported causal episode manifest schema")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != {"episode.npz", "metadata.json"}:
        raise ValueError("Episode manifest must hash exactly episode.npz and metadata.json")
    for name, digest in manifest["files"].items():
        _hash(digest, name)
        if sha256_file(path / name) != digest:
            raise ValueError(f"Episode artifact hash mismatch: {name}")
    with (path / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("data_sha256") != manifest["files"]["episode.npz"] or metadata.get("episode_id") != manifest.get("episode_id"):
        raise ValueError("Episode metadata and manifest disagree")
    with np.load(path / "episode.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(ARRAY_NAMES):
            raise ValueError("Unexpected episode arrays; legacy post-action datasets are unsupported")
        episode = Episode(**{name: arrays[name] for name in ARRAY_NAMES}, metadata=metadata)
    validate_episode(episode, expected_split=expected_split)
    return episode


def load_episodes(paths: Iterable[str | Path], *, expected_split: str) -> list[Episode]:
    """Load explicitly assigned whole episodes, with no window-level split."""
    if expected_split not in SPLITS:
        raise ValueError("expected_split must be train, validation or test")
    episodes, seen_ids, seen_seeds, seen_hashes = [], set(), set(), set()
    for path in paths:
        episode = load_episode(path, expected_split=expected_split)
        meta = episode.metadata
        seed_identity = (meta["patient_type"], meta["patient_name"], meta["seed"])
        if meta["episode_id"] in seen_ids or seed_identity in seen_seeds or meta["data_sha256"] in seen_hashes:
            raise ValueError("Duplicate episode identity, patient/environment seed, or data hash")
        seen_ids.add(meta["episode_id"])
        seen_seeds.add(seed_identity)
        seen_hashes.add(meta["data_sha256"])
        episodes.append(episode)
    if not episodes:
        raise ValueError("At least one explicit episode directory is required")
    return episodes


def _features(observations, recommendations):
    """Raw 14 observations followed by two independent five-way one-hots."""
    result = np.concatenate((observations, np.eye(5)[recommendations[:, 0]],
                             np.eye(5)[recommendations[:, 1]]), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite raw recommendation features")
    return result


def _window(x, t, length):
    return x[t - length + 1:t + 1].copy()


def _deltas(observations, t, horizon):
    with np.errstate(over="raise", invalid="raise"):
        try:
            target = observations[t + 1:t + horizon + 1, 0].astype(np.float64) - float(observations[t, 0])
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("Non-finite cumulative glucose target") from exc
    if not np.isfinite(target).all():
        raise ValueError("Non-finite cumulative glucose target")
    return target


def causal_windows(observations, recommended_actions, *, history_length: int, horizon_steps: int):
    """Return (x[M,L,24], delta_CGM[M,H], action_indices[M]) without padding."""
    length = _positive_int(history_length, "history_length")
    horizon = _positive_int(horizon_steps, "horizon_steps")
    obs = _observations(observations)
    actions = _actions(recommended_actions, len(obs) - 1, "recommended_actions")
    anchors = np.arange(length - 1, len(obs) - horizon, dtype=np.int64)
    x = _features(obs[:-1], actions)
    if not len(anchors):
        return np.empty((0, length, 24)), np.empty((0, horizon)), anchors
    return np.stack([_window(x, t, length) for t in anchors]), np.stack([_deltas(obs, t, horizon) for t in anchors]), anchors


def build_context_query(observations, recommended_actions, *, query_index: int,
                        history_length: int, horizon_steps: int, context_size: int,
                        candidate_actions=None) -> ContextQuery:
    """Contexts end at t-H; every context target is observed before the query.

    Online calls provide N completed recommendations and query_index=N with
    candidate_actions. Offline query targets are returned only when complete;
    query_y always describes the logged recommendation's observed outcome,
    never a counterfactual label for an alternative candidate.
    """
    length = _positive_int(history_length, "history_length")
    horizon = _positive_int(horizon_steps, "horizon_steps")
    count = _positive_int(context_size, "context_size")
    obs = _observations(observations)
    actions = _actions(recommended_actions, len(obs) - 1, "recommended_actions")
    if not isinstance(query_index, (int, np.integer)) or isinstance(query_index, (bool, np.bool_)):
        raise ValueError("query_index must index an observed state")
    t = int(query_index)
    if not 0 <= t < len(obs):
        raise ValueError("query_index must index an observed state")
    context_first = t - horizon - count + 1
    if context_first < length - 1:
        raise ValueError(f"Insufficient completed history: need {length + horizon + count - 2} transitions")
    if candidate_actions is None:
        if t >= len(actions):
            raise ValueError("Current candidate recommendations are required for an online query")
        candidates = actions[t:t + 1]
    else:
        candidates = np.asarray(candidate_actions)
        if candidates.ndim != 2 or len(candidates) < 1:
            raise ValueError("candidate_actions must have shape [C,2] with C >= 1")
        candidates = _actions(candidates, len(candidates), "candidate_actions")
    x = _features(obs[:-1], actions)
    context_anchors = range(context_first, t - horizon + 1)
    context_x = np.stack([_window(x, u, length) for u in context_anchors])
    context_y = np.stack([_deltas(obs, u, horizon) for u in context_anchors])
    candidate_x = _features(np.repeat(obs[t:t + 1], len(candidates), axis=0), candidates)
    query_x = np.stack([np.concatenate((x[t - length + 1:t], row[None]), axis=0) for row in candidate_x])
    query_y = _deltas(obs, t, horizon) if t + horizon < len(obs) else None
    return ContextQuery(context_x, context_y, query_x, query_y)


def online_context_query(past_transitions, current_observation, candidate_actions, *,
                         history_length: int, horizon_steps: int, context_size: int) -> ContextQuery:
    """Reconstruct identical causal windows from immutable raw shield requests."""
    transitions = tuple(past_transitions)
    required = (_positive_int(history_length, "history_length") +
                _positive_int(horizon_steps, "horizon_steps") +
                _positive_int(context_size, "context_size") - 2)
    if len(transitions) < required:
        raise ValueError(f"Insufficient completed history: need {required} transitions")
    transitions = transitions[-required:]
    observations, recommendations = [], []
    for index, transition in enumerate(transitions):
        pre = _finite_array(transition.pre_observation, (14,), "pre_observation")
        post = _finite_array(transition.next_observation, (14,), "next_observation")
        if transition.recommended_action is None:
            raise ValueError("Past recommended_action is required; executed actions cannot substitute")
        recommendation = _actions(np.asarray(transition.recommended_action)[None], 1, "recommended_action")[0]
        if index and not np.array_equal(observations[-1], pre):
            raise ValueError("Completed transition observations are not contiguous")
        if not index:
            observations.append(pre.copy())
        observations.append(post.copy())
        recommendations.append(recommendation.copy())
    current = _finite_array(current_observation, (14,), "current_observation")
    if not np.array_equal(observations[-1], current):
        raise ValueError("Current observation differs from the last completed outcome")
    return build_context_query(np.stack(observations), np.stack(recommendations), query_index=required,
                               history_length=history_length, horizon_steps=horizon_steps,
                               context_size=context_size, candidate_actions=candidate_actions)
