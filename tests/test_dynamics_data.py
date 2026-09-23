"""Causal timing, episode boundaries, split integrity and artifact failure gates."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from glucoalg.dynamics.data import (
    Episode, EXECUTION_SEMANTICS, FEATURES, SCHEMA, SCHEMA_VERSION,
    build_context_query, causal_windows, episode_metrics, load_episode, load_episodes,
    online_context_query, save_episode, sha256_file, validate_episode,
)


def make_episode(n=10, *, split="train", seed=123, episode_id="example"):
    obs = np.zeros((n + 1, 14), dtype=np.float32)
    obs[:, 0] = 100 + np.arange(n + 1) ** 2
    obs[:, 1] = np.arange(n + 1)
    actions = np.stack((np.arange(n) % 5, (np.arange(n) + 2) % 5), axis=1)
    accepted = np.ones((n, 2), dtype=bool)
    accepted[::2, 0] = False
    terminated = np.zeros(n, dtype=bool)
    terminated[-1] = True
    metadata = dict(schema=SCHEMA, schema_version=SCHEMA_VERSION, episode_id=episode_id,
                    patient_type="t1d", patient_name="adolescent#001", seed=seed,
                    action_seed=seed, exploration_seed=seed + 1000, split=split,
                    controller_interval_minutes=5, horizon_steps=288,
                    observation_features=list(FEATURES), execution_semantics=EXECUTION_SEMANTICS,
                    continuation_policy="fixed synthetic policy, no exploration",
                    policy=dict(checkpoint="/fake/model.pt", checkpoint_sha256="a" * 64,
                                config="/fake/config.json", config_sha256="b" * 64),
                    behavior=dict(action_mode="stochastic", exploration_probability=0.0,
                                  exploration_rng="numpy.PCG64", description="synthetic test policy"),
                    runtime={"glucosim_commit": "fake-simulator", "glucosim_content_sha256": "c" * 64}, metrics={})
    episode = Episode(obs, actions, np.where(accepted, actions, 0), accepted,
                      np.ones(n), np.full(n, 2.0), terminated, np.zeros(n, dtype=bool), metadata)
    return replace(episode, metadata=dict(metadata, metrics=episode_metrics(episode, termination_cause=1)))


def transitions_for(episode, t):
    return [SimpleNamespace(pre_observation=tuple(episode.observations[u]),
                            executed_action=tuple(episode.executed_actions[u]),
                            next_observation=tuple(episode.observations[u + 1]),
                            recommended_action=tuple(episode.recommended_actions[u]),
                            accepted=tuple(episode.accepted[u])) for u in range(t)]


def test_causal_quadratic_targets_and_history_timing():
    episode = make_episode()
    x, y, anchors = causal_windows(episode.observations, episode.recommended_actions,
                                   history_length=3, horizon_steps=2)
    np.testing.assert_array_equal(anchors, np.arange(2, 9))
    assert x.shape == (7, 3, 24)
    np.testing.assert_array_equal(x[0, :, 0], [100, 101, 104])
    np.testing.assert_array_equal(y[0], [5, 12])  # CGM[t+1/2] - CGM[t], not increments
    np.testing.assert_array_equal(x[0, -1, 14:19], np.eye(5)[2])
    np.testing.assert_array_equal(x[0, -1, 19:], np.eye(5)[4])
    assert x[0, -1, 16] == 1  # recommendation bolus2 despite rejection to executed0


def test_incomplete_horizons_and_warmup_drop_without_padding():
    episode = make_episode(3)
    x, y, anchors = causal_windows(episode.observations, episode.recommended_actions,
                                   history_length=3, horizon_steps=2)
    assert x.shape == (0, 3, 24) and y.shape == (0, 2) and anchors.shape == (0,)
    with pytest.raises(ValueError, match="Insufficient"):
        build_context_query(episode.observations, episode.recommended_actions, query_index=2,
                            history_length=2, horizon_steps=2, context_size=2)


def test_online_and_offline_contexts_queries_identical_and_causal():
    episode = make_episode()
    t = 6
    candidates = np.array([[0, 1], [3, 4]])
    offline = build_context_query(episode.observations, episode.recommended_actions, query_index=t,
                                  history_length=3, horizon_steps=2, context_size=2, candidate_actions=candidates)
    online = online_context_query(transitions_for(episode, t), episode.observations[t], candidates,
                                  history_length=3, horizon_steps=2, context_size=2)
    for name in ("context_x", "context_y", "query_x"):
        np.testing.assert_array_equal(getattr(offline, name), getattr(online, name))
    np.testing.assert_array_equal(offline.context_x[:, -1, 0], [109, 116])
    np.testing.assert_array_equal(offline.context_y, [[7, 16], [9, 20]])
    np.testing.assert_array_equal(offline.query_y, [13, 28])
    assert online.query_y is None
    changed_future = episode.observations.copy()
    changed_future[t + 1:, 0] += 10000
    changed = build_context_query(changed_future, episode.recommended_actions, query_index=t,
                                 history_length=3, horizon_steps=2, context_size=2, candidate_actions=candidates)
    for name in ("context_x", "context_y", "query_x"):
        np.testing.assert_array_equal(getattr(offline, name), getattr(changed, name))


@pytest.mark.parametrize("damage", ["missing_recommendation", "discontinuity", "different_current", "warmup"])
def test_online_rejects_incomplete_or_noncausal_history(damage):
    episode = make_episode()
    past = transitions_for(episode, 6)
    current = episode.observations[6].copy()
    if damage == "missing_recommendation":
        past[-1].recommended_action = None
    elif damage == "discontinuity":
        past[-1].pre_observation = tuple(episode.observations[0])
    elif damage == "different_current":
        current[0] += 1
    else:
        past = past[:2]
    with pytest.raises(ValueError):
        online_context_query(past, current, np.array([[1, 1]]), history_length=3, horizon_steps=2, context_size=2)


def test_snapshot_prevents_action_and_metadata_alias_mutation():
    episode = make_episode()
    raw = episode.recommended_actions.copy()
    metadata = dict(episode.metadata)
    detached = replace(episode, recommended_actions=raw, metadata=metadata)
    raw[:] = 4
    metadata["policy"]["checkpoint"] = "mutated"
    assert not np.all(detached.recommended_actions == 4)
    assert detached.metadata["policy"]["checkpoint"] == "/fake/model.pt"
    with pytest.raises(ValueError, match="read-only"):
        detached.recommended_actions[0, 0] = 3


def test_round_trip_hashes_and_exclusive_save(tmp_path):
    episode = make_episode()
    path = save_episode(tmp_path / "episode", episode)
    loaded = load_episode(path, expected_split="train")
    np.testing.assert_array_equal(episode.observations, loaded.observations)
    assert loaded.metadata["data_sha256"] == sha256_file(path / "episode.npz")
    with pytest.raises(FileExistsError):
        save_episode(path, episode)
    assert load_episode(path).metadata == loaded.metadata


def test_corrupt_artifact_and_old_schema_rejected(tmp_path):
    path = save_episode(tmp_path / "episode", make_episode())
    with (path / "episode.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_episode(path)
    episode = make_episode()
    for key, value in (("schema_version", 0), ("schema", "legacy-postaction")):
        with pytest.raises(ValueError, match="schema"):
            validate_episode(replace(episode, metadata=dict(episode.metadata, **{key: value})))


def test_explicit_episode_split_and_duplicate_gates(tmp_path):
    train = save_episode(tmp_path / "train", make_episode())
    val = save_episode(tmp_path / "val", make_episode(split="validation", seed=456, episode_id="val"))
    assert len(load_episodes([train], expected_split="train")) == 1
    with pytest.raises(ValueError, match="split"):
        load_episodes([train, val], expected_split="train")
    with pytest.raises(ValueError, match="Duplicate"):
        load_episodes([train, train], expected_split="train")
    with pytest.raises(ValueError, match="At least"):
        load_episodes([], expected_split="train")


@pytest.mark.parametrize("field,value", [
    ("observations", np.ones((10, 14))), ("observations", np.full((11, 14), np.inf)),
    ("recommended_actions", np.zeros((10, 2), dtype=float)),
    ("recommended_actions", np.full((10, 2), 5)),
    ("executed_actions", np.full((10, 2), 4)),
    ("accepted", np.ones((10, 2), dtype=int)),
    ("costs", -np.ones(10)), ("rewards", np.full(10, np.nan)),
    ("terminated", np.ones(10, dtype=bool)), ("terminated", np.zeros(10, dtype=bool)),
])
def test_array_failure_gates(field, value):
    with pytest.raises(ValueError):
        validate_episode(replace(make_episode(), **{field: value}))


@pytest.mark.parametrize("changes", [
    {"patient_name": "adolescent#999"}, {"patient_name": "adolescent#٠٠١"}, {"seed": True},
    {"split": "random-window"}, {"controller_interval_minutes": 1}, {"horizon_steps": 287},
    {"continuation_policy": ""}, {"runtime": {}}, {"metrics": {}},
    {"execution_semantics": "delivered exact dose"},
])
def test_metadata_failure_gates(changes):
    episode = make_episode()
    with pytest.raises(ValueError):
        validate_episode(replace(episode, metadata=dict(episode.metadata, **changes)))


def test_hashed_object_payload_never_unpickled(tmp_path):
    path = save_episode(tmp_path / "episode", make_episode())
    episode = make_episode()
    from glucoalg.dynamics.data import ARRAY_NAMES
    arrays = {name: getattr(episode, name) for name in ARRAY_NAMES}
    arrays["observations"] = np.full((11, 14), object(), dtype=object)
    np.savez(path / "episode.npz", **arrays)
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["data_sha256"] = sha256_file(path / "episode.npz")
    (path / "metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"] = {name: sha256_file(path / name) for name in ("metadata.json", "episode.npz")}
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_episode(path)
