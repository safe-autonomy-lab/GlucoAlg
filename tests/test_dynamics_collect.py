"""Fake-environment collection validates outcome evidence without JAX runs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glucoalg.dynamics import collect
from glucoalg.dynamics.data import load_episode
from test_dynamics_data import make_episode


class FakeEnv:
    action_space = SimpleNamespace(nvec=np.array([5, 5]))
    sample_time = 5
    simulation_minutes = 1440

    def __init__(self, *, info_change=None, mutate=False, cost=2.0, reward=1.0, interval=5):
        self.info_change = info_change or {}
        self.mutate = mutate
        self.cost = cost
        self.reward = reward
        self.sample_time = interval
        self.closed = False
        self.seeds = []
        self.actions = []
        self.n = 0

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.n = 0
        return torch.tensor([[100.0] + [0.0] * 13]), {}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        if self.mutate:
            action[:] = 0
        self.n += 1
        if self.n > 3:
            raise AssertionError("collector stepped after terminal")
        info = dict(bolus_accepted=np.array([False]), meal_accepted=np.array([True]),
                    termination_cause=np.array([1 if self.n == 3 else 0]))
        info.update(self.info_change)
        return (torch.tensor([[100.0 + self.n] + [0.0] * 13]),
                torch.tensor([self.reward]), torch.tensor([self.cost]),
                torch.tensor([float(self.n == 3)]), torch.tensor([0.0]), info)

    def close(self):
        self.closed = True


class FakeActor:
    shield = None

    def __call__(self, normalized, raw):
        self.normalized = normalized.clone()
        self.raw = raw.clone()
        return SimpleNamespace(mode=torch.tensor([[2, 4]]), sample=lambda: torch.tensor([[2, 4]]))


def test_collection_records_pre_and_post_observation_and_distinct_actions():
    env = FakeEnv(mutate=True)
    episode = collect.collect_episode(env, FakeActor(), None, metadata=make_episode().metadata)
    np.testing.assert_array_equal(episode.observations[:, 0], [100, 101, 102, 103])
    np.testing.assert_array_equal(episode.recommended_actions, [[2, 4]] * 3)
    np.testing.assert_array_equal(episode.executed_actions, [[0, 4]] * 3)
    np.testing.assert_array_equal(episode.accepted, [[False, True]] * 3)
    assert env.seeds == [123]
    assert episode.metadata["metrics"]["coverage_fraction"] == 3 / 288
    assert episode.metadata["metrics"]["cost"] == 6
    assert episode.metadata["metrics"]["early_termination"] is True


def test_policy_normalizer_never_changes_raw_record():
    normalizer = SimpleNamespace(normalize=lambda raw: raw * 0.1)
    actor = FakeActor()
    episode = collect.collect_episode(FakeEnv(), actor, normalizer, metadata=make_episode().metadata)
    assert actor.normalized[0, 0] == pytest.approx(10.2)
    assert actor.raw[0, 0] == 102
    assert episode.observations[-2, 0] == 102


@pytest.mark.parametrize("flags", [
    {}, {"bolus_accepted": True}, {"bolus_accepted": 1, "meal_accepted": True},
    {"bolus_accepted": False, "meal_accepted": 0.0},
    {"bolus_accepted": [True, False], "meal_accepted": True},
    {"bolus_accepted": "true", "meal_accepted": True},
])
def test_acceptance_requires_authoritative_boolean_flags(flags):
    with pytest.raises(ValueError):
        collect.accepted_execution(np.array([2, 3]), flags)


@pytest.mark.parametrize("env", [FakeEnv(cost=-1), FakeEnv(cost=float("inf")),
                                  FakeEnv(reward=float("nan")), FakeEnv(interval=1),
                                  FakeEnv(info_change={"meal_accepted": 1.0}),
                                  FakeEnv(info_change={"termination_cause": 0.5})])
def test_collector_rejects_invalid_metrics_or_simulator_contract(env):
    with pytest.raises(ValueError):
        collect.collect_episode(env, FakeActor(), None, metadata=make_episode().metadata)


def test_exploration_rng_reproducible_and_separate_from_policy():
    metadata = make_episode().metadata
    metadata["behavior"]["exploration_probability"] = 1.0
    first = collect.collect_episode(FakeEnv(), FakeActor(), None, metadata=metadata)
    second = collect.collect_episode(FakeEnv(), FakeActor(), None, metadata=metadata)
    np.testing.assert_array_equal(first.recommended_actions, second.recommended_actions)
    assert not np.all(first.recommended_actions == [2, 4])
    metadata["exploration_seed"] += 1
    third = collect.collect_episode(FakeEnv(), FakeActor(), None, metadata=metadata)
    assert not np.array_equal(first.recommended_actions, third.recommended_actions)


def setup_runner(tmp_path, monkeypatch, *, env=None):
    env = env or FakeEnv()
    checkpoint, config = tmp_path / "policy.pt", tmp_path / "config.json"
    checkpoint.write_bytes(b"fake policy bytes")
    config.write_text("{}")
    simulator_file = tmp_path / "fake-simulator" / "glucosim" / "__init__.py"
    simulator_file.parent.mkdir(parents=True)
    simulator_file.write_text("# fake simulator source\n")
    monkeypatch.setattr(collect, "initialize_simulator", lambda root: {
        "glucosim_commit": "fake", "glucosim_file": str(simulator_file)})
    monkeypatch.setattr(collect, "_load_policy", lambda *args: (env, FakeActor(), None))
    kwargs = dict(checkpoint=checkpoint, config=config, simulator_root="/fake/simulator",
                  output_dir=tmp_path / "collection", patient_type="t1d", patient_name="adolescent#001",
                  split="train", episodes=2, env_seed_base=700, action_seed_base=1700,
                  exploration_seed_base=10700, horizon_days=1, action_mode="stochastic",
                  exploration_probability=0.1)
    return env, kwargs


def test_complete_collection_manifest_and_explicit_seeds(tmp_path, monkeypatch):
    env, kwargs = setup_runner(tmp_path, monkeypatch)
    result = collect.run_collection(**kwargs)
    assert result["status"] == "complete"
    assert env.closed and env.seeds == [700, 701]
    paths = [kwargs["output_dir"] / row["path"] for row in result["episodes"]]
    episodes = [load_episode(path, expected_split="train") for path in paths]
    assert [episode.metadata["action_seed"] for episode in episodes] == [1700, 1701]
    assert [episode.metadata["exploration_seed"] for episode in episodes] == [10700, 10701]
    assert episodes[0].metadata["continuation_policy"] == episodes[1].metadata["continuation_policy"]
    assert "noise" in episodes[0].metadata["execution_semantics"]
    with pytest.raises(FileExistsError):
        collect.run_collection(**kwargs)


def test_failure_closes_env_and_cannot_publish_success(tmp_path, monkeypatch):
    env, kwargs = setup_runner(tmp_path, monkeypatch, env=FakeEnv(info_change={"meal_accepted": 1}))
    with pytest.raises(ValueError, match="boolean"):
        collect.run_collection(**kwargs)
    manifest = json.loads((kwargs["output_dir"] / "collection.json").read_text())
    assert manifest["status"] == "failed" and manifest["episodes"] == [] and env.closed


def test_checkpoint_mutation_during_collection_rejected(tmp_path, monkeypatch):
    env, kwargs = setup_runner(tmp_path, monkeypatch)
    original = env.step

    def step(action):
        kwargs["checkpoint"].write_bytes(b"changed")
        return original(action)

    env.step = step
    with pytest.raises(ValueError, match="changed during collection"):
        collect.run_collection(**kwargs)
    assert env.closed
    assert not list(kwargs["output_dir"].glob("episode-*"))


def test_source_mutation_during_collection_rejected(tmp_path, monkeypatch):
    env, kwargs = setup_runner(tmp_path, monkeypatch)
    states = iter([{"sha256": "a" * 64, "simulator_content_sha256": "a" * 64},
                   {"sha256": "b" * 64, "simulator_content_sha256": "b" * 64}])
    monkeypatch.setattr(collect, "_source_hashes", lambda runtime: next(states))
    with pytest.raises(ValueError, match="sources changed"):
        collect.run_collection(**kwargs)
    assert env.closed


@pytest.mark.parametrize("override", [dict(episodes=0), dict(horizon_days=0.5), dict(split="random"),
                                       dict(action_seed_base=2**32 - 1), dict(exploration_probability=float("nan")),
                                       dict(exploration_probability=True), dict(action_mode="greedy")])
def test_invalid_protocol_fails_before_environment_creation(tmp_path, monkeypatch, override):
    env, kwargs = setup_runner(tmp_path, monkeypatch)
    kwargs.update(override)
    with pytest.raises(ValueError):
        collect.run_collection(**kwargs)
    assert env.seeds == []
    assert not kwargs["output_dir"].exists()


def test_cli_explicit_paths_and_sampling_defaults():
    args = collect.build_parser().parse_args([
        "--checkpoint", "model.pt", "--config", "config.json", "--simulator-root", "/sim",
        "--output-dir", "/new", "--patient-type", "t1d", "--patient-name", "adult#001", "--split", "train"])
    assert args.action_mode == "stochastic"
    assert args.horizon_days == 1 and args.exploration_probability == 0
