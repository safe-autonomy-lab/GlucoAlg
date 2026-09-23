"""Evaluation protocol, normalization and failure-gate regressions."""
import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from glucoalg import evaluation
from glucoalg.eval_grid import build_parser, run_grid, sha256_file, validate_episode_rows, validate_protocol


def valid_row(**changes):
    row = {
        "episode": 0, "eval_seed": 22, "time_in_range_pct": 75.0, "risk_index": 2.0,
        "reward": 5.0, "cost": 3.0, "length": 4, "horizon_steps": 4, "coverage_fraction": 1.0,
        "terminated": False, "truncated": True, "early_termination": False,
        "termination_cause": 0, "ghost_cost": 0.0, "hypo_events": 1, "hyper_events": 0,
        "time_below_range_pct": 25.0, "time_above_range_pct": 0.0, "severe_hypo_samples": 0,
        "time_in_range_frac": 0.75, "sd_mgdl": 1.0, "cv_pct": 1.0,
        "mag_mgdl_per_min": 1.0, "mage_mgdl": 1.0, "mean_glucose": 105.0,
        "meal_recommendations_per_day": 0.0, "bolus_recommendations_per_day": 0.0,
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("changes", [
    {"time_in_range_pct": float("nan")}, {"risk_index": float("inf")}, {"cost": ""},
    {"episode": 1}, {"length": 5}, {"coverage_fraction": 0.5},
    {"terminated": False, "truncated": False}, {"early_termination": True},
    {"time_in_range_pct": 90}, {"cost": -1}, {"truncated": "maybe"},
])
def test_result_gate_rejects_bad_evidence(changes):
    with pytest.raises(ValueError):
        validate_episode_rows([valid_row(**changes)], 1)


def test_result_gate_requires_all_episodes_and_metrics():
    row = valid_row()
    validate_episode_rows([row], 1)
    with pytest.raises(ValueError, match="Expected 2"):
        validate_episode_rows([row], 2)
    del row["cost"]
    with pytest.raises(ValueError, match="missing metrics"):
        validate_episode_rows([row], 1)


@pytest.mark.parametrize("changes", [
    {"patients": []}, {"patients": ["adult#002", "adult#002"]},
    {"patients": ["../../other"]}, {"patients": ["adult#011"]},
    {"episodes": 0}, {"horizon_days": 0.5}, {"horizon_days": float("nan")},
    {"horizon_days": 1.0001}, {"eval_seed_base": -1}, {"action_mode": "greedy"},
])
def test_protocol_validation(changes):
    values = dict(patient_type="t1d", patients=["adult#002"], episodes=1,
                  horizon_days=7, eval_seed_base=22, action_mode="stochastic")
    values.update(changes)
    with pytest.raises(ValueError):
        validate_protocol(**values)


def test_cli_preserves_legacy_sampling_defaults():
    args = build_parser().parse_args([
        "--checkpoint", "checkpoint.pt", "--config", "config.json", "--output-dir", "output",
        "--patient-type", "t1d", "--patients", "adult#002",
    ])
    assert (args.episodes, args.horizon_days, args.eval_seed_base, args.action_mode) == (10, 7, 22, "stochastic")
    assert args.save_traces is False


def test_vector_tensor_normalizer_matches_saved_transform():
    normalizer = evaluation.SimpleNormalizer((2,))
    normalizer.load_state_dict({
        "_mean": torch.tensor([0.0, 2.0]), "_var": torch.tensor([0.0, 4.0]),
        "_std": torch.tensor([0.01, 2.0]), "_clip": torch.tensor([2.0, 2.0]), "_count": torch.tensor(10),
    })
    np.testing.assert_allclose(normalizer.normalize(np.array([1.0, 4.0])), [2.0, 1.0])


def test_zero_count_and_public_normalizer_keys_do_not_fall_back():
    normalizer = evaluation.SimpleNormalizer((2,))
    normalizer.load_state_dict({"mean": torch.zeros(2), "var": torch.ones(2),
                                "count": torch.tensor(0), "_count": torch.tensor(20)})
    obs = np.array([4.0, 2.0])
    np.testing.assert_array_equal(normalizer.normalize(obs), obs)
    assert normalizer.count == 0


def test_deterministic_multicategorical_action_and_lazy_shield():
    actor = evaluation.SimpleCategoricalActor(3, gym.spaces.MultiDiscrete([2, 3]), hidden_sizes=())
    with torch.no_grad():
        actor.logits_net[0].weight.zero_()
        actor.logits_net[0].bias.copy_(torch.tensor([0.0, 1.0, 4.0, 2.0, 3.0]))
    torch.testing.assert_close(actor(torch.zeros(1, 3)).mode, torch.tensor([[1, 0]]))
    # Check lazy imports in a fresh process: other tests legitimately import
    # the experimental shield during collection.
    subprocess.run([
        sys.executable, "-c",
        "import sys\n"
        "import gymnasium as gym\n"
        "from glucoalg.evaluation import SimpleCategoricalActor\n"
        "SimpleCategoricalActor(3, gym.spaces.MultiDiscrete([2, 3]))\n"
        "assert 'shield.predictive_shield' not in sys.modules\n"
        "assert 'plot_utils' not in sys.modules\n",
    ], check=True, capture_output=True, text=True)
    with pytest.raises(ValueError, match="experimental glucoalg-dynamics rollout"):
        evaluation.SimpleCategoricalActor(3, gym.spaces.MultiDiscrete([2, 3]), shield_type="adult#001")


class TinyEnv:
    sample_time = 5
    simulation_minutes = 20
    action_space = gym.spaces.MultiDiscrete([2, 3])
    observation_space = gym.spaces.Box(-1000, 1000, shape=(3,))

    def __init__(self, early=False, omit_cgm=False):
        self.early = early
        self.omit_cgm = omit_cgm
        self.actions = []
        self.seeds = []
        self.closed = False

    def reset(self, seed):
        self.seeds.append(seed)
        self.steps = 0
        return torch.tensor([[100.0, 0.0, 0.0]]), {}

    def step(self, action):
        self.actions.append(np.array(action))
        self.steps += 1
        cgm = [100, 60, 140, 120][self.steps - 1]
        info = {} if self.omit_cgm else {"cgm": cgm}
        info["termination_cause"] = 1 if self.early else 0
        info["ghost_cost"] = 4 if self.early and self.steps == 2 else 0
        return torch.tensor([[cgm, 0.0, 0.0]]), 1.0, 2.0, self.early and self.steps == 2, self.steps == 4, info

    def close(self):
        self.closed = True


def simple_actor():
    actor = evaluation.SimpleCategoricalActor(3, TinyEnv.action_space, hidden_sizes=())
    with torch.no_grad():
        actor.logits_net[0].weight.zero_()
        actor.logits_net[0].bias.copy_(torch.tensor([0.0, 1.0, 4.0, 2.0, 3.0]))
    return actor


def fake_metrics(trace, sample_time):
    return {"time_in_range_pct": 100 * sum(70 <= value <= 180 for value in trace) / len(trace),
            "time_in_range_frac": sum(70 <= value <= 180 for value in trace) / len(trace),
            "risk_index": 2.0, "sd_mgdl": 1.0, "cv_pct": 1.0,
            "mag_mgdl_per_min": 1.0, "mage_mgdl": 1.0, "mean_glucose": np.mean(trace)}


def test_early_termination_cost_and_policy_seed_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "_glucose_metrics", fake_metrics)
    env = TinyEnv(early=True)
    result = evaluation.evaluate_model(env, simple_actor(), num_episodes=2, seed=22,
                                       output_dir=tmp_path, action_mode="deterministic", save_traces=False, cost_limit=3)
    assert env.seeds == [22, 23]
    assert all(np.array_equal(action, [[1, 0]]) for action in env.actions)
    for row in result["episode_details"]:
        assert row["coverage_fraction"] == 0.5
        assert row["terminated"] and row["early_termination"] and not row["truncated"]
        assert row["cost"] == 4 and row["ghost_cost"] == 4 and row["cost_limit_exceeded"]
    assert not list(tmp_path.glob("*controller*"))
    assert (tmp_path / "detailed_results.csv").is_file()


def test_explicit_legacy_vs_post_step_glucose_alignment(tmp_path, monkeypatch):
    traces = []

    def metrics(trace, sample_time):
        traces.append(trace)
        return fake_metrics(trace, sample_time)

    monkeypatch.setattr(evaluation, "_glucose_metrics", metrics)
    for source in ("legacy", "post-step"):
        evaluation.evaluate_model(TinyEnv(omit_cgm=True), simple_actor(), num_episodes=1,
                                  output_dir=tmp_path / source, save_traces=False, glucose_source=source)
    assert traces == [[100, 100, 60, 140], [100, 60, 140, 120]]


def test_checkpoint_width_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    actor = evaluation.SimpleCategoricalActor(4, TinyEnv.action_space, hidden_sizes=())
    torch.save({"pi": actor.state_dict()}, checkpoint)
    config = {"model_cfgs": {"actor_type": "categorical", "actor": {"hidden_sizes": [], "activation": "tanh"}}}
    with pytest.raises(ValueError, match="does not match simulator width"):
        evaluation.load_model(checkpoint, config, TinyEnv())


def setup_mock_grid(tmp_path, monkeypatch, fail=False):
    from glucoalg import runtime

    monkeypatch.setattr(runtime, "configure_runtime", lambda: None)
    monkeypatch.setattr(runtime, "initialize_simulator", lambda root: {"glucosim_file": "/pinned/simulator"})
    checkpoint, config = tmp_path / "policy.pt", tmp_path / "config.json"
    checkpoint.write_bytes(b"frozen checkpoint")
    config.write_text("{}")
    envs = []

    def factory(*args, **kwargs):
        env = TinyEnv()
        if fail:
            env.reset = lambda seed: (torch.tensor([[float("nan"), 0.0, 0.0]]), {})
        envs.append(env)
        return env

    monkeypatch.setattr(evaluation, "create_diabetes_env", factory)
    monkeypatch.setattr(evaluation, "load_model", lambda path, config, env, *args: (simple_actor(), env, None))
    monkeypatch.setattr(evaluation, "_glucose_metrics", fake_metrics)
    kwargs = dict(checkpoint=checkpoint, config=config, output_dir=tmp_path / "results",
                  patient_type="t1d", patients=["adult#002", "adult#003"], episodes=1)
    return kwargs, envs


def test_grid_manifest_records_hashes_safety_and_no_overwrite(tmp_path, monkeypatch):
    kwargs, envs = setup_mock_grid(tmp_path, monkeypatch)
    manifest = run_grid(**kwargs)
    assert manifest["status"] == "complete"
    assert manifest["checkpoint"]["sha256"] == sha256_file(kwargs["checkpoint"])
    assert manifest["overall"]["n_patients"] == 2
    assert manifest["overall"]["n_episodes"] == 2
    assert manifest["overall"]["mean_cost"] == 8
    assert manifest["action_mode"] == "stochastic"
    assert all(env.closed for env in envs)
    with pytest.raises(FileExistsError):
        run_grid(**kwargs)


def test_grid_failure_is_recorded_and_environment_closed(tmp_path, monkeypatch):
    kwargs, envs = setup_mock_grid(tmp_path, monkeypatch, fail=True)
    with pytest.raises(ValueError, match="Invalid observation"):
        run_grid(**kwargs)
    manifest = json.loads((kwargs["output_dir"] / "EVAL_MANIFEST.json").read_text())
    assert manifest["status"] == "failed"
    assert "Invalid observation" in manifest["error"]
    assert envs[0].closed
    assert "overall" not in manifest


def test_two_evaluators_cannot_claim_the_same_empty_output(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from glucoalg import eval_grid

    kwargs, _ = setup_mock_grid(tmp_path, monkeypatch)
    kwargs["output_dir"].mkdir()
    barrier = Barrier(2)
    original_hash = eval_grid.sha256_file

    def synchronize_after_empty_check(path):
        digest = original_hash(path)
        # Only pause the first hash call in each invocation, after both have
        # observed an empty output and before either can claim it.
        if path == kwargs["checkpoint"] and not (kwargs["output_dir"] / ".evaluation.lock").exists():
            barrier.wait(timeout=10)
        return digest

    monkeypatch.setattr(eval_grid, "sha256_file", synchronize_after_empty_check)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_grid, **kwargs) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError as exc:
                outcomes.append(exc)
    assert sum(isinstance(value, FileExistsError) for value in outcomes) == 1
    assert sum(isinstance(value, dict) and value["status"] == "complete" for value in outcomes) == 1


@pytest.mark.parametrize("changes, message", [
    ({"shield_type": "predictive"}, "glucoalg-dynamics rollout"),
    ({"shield_type": "adult#001"}, "glucoalg-dynamics rollout"),
    ({"logit_penalty": -10}, "nonnegative magnitude"),
    ({"logit_penalty": float("nan")}, "nonnegative magnitude"),
    ({"logit_penalty": float("inf")}, "nonnegative magnitude"),
])
def test_invalid_shield_options_fail_before_inputs_or_output(tmp_path, changes, message):
    output = tmp_path / "results"
    with pytest.raises(ValueError, match=message):
        run_grid(checkpoint=tmp_path / "missing.pt", config=tmp_path / "missing.json",
                 output_dir=output, patient_type="t1d", patients=["adult#002"], **changes)
    assert not output.exists()


@pytest.mark.parametrize("option", [["--shield-type", "predictive"], ["--logit-penalty", "-10"]])
def test_cli_rejects_unsupported_shield_settings(option):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([
            "--checkpoint", "missing.pt", "--config", "missing.json", "--output-dir", "output",
            "--patient-type", "t1d", "--patients", "adult#002", *option,
        ])
    assert error.value.code == 2


def test_legacy_shield_flag_fails_with_actionable_message(monkeypatch, capsys):
    from glucoalg import eval_grid

    monkeypatch.setattr(eval_grid, "run_legacy", lambda **kwargs: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as error:
        eval_grid.legacy_main(["t1d", "CPO", "adult#001", "100", "--shield"])
    assert error.value.code == 2
    assert "glucoalg-dynamics rollout" in capsys.readouterr().err


def test_legacy_rule_based_default_suppresses_risky_bolus(monkeypatch):
    from glucoalg import eval_grid
    from shield.rule_based_shield import RuleBasedShield

    def check_options(**kwargs):
        assert kwargs["logit_penalty"] == 10.0
        shield = RuleBasedShield(logit_penalty=kwargs["logit_penalty"])
        obs = torch.zeros(14)
        obs[0] = 80
        result = shield.apply(obs, torch.zeros(10), [5, 5])
        assert result[0] == 0
        assert torch.all(result[1:5] < 0)
        return {"overall": {}}

    monkeypatch.setattr(eval_grid, "run_legacy", check_options)
    eval_grid.legacy_main(["t1d", "CPO", "adult#001", "100", "--shield-type", "rule_based"])


@pytest.mark.parametrize("shield_type", ["adult", "predictive", "rule_based"])
def test_training_actor_rejects_unsupported_shield(shield_type):
    from omnisafe.models.actor.categorical_actor import CategoricalActor

    with pytest.raises(ValueError, match="Training-time shielding is not supported"):
        CategoricalActor(TinyEnv.observation_space, TinyEnv.action_space, [], shield_type=shield_type)


def test_training_actor_default_preserves_categorical_distribution():
    from omnisafe.models.actor.categorical_actor import CategoricalActor

    actor = CategoricalActor(TinyEnv.observation_space, TinyEnv.action_space, [])
    assert actor.shield is None
    assert actor(torch.zeros(2, 3)).sample().shape == (2, 2)


@pytest.mark.parametrize("changes", [{"shield_type": "predictive"}, {"logit_penalty": -10}])
def test_legacy_api_rejects_shield_options_before_checkpoint_lookup(changes):
    from glucoalg.eval_grid import run_legacy

    with pytest.raises(ValueError, match="shielding|magnitude"):
        run_legacy("t1d", "CPO", "adult#001", 100, saved_models="missing-models", **changes)
