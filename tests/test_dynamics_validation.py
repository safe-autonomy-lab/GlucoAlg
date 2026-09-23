"""Held-out metrics and real adapter boundary checks; no simulator execution."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glucoalg.dynamics import validate
from shield.predictor import PointForecast, PredictorMetadata


def episode(n=9):
    obs = np.zeros((n + 1, 14))
    obs[:, 0] = 100 + np.arange(n + 1) * 2
    actions = np.zeros((n, 2), dtype=int)
    actions[4] = [2, 0]
    meta = dict(episode_id="test", data_sha256="test-hash", patient_type="t1d", patient_name="adolescent#001",
                seed=900, split="test", policy={"checkpoint_sha256": "cp", "config_sha256": "cfg"},
                behavior={"action_mode": "stochastic", "exploration_probability": .1, "exploration_rng": "numpy.PCG64"},
                continuation_policy="fixed", runtime={"glucosim_commit": "sim", "glucosim_content_sha256": "content"}, metrics={})
    return SimpleNamespace(observations=obs, recommended_actions=actions, executed_actions=np.zeros_like(actions),
                           accepted=np.zeros_like(actions, dtype=bool), metadata=meta)


def artifact():
    known = {"episode_id": "fit", "data_sha256": "fit-hash", "patient_type": "t1d", "patient_name": "adolescent#001", "seed": 700}
    return {"continuation_policy": "fixed", "weights_sha256": "weights", "provenance": {
        "train_episodes": [known], "validation_episodes": [dict(known, episode_id="select", seed=800, data_sha256="select-hash")],
        "policy_checkpoint_sha256": "cp", "policy_config_sha256": "cfg", "simulator_commit": "sim",
        "simulator_content_sha256": "content", "behavior": episode().metadata["behavior"]}}


@pytest.mark.parametrize("field,value", [("episode_id", "fit"), ("data_sha256", "select-hash"), ("seed", 800)])
def test_holdout_rejects_relabelled_fit_or_selection(field, value):
    sample = episode()
    sample.metadata[field] = value
    with pytest.raises(ValueError, match="overlaps"):
        validate.audit_holdout(artifact(), [sample])


@pytest.mark.parametrize("field", ["checkpoint_sha256", "config_sha256"])
def test_holdout_rejects_different_policy(field):
    sample = episode()
    sample.metadata["policy"][field] = "different"
    with pytest.raises(ValueError, match="policy"):
        validate.audit_holdout(artifact(), [sample])


def test_holdout_matches_structured_epsilon_and_both_provenance_lists():
    sample, manifest = episode(), artifact()
    sample.metadata["behavior"]["exploration_probability"] += 1e-8
    with pytest.raises(ValueError, match="behavior"):
        validate.audit_holdout(manifest, [sample])
    manifest["provenance"]["validation_episodes"] = []
    with pytest.raises(ValueError, match="both"):
        validate.audit_holdout(manifest, [episode()])


@pytest.mark.parametrize("partition", ["train_branches", "validation_branches"])
@pytest.mark.parametrize("identity", ["reset", "prefix_hash", "group_id"])
def test_factual_holdout_rejects_branch_family_reuse(partition, identity):
    sample, manifest = episode(), artifact()
    branch = dict(group_id="branch-anchor27", patient_type="t1d", patient_name="adolescent#001",
                  seed=2100, prefix_data_sha256="prefix-hash")
    if identity == "reset":
        branch["seed"] = sample.metadata["seed"]
    elif identity == "prefix_hash":
        branch["prefix_data_sha256"] = sample.metadata["data_sha256"]
    else:
        branch["group_id"] = sample.metadata["episode_id"]
    manifest["provenance"][partition] = [branch]
    with pytest.raises(ValueError, match="overlaps"):
        validate.audit_holdout(manifest, [sample])


@pytest.mark.parametrize("entry", [None, {}, {"group_id": "g"}])
def test_incomplete_branch_provenance_is_not_ignored(entry):
    manifest = artifact()
    manifest["provenance"]["train_branches"] = [entry]
    with pytest.raises(ValueError, match="provenance"):
        validate.audit_holdout(manifest, [episode()])


def test_request_is_past_only_and_detached_from_mutable_future():
    sample = episode()
    before = validate.request_at(sample, 5, 3)
    sample.observations[6:] = 999
    sample.recommended_actions[6:] = 4
    assert before == validate.request_at(sample, 5, 3)
    assert len(before.past_transitions) == 3
    assert before.past_transitions[0].pre_observation[0] == 104
    assert before.past_transitions[-1].next_observation == before.current_observation
    sample.observations[0] = -100
    assert before == validate.request_at(sample, 5, 3)


def test_fixed_baselines_and_errors_have_exact_horizon_units():
    persistence, trend = validate.fixed_baselines([100, 102, 104], 3)
    np.testing.assert_array_equal(persistence, [104, 104, 104])
    np.testing.assert_array_equal(trend, [106, 108, 110])
    assert validate.error_metrics([[1, 3], [3, 1]], [[2, 1], [2, 3]]) == {
        "queries": 2, "mae": [1., 2.], "rmse": [1., 2.]}


def test_forecast_scores_exact_last_full_window_and_separates_noops(tmp_path, monkeypatch):
    sample = episode()
    manifest = artifact()
    predictor = SimpleNamespace(artifact=manifest, config=SimpleNamespace(horizon_steps=2, history_length=3),
                                metadata=PredictorMetadata(3, 2, "fixed"), reset=lambda: None,
                                forecast=lambda req: PointForecast(torch.tensor([[req.current_observation[0] + 2,
                                                                                 req.current_observation[0] + 4]])))
    monkeypatch.setattr(validate, "load_predictor", lambda *a, **k: predictor)
    monkeypatch.setattr(validate, "load_episodes", lambda *a, **k: [sample])
    model = tmp_path / "model"
    model.mkdir()
    (model / "artifact.json").write_text("fixture")
    (model / "artifact.sha256").write_text("fixture")
    (model / "weights.pt").write_text("fixture")
    output = tmp_path / "score"
    report = validate.evaluate_forecasts(model, ["fixture"], output)
    assert report["summary"]["model"] == {"queries": 5, "mae": [0., 0.], "rmse": [0., 0.]}
    with np.load(output / "forecasts.npz") as arrays:
        np.testing.assert_array_equal(arrays["anchor_index"], [3, 4, 5, 6, 7])
        assert arrays["targets"][-1, -1] == sample.observations[-1, 0]
    groups = report["by_region_and_acceptance"]
    assert groups["active_recommendation_rejected"]["queries"] == 1
    assert groups["no_active_recommendation"]["queries"] == 4
    assert groups["both_acceptance_flags_true"]["queries"] == 0
    assert groups["cgm_below_70"] == {"queries": 0}
    with pytest.raises(FileExistsError):
        validate.evaluate_forecasts(model, ["fixture"], output)
    # Keep an early-terminated episode in the coverage report even when no
    # complete forecast target exists for it; never invent zero error for it.
    short = deepcopy(sample)
    short.observations = short.observations[:3]
    short.recommended_actions = short.recommended_actions[:2]
    short.executed_actions = short.executed_actions[:2]
    short.accepted = short.accepted[:2]
    short.metadata.update(episode_id="short", data_sha256="short-hash", seed=901)
    monkeypatch.setattr(validate, "load_episodes", lambda *a, **k: [sample, short])
    repeated = validate.evaluate_forecasts(model, ["fixture"], tmp_path / "with-short")
    assert repeated["episodes"][1]["model"]["queries"] == 0
    assert "mae" not in repeated["episodes"][1]["model"]
    assert repeated["summary"]["model"]["queries"] == 5
    # Bind the report to what was actually loaded, even with an external edit.
    original_forecast = predictor.forecast
    def changing_forecast(request):
        (model / "artifact.json").write_text("changed")
        return original_forecast(request)
    predictor.forecast = changing_forecast
    with pytest.raises(ValueError, match="changed during"):
        validate.evaluate_forecasts(model, ["fixture"], tmp_path / "drift")


@pytest.mark.parametrize("predicted,target", [([], []), ([[float("nan")]], [[1]]), ([[1]], [[1, 2]])])
def test_invalid_forecast_metrics_fail(predicted, target):
    with pytest.raises(ValueError):
        validate.error_metrics(predicted, target)
