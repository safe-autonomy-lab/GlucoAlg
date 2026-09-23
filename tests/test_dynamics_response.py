"""Independent arithmetic and negative controls for paired response scoring."""
import numpy as np
import pytest

from glucoalg.dynamics.response import anchor_mean, summarize_responses


def fixture():
    actions = np.array([(b, m) for b in range(5) for m in range(5)])
    targets = np.broadcast_to((actions[:, 1] - actions[:, 0])[None, :, None] * np.array([10., 20.]), (2, 25, 2)).copy()
    accepted = np.broadcast_to(actions > 0, (2, 25, 2)).copy()
    return targets, np.ones_like(targets, dtype=bool), actions, accepted


def test_perfect_responses_and_action_ablated_negative_control():
    targets, valid, actions, accepted = fixture()
    perfect = summarize_responses(targets + 100, targets, valid, actions, accepted)
    for name in ("accepted_pure_bolus", "accepted_pure_meal"):
        row = perfect["strata"][name]
        assert row["anchors"] == 2
        assert row["candidate_pairs"] == 8
        assert row["response_mae"] == 0
        assert row["zero_response_mae"] == 37.5
        assert row["sign_agreement_over_5_mg_dl"] == 1
        assert row["absolute_mae"] == 100  # A response fit is not an absolute-forecast fit.
    ablated = summarize_responses(np.zeros_like(targets), targets, valid, actions, accepted)
    assert ablated["strata"]["accepted_pure_meal"]["improvement_over_zero_pct"] == 0


def test_partial_horizons_keep_equal_candidate_and_anchor_weight():
    values = np.array([[[2., 200.], [6., 10.]], [[20., 40.], [30., 50.]]])
    mask = np.array([[[True, False], [True, True]], [[True, True], [False, False]]])
    # Anchor0 = mean(2, mean(6,10)) =5; anchor1 =mean(20,40)=30.
    assert anchor_mean(values, mask) == 17.5
    targets, valid, actions, accepted = fixture()
    valid[0, 0, 1] = False  # no-op control ended early, restricting every paired response.
    targets[~valid] = np.nan
    result = summarize_responses(np.zeros_like(targets), targets, valid, actions, accepted)
    assert result["strata"]["all_nonzero"]["observed_horizon_points"] == 24 * 3
    assert result["observed_coverage_fraction"] == .99


def test_candidate_permutation_is_not_row_position_dependence():
    targets, valid, actions, accepted = fixture()
    order = np.arange(25)[::-1]
    before = summarize_responses(targets, targets, valid, actions, accepted)
    after = summarize_responses(targets[:, order], targets[:, order], valid[:, order], actions[order], accepted[:, order])
    assert before == after


def test_empty_accepted_stratum_is_coverage_failure_not_zero_error():
    targets, valid, actions, accepted = fixture()
    accepted[:] = False
    row = summarize_responses(targets, targets, valid, actions, accepted)["strata"]["accepted_pure_meal"]
    assert row == {"anchors": 0, "candidate_pairs": 0, "observed_horizon_points": 0}


@pytest.mark.parametrize("corrupt", ["grid", "gap", "infinite", "accept_noop"])
def test_invalid_response_evidence_fails(corrupt):
    targets, valid, actions, accepted = fixture()
    predicted = targets.copy()
    if corrupt == "grid": actions[-1] = actions[0]
    if corrupt == "gap": valid[0, 0, 0] = False
    if corrupt == "infinite": predicted[0, 0, 0] = np.inf
    if corrupt == "accept_noop": accepted[0, 0, 0] = True
    with pytest.raises(ValueError):
        summarize_responses(predicted, targets, valid, actions, accepted)


def evaluation_fixture(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import torch
    from shield.predictor import PointForecast, PredictorMetadata
    from glucoalg.dynamics import branches, response
    from test_dynamics_branches import mock_collection_runtime
    from test_dynamics_validation import artifact
    _, arguments = mock_collection_runtime(monkeypatch, tmp_path, terminal=35)
    arguments.update(split="test", env_seed=2200, action_seed=2200, exploration_seed=12200)
    record = branches.run_collection(**arguments)
    manifest = artifact()
    meta = record["metadata"]
    manifest["continuation_policy"] = meta["continuation_policy"]
    for key in ("checkpoint", "config"):
        manifest["provenance"]["policy_" + key + "_sha256"] = meta["policy"][key + "_sha256"]
    manifest["provenance"].update(simulator_commit="fixture", simulator_content_sha256="c" * 64)
    requests = []
    def forecast(request):
        requests.append(request)
        change = torch.tensor([meal - bolus for bolus, meal in request.candidate_actions], dtype=torch.float64)
        return PointForecast(request.current_observation[0] + change[:, None] * torch.arange(1, 13)[None])
    adapter = SimpleNamespace(artifact=manifest, config=SimpleNamespace(horizon_steps=12, history_length=12),
        metadata=PredictorMetadata(27, 12, meta["continuation_policy"]), reset=lambda: None, forecast=forecast)
    monkeypatch.setattr(response, "load_predictor", lambda *args, **kwargs: adapter)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("artifact.json", "artifact.sha256", "weights.pt"):
        (model / name).write_text("fixture")
    return response, model, arguments["output_dir"], adapter, requests


def test_saved_corpus_response_evaluation_is_causal_and_retains_missing_coverage(monkeypatch, tmp_path):
    import json
    response, model, collection, _, requests = evaluation_fixture(monkeypatch, tmp_path)
    output = tmp_path / "score"
    report = response.evaluate_responses(model, [collection], output)
    assert report["summary"]["observed_horizon_points"] == 200
    assert report["summary"]["requested_horizon_points"] == 600
    assert report["summary"]["observed_coverage_fraction"] == 1 / 3
    assert report["diagnostics"]["candidate_ablation"]["observed_coverage_fraction"] == 1 / 3
    assert report["summary"]["anchors_requested"] == 2 and len(report["groups"]) == 1
    assert len(requests) == 3
    assert requests[0].current_observation == requests[1].current_observation == requests[2].current_observation
    assert all(request.past_transitions == requests[0].past_transitions for request in requests)
    assert all(transition.recommended_action == (0, 0) for transition in requests[0].past_transitions)
    assert report["diagnostics"]["candidate_order_max_absolute_error"] == 0
    ablated = report["diagnostics"]["candidate_ablation"]["strata"]["all_nonzero"]
    assert ablated["response_mae"] == ablated["zero_response_mae"]
    with np.load(output / "responses.npz", allow_pickle=False) as data:
        assert np.isnan(data["targets"][:, :, 8:]).all()
        assert not data["valid"][:, :, 8:].any()
        assert data["label_permutations"][0, 0] == 0
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        response.evaluate_responses(model, [collection], output)


@pytest.mark.parametrize("failure", ["order", "branch_family", "input_change"])
def test_response_evaluator_rejects_invalid_forecast_or_evidence(monkeypatch, tmp_path, failure):
    import json
    from shield.predictor import PointForecast
    import torch
    response, model, collection, adapter, _ = evaluation_fixture(monkeypatch, tmp_path)
    original = adapter.forecast
    if failure == "order":
        adapter.forecast = lambda request: PointForecast(torch.arange(25.)[:, None].repeat(1, 12))
    elif failure == "branch_family":
        adapter.artifact["provenance"]["train_branches"] = [dict(group_id="other", patient_type="t1d",
            patient_name="adolescent#001", seed=2200, prefix_data_sha256="other")]
    else:
        def changed(request):
            (collection / "split_check.json").write_text("changed")
            return original(request)
        adapter.forecast = changed
    output = tmp_path / "score"
    with pytest.raises(ValueError, match="candidate-order|overlaps|inputs changed"):
        response.evaluate_responses(model, [collection], output)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_response_requires_complete_collection_not_selected_group(monkeypatch, tmp_path):
    response, model, collection, _, _ = evaluation_fixture(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        response.evaluate_responses(model, [collection / "anchor-000027"], tmp_path / "score")


def test_patient_with_no_reachable_anchor_is_reported_with_zero_coverage(monkeypatch, tmp_path):
    from test_dynamics_branches import mock_collection_runtime
    from glucoalg.dynamics import branches
    response, model, collection, _, _ = evaluation_fixture(monkeypatch, tmp_path)
    other = tmp_path / "empty-patient"
    other.mkdir()
    _, arguments = mock_collection_runtime(monkeypatch, other, terminal=10)
    arguments.update(split="test", patient_name="adolescent#004", env_seed=2201,
                     action_seed=2201, exploration_seed=12201)
    branches.run_collection(**arguments)
    report = response.evaluate_responses(model, [collection, arguments["output_dir"]], tmp_path / "score")
    row = report["by_patient"]["t1d/adolescent#004"]
    assert row["requested_horizon_points"] == 600 and row["observed_coverage_fraction"] == 0
    assert "unavailable" in row["reason"] and "absolute_mae" not in row
    assert report["summary"]["observed_coverage_fraction"] == 1 / 6
