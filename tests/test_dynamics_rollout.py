"""Paired sampling and actual recorded outcomes using a terminating fake env."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glucoalg.dynamics.rollout import coupled_proposals, paired_proposals, rollout_episode
from shield.predictive_shield import PredictiveShieldConfig, Shield
from shield.predictor import PointForecast, PredictorMetadata, PatientIdentity
from test_dynamics_collect import FakeEnv


class Actor:
    shield = None

    def logits_net(self, normalized):
        self.normalized = normalized
        return torch.tensor([[0., 0., 20., 0., 0., 0., 0., 0., 0., 20.]])


def run(shield=None, probability=0):
    return rollout_episode(FakeEnv(mutate=True), Actor(), None, shield=shield, seed=1000, action_seed=1000,
                           exploration_seed=11000, horizon_steps=288, action_mode="deterministic",
                           exploration_probability=probability)


def test_pairing_preserves_base_draw_and_advances_rng_once():
    logits = torch.zeros((1, 10))
    torch.manual_seed(50)
    expected = torch.stack([torch.distributions.Categorical(logits=chunk).sample()
                            for chunk in logits.split((5, 5), dim=-1)], dim=-1).numpy().reshape(2)
    expected_state = torch.random.get_rng_state().clone()
    torch.manual_seed(50)
    base, adjusted = paired_proposals(logits, logits.clone(), "stochastic")
    np.testing.assert_array_equal(base, expected)
    np.testing.assert_array_equal(adjusted, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_counterfactual_proposal_uses_same_draw_without_changing_next_stream():
    logits = torch.zeros((1, 10))
    shifted = logits.clone()
    shifted[:, [2, 9]] = 100
    torch.manual_seed(51)
    _, changed = paired_proposals(logits, shifted, "stochastic")
    np.testing.assert_array_equal(changed, [2, 4])
    after = torch.random.get_rng_state().clone()
    torch.manual_seed(51)
    paired_proposals(logits, logits, "stochastic")
    assert torch.equal(torch.random.get_rng_state(), after)


def test_three_way_attribution_keeps_original_pair_and_rng():
    base = torch.zeros((1, 10))
    static = base.clone()
    static[:, [0, 6]] = 100
    full = base.clone()
    full[:, [3, 9]] = 100
    torch.manual_seed(83)
    expected = paired_proposals(base, full, "stochastic")
    after = torch.random.get_rng_state().clone()
    torch.manual_seed(83)
    actual = coupled_proposals("stochastic", base, static, full)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], [0, 1])
    np.testing.assert_array_equal(actual[2], expected[1])
    assert torch.equal(torch.random.get_rng_state(), after)


@pytest.mark.parametrize("probability", [0, 1])
def test_prediction_attribution_separates_static_and_exploration(probability):
    class SensitiveActor(Actor):
        def logits_net(self, normalized):
            return torch.tensor([[0., 0., 5., 0., 0., 0., 0., 0., 0., 5.]])

    predictor = SimpleNamespace(metadata=PredictorMetadata(1, 2, "fixture"), reset=lambda: None)
    predictor.forecast = lambda request: PointForecast(torch.full((len(request.candidate_actions), 2), 65.))
    shield = Shield(predictor=predictor, patient=PatientIdentity("t1d", "adolescent#001"))
    decisions = []
    arrays, metrics, _ = rollout_episode(
        FakeEnv(), SensitiveActor(), None, shield=shield, seed=1000, action_seed=1000,
        exploration_seed=11000, horizon_steps=288, action_mode="deterministic",
        exploration_probability=probability, decision_records=decisions)
    assert metrics["static_proposal_changed_steps"] == 0
    assert metrics["prediction_proposal_changed_steps"] == 2
    assert metrics["prediction_recommendation_changed_steps"] == (2 if probability == 0 else 0)
    assert len(decisions) == metrics["length"] == 3
    np.testing.assert_array_equal(arrays["base_proposals"], arrays["static_proposals"])
    assert not np.any(arrays["prediction_proposal_changed"][:1])  # warmup is not a prediction.


def test_disabled_forecast_ablation_keeps_history_without_prediction_calls():
    def forbidden(request):
        raise AssertionError("static ablation called predictor")
    predictor = SimpleNamespace(metadata=PredictorMetadata(1, 2, "fixture"), reset=lambda: None,
                                forecast=forbidden)
    shield = Shield(predictor=predictor, patient=PatientIdentity("t1d", "adolescent#001"),
                    config=PredictiveShieldConfig(use_forecast=False))
    arrays, metrics, _ = run(shield)
    assert metrics["forecast_available_steps"] == 0
    assert metrics["prediction_logit_changed_steps"] == 0
    assert metrics["prediction_recommendation_changed_steps"] == 0
    np.testing.assert_array_equal(arrays["static_actions"], arrays["recommended_actions"])


def test_rollout_rejects_training_branch_reset_family(tmp_path):
    from glucoalg.dynamics.rollout import audit_rollout
    from glucoalg.dynamics.model import sha256_file
    from test_dynamics_validation import artifact

    manifest = artifact()
    patient = PatientIdentity("t1d", "adolescent#001")
    manifest["patient_scope"] = {"supported_patients": [vars(patient)]}
    manifest["provenance"]["train_branches"] = [dict(
        group_id="train-anchor-27", patient_type="t1d", patient_name=patient.patient_name,
        seed=2000, prefix_data_sha256="prefix-hash")]
    checkpoint, config = tmp_path / "checkpoint", tmp_path / "config"
    checkpoint.write_text("policy")
    config.write_text("config")
    manifest["provenance"].update(policy_checkpoint_sha256=sha256_file(checkpoint),
                                   policy_config_sha256=sha256_file(config))
    runtime = {"glucosim_commit": "sim", "glucosim_content_sha256": "content"}
    with pytest.raises(ValueError, match="overlaps"):
        audit_rollout(manifest, patient, 2000, checkpoint, config, runtime, [])
    assert audit_rollout(manifest, patient, 2400, checkpoint, config, runtime, [])["fit_selection_branch_groups_checked"] == 1


@pytest.mark.parametrize('changed', ['checkpoint', 'config'])
def test_predictive_policy_hash_binding_remains_strict(tmp_path, changed):
    from glucoalg.dynamics.rollout import audit_rollout
    from glucoalg.dynamics.model import sha256_file
    from test_dynamics_validation import artifact

    manifest = artifact()
    patient = PatientIdentity('t1d', 'adolescent#001')
    manifest['patient_scope'] = {'supported_patients': [vars(patient)]}
    files = {name: tmp_path / name for name in ('checkpoint', 'config')}
    for name, path in files.items():
        path.write_text('original ' + name)
        manifest['provenance']['policy_' + name + '_sha256'] = sha256_file(path)
    files[changed].write_text('different CPO ' + changed)
    with pytest.raises(ValueError, match='rollout policy ' + changed + ' mismatch'):
        audit_rollout(manifest, patient, 2400, files['checkpoint'], files['config'],
                      {'glucosim_commit': 'sim', 'glucosim_content_sha256': 'content'}, [])


def test_rollout_records_actual_acceptance_and_early_termination():
    arrays, metrics, reasons = run()
    np.testing.assert_array_equal(arrays["recommended_actions"], [[2, 4]] * 3)
    np.testing.assert_array_equal(arrays["executed_actions"], [[0, 4]] * 3)
    np.testing.assert_array_equal(arrays["observations"][:, 0], [100, 101, 102, 103])
    assert metrics["cost"] == 6
    assert metrics["coverage_fraction"] == 3 / 288
    assert metrics["terminated"]
    assert metrics["recommendation_changed_steps"] == 0
    assert metrics["active_recommendation_rejected_steps"] == 3
    assert reasons == {}


def test_predictive_history_receives_executed_and_recommended_separately():
    requests = []
    predictor = SimpleNamespace(metadata=PredictorMetadata(1, 2, "fixture"), reset=lambda: None)
    predictor.forecast = lambda request: (requests.append(request) or PointForecast(torch.full((len(request.candidate_actions), 2), 100.)))
    shield = Shield(predictor=predictor, patient=PatientIdentity("t1d", "adolescent#001"))
    _, metrics, reasons = run(shield)
    assert reasons == {"warmup": 1}
    assert metrics["forecast_available_steps"] == 2
    assert requests[0].past_transitions[0].executed_action == (0, 4)
    assert requests[0].past_transitions[0].recommended_action == (2, 4)
    assert requests[0].current_observation[0] == 101
    # Every episode clears predictor/shield history.
    _, _, repeated_reasons = run(shield)
    assert repeated_reasons == reasons


def test_post_intervention_exploration_can_bypass_the_soft_rule():
    class Change:
        def reset(self):
            pass

        def apply(self, raw, logits, dims):
            result = torch.zeros_like(logits)
            result[:, [0, 5]] = 100
            return result

    arrays, metrics, _ = run(Change(), probability=1)
    assert metrics["logit_intervention_steps"] == 3
    assert metrics["proposal_changed_steps"] == 3
    assert metrics["recommendation_changed_steps"] == 0
    assert arrays["explored"].all()


@pytest.mark.parametrize("value", [float("nan"), -1, 2])
def test_invalid_exploration_fails(value):
    with pytest.raises(ValueError):
        run(probability=value)


@pytest.mark.parametrize('condition,scale', [('none', 1.), ('predictive_static', 1.),
                                           ('predictive', 0.), ('predictive', .3), ('predictive', 1.)])
def test_rollout_orchestration_checks_explicit_test_exclusions_and_publishes_trace(tmp_path, monkeypatch,
                                                                                 condition, scale):
    import json
    import glucoalg.evaluation as evaluation
    import glucoalg.tuning.plan as plan
    from glucoalg.dynamics import rollout
    from glucoalg.dynamics.model import sha256_file
    from test_dynamics_validation import artifact, episode

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    for name in ("artifact.json", "artifact.sha256", "weights.pt"):
        (artifact_dir / name).write_text("fixture")
    checkpoint, config = tmp_path / "checkpoint", tmp_path / "config"
    checkpoint.write_text("fixture-policy")
    config.write_text("{}")
    manifest = artifact()
    manifest.update(patient_scope={"supported_patients": [vars(PatientIdentity("t1d", "adolescent#001"))]}, training={"seed": 1101})
    manifest["provenance"].update(policy_checkpoint_sha256=sha256_file(checkpoint), policy_config_sha256=sha256_file(config))
    predictor = SimpleNamespace(artifact=manifest, metadata=PredictorMetadata(1, 2, "fixed"),
                                reset=lambda: None,
                                forecast=lambda request: PointForecast(torch.full((len(request.candidate_actions), 2), 65.)))
    monkeypatch.setattr(rollout, "load_predictor", lambda *a, **k: predictor)
    monkeypatch.setattr(rollout, "initialize_simulator", lambda *a: {"glucosim_file": str(tmp_path / "glucosim" / "__init__.py"), "glucosim_commit": "sim"})
    monkeypatch.setattr(plan, "_content_hash", lambda *a: "content")
    monkeypatch.setattr(plan, "probe_sources", lambda *a: {"repo": {"content_sha256": "repo"}, "simulator": {"content_sha256": "content"}})
    monkeypatch.setattr(evaluation, "create_diabetes_env", lambda *a, **k: FakeEnv())
    monkeypatch.setattr(evaluation, "load_model", lambda checkpoint, config, env: (Actor(), env, None))
    monkeypatch.setattr(evaluation, "_glucose_metrics", lambda cgm, dt: {"risk_index": 0.0})
    def excluded(paths, *, expected_split):
        assert expected_split == "test" and paths == ["test-episode"]
        return [episode()]
    monkeypatch.setattr(rollout, "load_episodes", excluded)
    output = tmp_path / "output"
    report = rollout.run_rollout(artifact_dir=artifact_dir, checkpoint=checkpoint, config=config, simulator_root=tmp_path,
                                output_dir=output, patient_type="t1d", patient_name="adolescent#001", condition=condition,
                                env_seed=1000, action_seed=1000, exploration_seed=11000, exclude_episodes=["test-episode"],
                                forecast_penalty_scale=scale)
    assert report["behavior_matches_collection"]
    assert report["forecast_penalty_scale"] == scale
    assert report["intervention_changes_continuation"] == (condition != 'none')
    if condition == 'none':
        assert report["shield_settings"] == {}
    else:
        assert report["shield_settings"]["config"]["forecast_penalty_scale"] == scale
        assert report["shield_settings"]["config"]["logit_penalty"] == 10.
        assert report["shield_settings"]["config"]["use_forecast"] == (condition == 'predictive')
    assert report["metrics"]["length"] == 3
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    assert json.loads((output / "split_check.json").read_text())["additional_episodes_checked"] == 1
    assert (output / "trace.npz").is_file()


@pytest.mark.parametrize('mode', ['stochastic', 'deterministic'])
@pytest.mark.parametrize('probability', [0., .1, 1.])
def test_shadow_zero_and_disabled_static_have_equal_episode_arrays(mode, probability):
    from glucoalg.dynamics.data import ARRAY_NAMES

    requests = []
    def predictor():
        return SimpleNamespace(metadata=PredictorMetadata(1, 2, "fixture"), reset=lambda: None,
                               forecast=lambda request: (requests.append(request) or PointForecast(
                                   torch.full((len(request.candidate_actions), 2), 65.))))
    def episode(config):
        shield = Shield(predictor=predictor(), patient=PatientIdentity('t1d', 'adolescent#001'), config=config)
        return rollout_episode(FakeEnv(), Actor(), None, shield=shield, seed=1000, action_seed=1000,
                               exploration_seed=11000, horizon_steps=288, action_mode=mode,
                               exploration_probability=probability)
    shadow, shadow_metrics, _ = episode(PredictiveShieldConfig(forecast_penalty_scale=0.))
    assert len(requests) == 2
    static, static_metrics, _ = episode(PredictiveShieldConfig(use_forecast=False))
    assert len(requests) == 2
    for name in ARRAY_NAMES + ('base_proposals', 'static_proposals', 'adjusted_proposals',
                              'explored', 'prediction_recommendation_changed'):
        np.testing.assert_array_equal(shadow[name], static[name])
    assert shadow_metrics['prediction_logit_changed_steps'] == static_metrics['prediction_logit_changed_steps'] == 0
    assert shadow_metrics['forecast_available_steps'] == 2 and static_metrics['forecast_available_steps'] == 0


@pytest.mark.parametrize('condition', ['none', 'rule_based', 'predictive_static'])
@pytest.mark.parametrize('scale', [0., .1, 3.])
def test_nondefault_scale_rejected_on_conditions_that_would_ignore_it(tmp_path, condition, scale):
    from glucoalg.dynamics.rollout import run_rollout
    with pytest.raises(ValueError, match='nondefault forecast_penalty_scale requires'):
        run_rollout(artifact_dir=tmp_path / 'missing', checkpoint='missing', config='missing',
                    simulator_root='missing', output_dir=tmp_path / 'output', patient_type='t1d',
                    patient_name='adolescent#001', condition=condition, env_seed=1000,
                    action_seed=1000, exploration_seed=11000, forecast_penalty_scale=scale)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('scale', [True, '1', None, -1, float('nan'), float('inf')])
def test_invalid_rollout_scale_rejected_before_artifact_reads(tmp_path, scale):
    from glucoalg.dynamics.rollout import run_rollout
    with pytest.raises(ValueError, match='forecast_penalty_scale must be finite and nonnegative'):
        run_rollout(artifact_dir=tmp_path / 'missing', checkpoint='missing', config='missing',
                    simulator_root='missing', output_dir=tmp_path / 'output', patient_type='t1d',
                    patient_name='adolescent#001', condition='predictive', env_seed=1000,
                    action_seed=1000, exploration_seed=11000, forecast_penalty_scale=scale)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('flags,expected', [([], 1.), (['--forecast-penalty-scale', '0'], 0.),
                                         (['--forecast-penalty-scale', '.3'], .3)])
def test_rollout_cli_forwards_forecast_scale(monkeypatch, flags, expected):
    from glucoalg.dynamics import rollout
    calls = []
    monkeypatch.setattr(rollout, 'run_rollout', lambda **kwargs: (calls.append(kwargs) or {'metrics': {}}))
    required = []
    for flag in ('artifact', 'checkpoint', 'config', 'simulator-root', 'output-dir', 'patient-name'):
        required += ['--' + flag, 'fixture']
    required += ['--condition', 'predictive', '--env-seed', '1', '--action-seed', '2', '--exploration-seed', '3']
    assert rollout.main(required + flags) == 0
    assert calls[0]['forecast_penalty_scale'] == expected
