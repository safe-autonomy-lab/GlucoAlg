"""Fake-predictor boundary tests; no evidence of trained forecast or safety quality."""

from dataclasses import FrozenInstanceError
import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from shield.predictive_shield import PredictiveShieldConfig, Shield, ShieldParams
from shield.predictor import (
    ForecastUnavailable,
    GLUCOSIM_OBSERVATION_FEATURES,
    PatientIdentity,
    PointForecast,
    PredictorMetadata,
)


class FakePredictor:
    def __init__(self, history_length=2, horizon_steps=3, result=None, **metadata):
        self.metadata = PredictorMetadata(history_length, horizon_steps, 'zero future recommendations', **metadata)
        self.requests = []
        self.reset_calls = 0
        self.episode_state = []
        self.result = result

    def reset(self):
        self.reset_calls += 1
        self.episode_state.clear()

    def forecast(self, request):
        self.requests.append(request)
        self.episode_state.append(request.current_observation)
        if callable(self.result):
            return self.result(request)
        if self.result is not None:
            return self.result
        return PointForecast(torch.full(
            (len(request.candidate_actions), self.metadata.horizon_steps), 120.0,
            device=request.device,
        ))


def observation(cgm=120.0):
    return torch.tensor([cgm, 2, 3, 4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4])


def shield_for(predictor=None, **kwargs):
    return Shield(predictor=predictor or FakePredictor(),
                  patient=PatientIdentity('t1d', 'adolescent#003'), **kwargs)


def advance(shield, start=120.0, count=2):
    for i in range(count):
        shield.apply(observation(start + i), torch.zeros(10), [5, 5])
        shield.record_action((i % 5, 0), next_observation=observation(start + i + 1))


def test_raw_transition_timing_distinguishes_execution_and_recommendation():
    predictor = FakePredictor(history_length=1)
    shield = shield_for(predictor)
    pre, outcome = observation(120), observation(121)
    shield.apply(pre, torch.arange(10.0), [5, 5])
    assert not predictor.requests
    assert isinstance(shield.last_forecast, ForecastUnavailable)
    assert '0/1' in shield.last_forecast.reason
    shield.record_action((0, 2), next_observation=outcome,
                         recommended_action=(4, 2), accepted=(False, True))
    shield.apply(outcome, torch.arange(10.0), [5, 5])
    request = predictor.requests[-1]
    transition, = request.past_transitions
    assert transition.pre_observation == tuple(pre.tolist())
    assert transition.next_observation == tuple(outcome.tolist())
    assert transition.executed_action == (0, 2)
    assert transition.recommended_action == (4, 2)
    assert transition.accepted == (False, True)
    assert request.current_observation == tuple(outcome.tolist())
    assert request.patient == PatientIdentity('t1d', 'adolescent#003')
    assert request.device == torch.device('cpu')
    assert request.candidate_actions == tuple((b, m) for b in (4, 3) for m in range(5))


def test_raw_snapshots_resist_caller_and_predictor_mutation():
    predictor = FakePredictor(history_length=1)
    shield = shield_for(predictor)
    pre, outcome, action = observation(120), observation(121), torch.tensor([1, 2])
    shield.apply(pre, torch.zeros(10), [5, 5])
    shield.record_action(action, next_observation=outcome)
    pre.fill_(999)
    action.fill_(4)
    outcome.fill_(999)
    shield.apply(observation(121), torch.zeros(10), [5, 5])
    request = predictor.requests[-1]
    transition = request.past_transitions[0]
    assert transition.pre_observation[0] == 120
    assert transition.next_observation[0] == 121
    assert transition.executed_action == (1, 2)
    with pytest.raises(TypeError):
        transition.pre_observation[0] = 999
    with pytest.raises(FrozenInstanceError):
        transition.executed_action = (4, 4)
    with pytest.raises(FrozenInstanceError):
        request.current_observation = (999,)


def test_history_keeps_sliding_beyond_first_twenty_transitions():
    predictor = FakePredictor(history_length=3)
    shield = shield_for(predictor)
    advance(shield, count=30)
    shield.apply(observation(150), torch.zeros(10), [5, 5])
    request = predictor.requests[-1]
    assert len(predictor.requests) == 28
    assert [t.pre_observation[0] for t in request.past_transitions] == [147, 148, 149]
    assert [t.next_observation[0] for t in request.past_transitions] == [148, 149, 150]
    assert shield.step_counter == 30


def test_reset_clears_history_cooldown_pending_and_adapter_state():
    predictor = FakePredictor(history_length=1)
    shield = shield_for(predictor)
    advance(shield, count=1)
    shield.apply(observation(121), torch.zeros(10), [5, 5])
    shield.record_action((0, 0), next_observation=observation(50))
    shield.apply(observation(50), torch.zeros(10), [5, 5])
    assert shield.last_rescue_step == 2
    assert predictor.episode_state
    shield.reset()
    assert predictor.reset_calls == 2  # constructor and explicit new episode
    assert predictor.episode_state == []
    assert shield.step_counter == 0
    assert shield.last_rescue_step is None
    assert isinstance(shield.last_forecast, ForecastUnavailable)
    with pytest.raises(ValueError, match='preceding apply'):
        shield.record_action((0, 0), next_observation=observation(100))
    request_count = len(predictor.requests)
    shield.apply(observation(130), torch.zeros(10), [5, 5])
    assert len(predictor.requests) == request_count
    assert '0/1' in shield.last_forecast.reason


def test_explicit_unavailable_and_warmup_keep_legacy_nonpredictive_behavior():
    predictor = FakePredictor(history_length=1, result=ForecastUnavailable('adapter cold start'))
    shield = shield_for(predictor)
    logits = torch.arange(10.0)
    torch.testing.assert_close(shield.apply(observation(120), logits, [5, 5]), logits)
    shield.record_action((0, 0), next_observation=observation(121))
    torch.testing.assert_close(shield.apply(observation(121), logits, [5, 5]), logits)
    assert shield.last_forecast.reason == 'adapter cold start'


def test_rule_uses_point_horizon_min_and_finite_component_penalties():
    def forecast(request):
        values = torch.full((len(request.candidate_actions), 3), 120.0)
        for row, (bolus, meal) in enumerate(request.candidate_actions):
            if (bolus, meal) == (4, 3):
                values[row, 1] = 79.0
        return PointForecast(values)
    predictor = FakePredictor(history_length=1, result=forecast)
    shield = shield_for(predictor)
    advance(shield, count=1)
    logits = torch.arange(10.0)
    adjusted = shield.apply(observation(121), logits, [5, 5])
    expected = logits.clone()
    expected[2:5] -= 10
    torch.testing.assert_close(adjusted, expected)
    assert torch.softmax(adjusted[:5], dim=-1).min() > 0
    torch.testing.assert_close(logits, torch.arange(10.0))


def test_critical_rescue_is_a_finite_preference_and_does_not_query_predictor():
    predictor = FakePredictor(history_length=1)
    shield = shield_for(predictor)
    logits = torch.zeros(1, 10)
    result = shield.apply(observation(50).unsqueeze(0), logits, [5, 5])
    expected = torch.full((1, 10), -10.0)
    expected[0, 0] = expected[0, 6] = 10.0
    torch.testing.assert_close(result, expected)
    assert torch.softmax(result, dim=-1).min() > 0
    assert not predictor.requests
    assert 'bypassed' in shield.last_forecast.reason


def test_forecast_result_is_snapshotted_from_adapter_tensor():
    values = torch.full((10, 3), 120.0)
    predictor = FakePredictor(history_length=1, result=PointForecast(values))
    shield = shield_for(predictor)
    advance(shield, count=1)
    shield.apply(observation(121), torch.zeros(10), [5, 5])
    values.fill_(999)
    assert torch.all(shield.last_forecast.glucose_mg_dl == 120)


@pytest.mark.parametrize('result, error', [
    (PointForecast(torch.ones(10)), ValueError),
    (PointForecast(torch.ones(10, 4)), ValueError),
    (PointForecast(torch.full((10, 3), float('nan'))), ValueError),
    (PointForecast(torch.full((10, 3), float('inf'))), ValueError),
    (PointForecast(torch.ones(10, 3, dtype=torch.int64)), ValueError),
    (PointForecast([[120.0] * 3] * 10), ValueError),
    ('unavailable', TypeError),
    (lambda request: None, TypeError),
])
def test_bad_forecasts_raise_instead_of_becoming_safe(result, error):
    predictor = FakePredictor(history_length=1, result=result)
    shield = shield_for(predictor)
    advance(shield, count=1)
    with pytest.raises(error):
        shield.apply(observation(121), torch.zeros(10), [5, 5])


@pytest.mark.parametrize('kind, name', [('t1d', 'child'), ('t1d', 'adolescent#000'),
                                      ('t1d', 'other#003'), ('unknown', 'adult#001')])
def test_patient_identity_requires_explicit_full_supported_format(kind, name):
    with pytest.raises(ValueError):
        PatientIdentity(kind, name)


@pytest.mark.parametrize('metadata', [
    {'history_length': 0}, {'history_length': True}, {'horizon_steps': 0},
    {'continuation_policy': ''}, {'controller_interval_minutes': float('nan')},
    {'controller_interval_minutes': 0}, {'action_dims': (2, 3)},
    {'observation_features': tuple(reversed(GLUCOSIM_OBSERVATION_FEATURES))},
])
def test_invalid_metadata_rejected(metadata):
    values = dict(history_length=1, horizon_steps=3, continuation_policy='zero recommendations')
    values.update(metadata)
    with pytest.raises(ValueError):
        PredictorMetadata(**values)


def test_controller_interval_must_match_and_default_is_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('implicit GPU detection'))
    predictor = FakePredictor(controller_interval_minutes=10.0)
    with pytest.raises(ValueError, match='intervals must match'):
        shield_for(predictor)
    shield = shield_for(predictor, controller_interval_minutes=10.0)
    assert shield.device == torch.device('cpu')


def test_legacy_constructor_and_initialization_never_load_files(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('shield attempted filesystem/model/dataset access')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(os, 'listdir', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='explicitly injected predictor'):
        Shield(shield_type='child')
    with pytest.raises(ValueError, match='full PatientIdentity'):
        Shield(predictor=FakePredictor(), patient='child')
    shield = shield_for()
    shield.apply(observation(), torch.zeros(10), [5, 5])


def test_import_has_no_dataset_or_dynamics_dependencies():
    script = '''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('dynamics_trainers', 'dynamics_utils', 'FunctionEncoder')):
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from shield.predictive_shield import Shield
try:
    Shield(shield_type='child')
except ValueError as exc:
    assert 'explicitly injected predictor' in str(exc)
else:
    raise AssertionError('legacy constructor must fail')
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=30)


def test_transition_order_and_outcome_continuity_are_checked():
    shield = shield_for()
    with pytest.raises(ValueError, match='preceding apply'):
        shield.record_action((0, 0), next_observation=observation())
    shield.apply(observation(120), torch.zeros(10), [5, 5])
    with pytest.raises(TypeError, match='next_observation'):
        shield.record_action((0, 0))
    with pytest.raises(ValueError, match='observation changed'):
        shield.apply(observation(121), torch.zeros(10), [5, 5])
    shield.record_action((0, 0), next_observation=observation(121))
    with pytest.raises(ValueError, match='preceding apply'):
        shield.record_action((0, 0), next_observation=observation(122))
    with pytest.raises(ValueError, match='observation changed'):
        shield.apply(observation(122), torch.zeros(10), [5, 5])
    shield.apply(observation(121), torch.zeros(10), [5, 5])


@pytest.mark.parametrize('kwargs', [
    {'executed_action': (5, 0)}, {'executed_action': (-1, 0)},
    {'executed_action': (1.5, 0)}, {'executed_action': (float('nan'), 0)},
    {'executed_action': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]},
    {'recommended_action': (0, 5)}, {'accepted': (1, 0)}, {'accepted': (True,)},
    {'next_observation': torch.zeros(13)},
    {'next_observation': torch.full((14,), float('nan'))},
])
def test_invalid_completed_transition_does_not_consume_pending_action(kwargs):
    shield = shield_for()
    shield.apply(observation(), torch.zeros(10), [5, 5])
    values = dict(executed_action=(0, 0), next_observation=observation(121))
    values.update(kwargs)
    with pytest.raises(ValueError):
        shield.record_action(**values)
    shield.record_action((0, 0), next_observation=observation(121))
    assert shield.step_counter == 1


@pytest.mark.parametrize('obs, logits, dims', [
    (torch.zeros(13), torch.zeros(10), [5, 5]),
    (torch.zeros(2, 14), torch.zeros(10), [5, 5]),
    (torch.full((14,), float('nan')), torch.zeros(10), [5, 5]),
    (observation(), torch.zeros(9), [5, 5]),
    (observation(), torch.zeros(2, 10), [5, 5]),
    (observation(), torch.full((10,), float('inf')), [5, 5]),
    (observation(), torch.zeros(10, dtype=torch.int64), [5, 5]),
    (observation(), torch.zeros(10), [4, 6]),
])
def test_apply_rejects_mismatched_schema_and_nonfinite_inputs(obs, logits, dims):
    with pytest.raises(ValueError):
        shield_for().apply(obs, logits, dims)


@pytest.mark.parametrize('options', [
    {'config': PredictiveShieldConfig(top_k_bolus_levels=0)},
    {'config': PredictiveShieldConfig(logit_penalty=float('nan'))},
    {'params': ShieldParams(RESCUE_MEAL_LEVEL=5)},
    {'params': ShieldParams(MIN_INTERVENTION_BG_LOW=170)},
])
def test_invalid_rule_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        shield_for(**options)


@pytest.mark.parametrize('name', ['adult#011', 'child#999', 'adolescent#000'])
def test_patient_id_is_within_locked_simulator_range(name):
    with pytest.raises(ValueError, match='#001-#010'):
        PatientIdentity('t1d', name)


def test_explicit_cpu_device_alias_is_canonicalized():
    shield = shield_for(device='cpu:0')
    assert shield.device == torch.device('cpu')
    shield.apply(observation(), torch.zeros(10), [5, 5])


def test_duplicate_apply_requires_completed_transition_and_preserves_rescue_state():
    shield = shield_for()
    first = shield.apply(observation(50), torch.zeros(10), [5, 5])
    with pytest.raises(ValueError, match='previous apply'):
        shield.apply(observation(50), torch.zeros(10), [5, 5])
    assert shield.last_rescue_step == 0
    assert first[0] == 10
    shield.record_action((0, 1), next_observation=observation(51))
    shield.apply(observation(51), torch.zeros(10), [5, 5])
    assert shield.last_rescue_step == 0


@pytest.mark.parametrize('dtype, penalty, logit', [
    (torch.float32, 1e38, 3e38),
    (torch.float16, 1e5, 0),
])
def test_penalty_overflow_fails_before_committing_rescue_or_pending_state(dtype, penalty, logit):
    shield = shield_for(config=PredictiveShieldConfig(logit_penalty=penalty))
    with pytest.raises(ValueError, match='representable|remain finite'):
        shield.apply(observation(50), torch.full((10,), logit, dtype=dtype), [5, 5])
    assert shield.last_rescue_step is None
    with pytest.raises(ValueError, match='preceding apply'):
        shield.record_action((0, 1), next_observation=observation(51))


@pytest.mark.parametrize('options', [{'config': False}, {'config': {}}, {'params': {}}])
def test_rule_config_requires_explicit_configuration_types(options):
    with pytest.raises(TypeError):
        shield_for(**options)


def test_patient_identity_rejects_non_ascii_digits():
    with pytest.raises(ValueError):
        PatientIdentity('t1d', 'adult#００１')
