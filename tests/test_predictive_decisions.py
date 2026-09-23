"""Static/forecast attribution; fake forecasts establish no glucose safety claim."""

from dataclasses import FrozenInstanceError, asdict
import json

import pytest
import torch

from shield.predictive_shield import PredictiveShieldConfig, Shield, ShieldDecision
from shield.predictor import ForecastUnavailable, PatientIdentity, PointForecast, PredictorMetadata


class Predictor:
    metadata = PredictorMetadata(1, 3, 'fixed diagnostic continuation')

    def __init__(self, result=None):
        self.result = result
        self.requests = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def forecast(self, request):
        self.requests.append(request)
        if callable(self.result):
            return self.result(request)
        return self.result if self.result is not None else PointForecast(torch.full((10, 3), 120.))


def obs(cgm):
    return torch.tensor([cgm] + [0.] * 13)


def make(result=None, **options):
    predictor = Predictor(result)
    return Shield(predictor=predictor, patient=PatientIdentity('t1d', 'adolescent#001'),
                  config=PredictiveShieldConfig(**options)), predictor


def warm(shield, cgm):
    shield.apply(obs(cgm), torch.arange(10.), [5, 5])
    shield.record_action((0, 0), next_observation=obs(cgm),
                         recommended_action=(0, 0), accepted=(False, False))


def one_risky_pair(request):
    values = torch.full((len(request.candidate_actions), 3), 120.)
    for i, pair in enumerate(request.candidate_actions):
        if pair == (4, 3):
            values[i, 1] = 79.
    return PointForecast(values)


@pytest.mark.parametrize('invalid', [0, 1, None, 'false'])
def test_forecast_ablation_is_an_explicit_boolean(invalid):
    with pytest.raises(ValueError, match='use_forecast must be boolean'):
        make(use_forecast=invalid)


@pytest.mark.parametrize('cgm', [50., 60., 69., 70., 89., 90., 91., 159., 160., 180., 181., 260.])
def test_disabled_forecasts_keep_the_same_static_rules_as_empty_risk_sets(cgm):
    disabled, unused = make(use_forecast=False)
    unavailable, _ = make(ForecastUnavailable('diagnostic unavailable'))
    warm(disabled, cgm)
    warm(unavailable, cgm)
    logits = torch.arange(10.)[None]
    actual = disabled.apply(obs(cgm), logits, [5, 5])
    expected = unavailable.apply(obs(cgm), logits, [5, 5])
    assert torch.equal(actual, expected)
    decision = disabled.last_decision
    assert isinstance(decision, ShieldDecision)
    assert decision.static_mask == decision.final_mask == unavailable.last_decision.static_mask
    assert decision.prediction_mask == (0.,) * 10
    assert not decision.prediction_changed and not unused.requests
    assert decision.forecast_status == 'disabled'
    assert decision.forecast_reason == 'disabled ablation'
    assert decision.forecast_min_cgm == decision.candidate_below_threshold == ()
    assert len(disabled._history) == 1 and disabled.step_counter == 1


def test_inner_window_cap_reactivation_is_attributed_to_prediction():
    shield, predictor = make(one_risky_pair)
    warm(shield, 120.)
    logits = torch.arange(10.)
    result = shield.apply(obs(120.), logits, [5, 5])
    decision = shield.last_decision
    expected_mask = (0., 0., -10., -10., -10., 0., 0., 0., 0., 0.)
    assert decision.static_mask == (0.,) * 10
    assert decision.final_mask == decision.prediction_mask == expected_mask
    assert torch.equal(result, logits + torch.tensor(expected_mask))
    assert decision.static_reasons == ('minimal_intervention_bypass',)
    assert decision.prediction_reasons == ('hypo_risk', 'forecast_reenabled_bolus_cap')
    assert decision.flagged_bolus_levels == (4,) and decision.flagged_meal_levels == ()
    assert not decision.static_changed and decision.prediction_changed and decision.final_changed
    assert decision.forecast_status == 'available' and decision.forecast_reason is None
    assert decision.candidate_actions == predictor.requests[-1].candidate_actions
    assert decision.forecast_min_cgm[3] == 79. and decision.candidate_below_threshold[3]
    assert sum(decision.candidate_below_threshold) == 1


def test_existing_static_penalty_is_not_counted_twice_as_prediction():
    shield, _ = make(one_risky_pair)
    warm(shield, 80.)
    logits = torch.arange(10.)
    result = shield.apply(obs(80.), logits, [5, 5])
    decision = shield.last_decision
    assert decision.flagged_bolus_levels == (4,)
    assert decision.static_mask == decision.final_mask
    assert decision.prediction_mask == (0.,) * 10 and not decision.prediction_changed
    assert decision.static_changed and torch.equal(result, logits + torch.tensor(decision.static_mask))


def test_forecast_penalty_outside_static_window_changes_only_flagged_positive_level():
    shield, _ = make(one_risky_pair)
    warm(shield, 200.)
    logits = torch.arange(10.)
    result = shield.apply(obs(200.), logits, [5, 5])
    d = shield.last_decision
    assert d.static_mask == (0.,) * 10
    assert d.prediction_mask == (0., 0., 0., 0., -10., 0., 0., 0., 0., 0.)
    assert torch.equal(result, logits + torch.tensor(d.prediction_mask))
    # The change is still a finite marginal penalty, not pair exclusion.
    assert torch.softmax(result[:5], 0)[4] > 0


@pytest.mark.parametrize('enabled', [False, True])
def test_raw_high_forecast_flags_are_distinct_from_enabled_meal_penalties(enabled):
    def high(request):
        values = torch.full((len(request.candidate_actions), 3), 120.)
        values[3, 2] = 251.
        return PointForecast(values)

    shield, _ = make(high, use_meal_hyper_check=enabled)
    warm(shield, 200.)
    shield.apply(obs(200.), torch.arange(10.), [5, 5])
    d = shield.last_decision
    assert sum(d.candidate_above_threshold) == 1 and d.forecast_max_cgm[3] == 251.
    assert d.flagged_meal_levels == ((3,) if enabled else ())
    assert d.prediction_mask[8] == (-10. if enabled else 0.)
    assert d.prediction_changed == enabled


def test_zero_bolus_risk_preserves_legacy_cap_gate_but_never_penalizes_zero_level():
    def low_zero(request):
        values = torch.full((len(request.candidate_actions), 3), 120.)
        for i, (bolus, _) in enumerate(request.candidate_actions):
            if bolus == 0:
                values[i, 0] = 50.
        return PointForecast(values)

    shield, _ = make(low_zero)
    warm(shield, 120.)
    logits = torch.tensor([5., 4., 3., 2., 1., 0., 0., 0., 0., 0.])
    shield.apply(obs(120.), logits, [5, 5])
    d = shield.last_decision
    assert d.flagged_bolus_levels == (0,)
    assert d.prediction_mask[:5] == (0., 0., -10., -10., -10.)
    assert 'forecast_reenabled_bolus_cap' in d.prediction_reasons


@pytest.mark.parametrize('use_forecast', [False, True])
def test_critical_rescue_is_entirely_static_in_both_arms(use_forecast):
    shield, predictor = make(use_forecast=use_forecast)
    result = shield.apply(obs(50.), torch.zeros(10), [5, 5])
    d = shield.last_decision
    assert d.forecast_status == 'critical_rescue' and d.candidate_actions == ()
    assert d.static_mask == d.final_mask == tuple(result.tolist())
    assert d.static_changed and not d.prediction_changed and d.final_changed
    assert not predictor.requests and shield.last_rescue_step == 0
    shield.record_action((0, 1), next_observation=obs(50.), recommended_action=(0, 1), accepted=(False, True))
    shield.apply(obs(50.), torch.zeros(10), [5, 5])
    assert 'rescue_cooldown' in shield.last_decision.static_reasons
    assert shield.last_rescue_step == 0


def test_warmup_available_no_risk_and_disabled_are_separate_reasons():
    shield, _ = make()
    shield.apply(obs(120.), torch.zeros(10), [5, 5])
    first = shield.last_decision
    assert first.forecast_status == 'warmup' and first.prediction_reasons == ('warmup',)
    shield.record_action((0, 0), next_observation=obs(120.))
    shield.apply(obs(120.), torch.zeros(10), [5, 5])
    assert shield.last_decision.forecast_status == 'available'
    assert shield.last_decision.prediction_reasons == ('no_candidate_risk',)
    assert not shield.last_decision.final_changed


def test_decision_is_immutable_serializable_and_resets_without_stale_attribution():
    shield, _ = make()
    raw, logits = obs(80.), torch.arange(10.)
    shield.apply(raw, logits, [5, 5])
    decision = shield.last_decision
    stored = json.dumps(asdict(decision), allow_nan=False)
    raw.fill_(999.)
    logits.fill_(999.)
    assert json.dumps(asdict(decision), allow_nan=False) == stored
    with pytest.raises(FrozenInstanceError):
        decision.current_cgm = 999.
    with pytest.raises(TypeError):
        decision.static_mask[0] = 1.
    shield.reset()
    assert shield.last_decision is None
    shield.apply(obs(120.), torch.zeros(10), [5, 5])
    assert shield.last_decision.step_index == 0


def test_invalid_forecast_does_not_publish_a_new_decision():
    shield, predictor = make()
    warm(shield, 120.)
    previous = shield.last_decision
    predictor.result = PointForecast(torch.full((10, 3), float('nan')))
    with pytest.raises(ValueError, match='finite'):
        shield.apply(obs(120.), torch.zeros(10), [5, 5])
    assert shield.last_decision is previous and not shield._awaiting_transition


def test_changed_flags_track_actual_logits_not_an_ineffective_rounded_offset():
    shield, _ = make(one_risky_pair)
    warm(shield, 200.)
    # At this float32 magnitude a ten-unit offset rounds away.
    logits = torch.full((10,), 1e20)
    shield.apply(obs(200.), logits, [5, 5])
    d = shield.last_decision
    assert any(value != 0 for value in d.prediction_mask)
    assert not d.static_changed and not d.prediction_changed and not d.final_changed
