"""Forecast-contribution controls using only fake point forecasts."""

import pytest
import torch

from shield.predictor import PointForecast
from test_predictive_decisions import make, obs, warm


def low_forecast(request):
    return PointForecast(torch.full((len(request.candidate_actions), 3), 65.))


def risky_level(level, *, meal=False):
    def forecast(request):
        values = torch.full((len(request.candidate_actions), 3), 120.)
        for index, pair in enumerate(request.candidate_actions):
            if pair[1 if meal else 0] == level:
                values[index, 1] = 260. if meal else 65.
        return PointForecast(values)
    return forecast


@pytest.mark.parametrize('scale', [-1, -.1, float('nan'), float('inf'), float('-inf'), True, False, '0', None])
def test_scale_requires_a_finite_nonnegative_number(scale):
    with pytest.raises(ValueError, match='forecast_penalty_scale.*finite and nonnegative'):
        make(forecast_penalty_scale=scale)


@pytest.mark.parametrize('cgm', [50., 60., 69., 70., 89., 90., 91., 159., 160., 180., 181., 260.])
@pytest.mark.parametrize('batched', [False, True])
def test_zero_is_static_with_forecast_risks_retained(cgm, batched):
    zero, queried = make(low_forecast, forecast_penalty_scale=0)
    static, disabled = make(low_forecast, use_forecast=False)
    warm(zero, cgm)
    warm(static, cgm)
    logits = torch.arange(10.)
    if batched:
        logits = logits[None]
    actual = zero.apply(obs(cgm), logits, [5, 5])
    expected = static.apply(obs(cgm), logits, [5, 5])
    assert torch.equal(actual, expected)
    d = zero.last_decision
    assert d.final_mask == d.static_mask == static.last_decision.final_mask
    assert d.prediction_mask == (0.,) * 10 and not d.prediction_changed
    assert d.forecast_status == 'available' and any(d.candidate_below_threshold)
    assert len(queried.requests) == 1 and not disabled.requests


@pytest.mark.parametrize('scale', [0., .1, .3, 1., 3.])
@pytest.mark.parametrize('cgm,level,static_levels,incremental_levels', [
    (80., 1, (2, 3, 4), (1,)),
    (80., 4, (2, 3, 4), ()),
    (120., 1, (), (1, 2, 3, 4)),
    (200., 4, (), (4,)),
])
def test_scale_changes_only_the_incremental_mask(scale, cgm, level, static_levels, incremental_levels):
    shield, _ = make(risky_level(level), forecast_penalty_scale=scale)
    warm(shield, cgm)
    logits = torch.tensor([0., 9., 0., 0., 8., 0., 0., 0., 0., 0.])
    actual = shield.apply(obs(cgm), logits, [5, 5])
    expected_static = torch.zeros(10)
    expected_static[list(static_levels)] = -10.
    expected_increment = torch.zeros(10)
    expected_increment[list(incremental_levels)] = -10. * scale
    d = shield.last_decision
    torch.testing.assert_close(torch.tensor(d.static_mask), expected_static, rtol=0, atol=0)
    torch.testing.assert_close(torch.tensor(d.prediction_mask), expected_increment, rtol=0, atol=0)
    torch.testing.assert_close(actual, logits + expected_static + expected_increment, rtol=0, atol=0)


def test_zero_level_risk_scales_cap_restoration_without_penalizing_zero():
    shield, _ = make(risky_level(0), forecast_penalty_scale=.3)
    warm(shield, 120.)
    shield.apply(obs(120.), torch.tensor([5., 4., 3., 2., 1., 0., 0., 0., 0., 0.]), [5, 5])
    d = shield.last_decision
    assert d.flagged_bolus_levels == (0,)
    assert d.final_mask == (0., 0., -3., -3., -3., 0., 0., 0., 0., 0.)
    assert 'forecast_reenabled_bolus_cap' in d.prediction_reasons


@pytest.mark.parametrize('enabled', [False, True])
def test_meal_contribution_is_scaled_only_when_hyper_check_is_enabled(enabled):
    shield, _ = make(risky_level(3, meal=True), forecast_penalty_scale=.3,
                     use_meal_hyper_check=enabled)
    warm(shield, 200.)
    shield.apply(obs(200.), torch.arange(10.), [5, 5])
    expected = [0.] * 10
    expected[8] = -3. if enabled else 0.
    assert shield.last_decision.prediction_mask == tuple(expected)


@pytest.mark.parametrize('scale', [0., .1, 1., 3.])
def test_rescue_and_cooldown_are_not_scaled(scale):
    shield, predictor = make(low_forecast, forecast_penalty_scale=scale)
    result = shield.apply(obs(50.), torch.zeros(10), [5, 5])
    expected = torch.full((10,), -10.)
    expected[[0, 6]] = 10.
    assert torch.equal(result, expected)
    assert shield.last_decision.static_mask == shield.last_decision.final_mask
    assert shield.last_decision.prediction_mask == (0.,) * 10
    assert shield.last_rescue_step == 0 and not predictor.requests
    shield.record_action((0, 1), next_observation=obs(50.), recommended_action=(0, 1))
    shield.apply(obs(50.), torch.zeros(10), [5, 5])
    assert shield.last_rescue_step == 0
    assert 'rescue_cooldown' in shield.last_decision.static_reasons


@pytest.mark.parametrize('scale', [0., .01])
def test_scaled_mask_is_applied_before_a_hypothetical_legacy_output_overflows(scale):
    shield, _ = make(low_forecast, logit_penalty=60000., forecast_penalty_scale=scale)
    warm(shield, 200.)
    logits = torch.full((10,), -60000., dtype=torch.float16)
    actual = shield.apply(obs(200.), logits, [5, 5])
    assert torch.isfinite(actual).all()
    assert shield._awaiting_transition
    if scale == 0:
        assert torch.equal(actual, logits)
    legacy, _ = make(low_forecast, logit_penalty=60000.)
    warm(legacy, 200.)
    with pytest.raises(ValueError, match='adjusted logits must remain finite'):
        legacy.apply(obs(200.), logits, [5, 5])
    assert not legacy._awaiting_transition


def test_scaled_mask_overflow_does_not_publish_a_decision_or_complete_a_transition():
    shield, _ = make(low_forecast, forecast_penalty_scale=1e20)
    warm(shield, 200.)
    before = shield.last_decision
    with pytest.raises(ValueError, match='adjusted logits must remain finite'):
        shield.apply(obs(200.), torch.zeros(10, dtype=torch.float16), [5, 5])
    assert shield.last_decision is before and not shield._awaiting_transition
    assert shield.step_counter == 1 and shield.last_rescue_step is None


@pytest.mark.parametrize('penalty', [.125, 3.25, 10., 31.])
def test_omitted_and_explicit_unit_scale_preserve_custom_legacy_penalties(penalty):
    omitted, _ = make(low_forecast, logit_penalty=penalty)
    explicit, _ = make(low_forecast, logit_penalty=penalty, forecast_penalty_scale=1.)
    warm(omitted, 120.)
    warm(explicit, 120.)
    logits = torch.arange(10.)
    assert torch.equal(omitted.apply(obs(120.), logits, [5, 5]),
                       explicit.apply(obs(120.), logits, [5, 5]))
    assert omitted.last_decision == explicit.last_decision
