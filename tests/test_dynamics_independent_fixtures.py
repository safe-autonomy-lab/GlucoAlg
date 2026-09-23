"""Arithmetic oracles for causal windows and training-only normalization.

Contexts use chronological order. Normalization expectations are computed
from integer sums over training pre-observations, excluding held-out values.
These synthetic fixtures do not represent patient outcomes.
"""
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from glucoalg.dynamics.data import build_context_query, causal_windows, online_context_query
from glucoalg.dynamics.model import DynamicsModel, ModelConfig, fit_normalization
from glucoalg.dynamics.train import prepare_windows
from shield.predictor import ObservedTransition


TAIL = (1, 5, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0)
RAMP = (100, 110, 120, 130, 140, 150, 160, 170, 180)
IRREGULAR = (120, 125, 123, 130, 128, 140, 137, 145, 142)
IRREGULAR_ACTIONS = ((0, 0), (2, 0), (1, 1), (3, 0), (0, 2), (2, 1), (1, 0), (4, 0))
ACCEPTED_CGM = (150, 152, 151, 120, 118, 122, 125)
REJECTED_CGM = (150, 152, 151, 165, 170, 168, 172)
ACCEPT_ACTIONS = ((0, 0), (0, 0), (3, 0), (0, 0), (0, 0), (0, 0))
BOUNDARY_A = (110, 112, 111, 113, 115)
BOUNDARY_B = (200, 198, 202, 199, 201, 203, 200)
HELDOUT_EXTREME = (60, 45, 30, 520, 480, 420, 380)
LEAK = (100, 108, 115, 125, 140, 150)


def raw(cgm, actions=None):
    observations = np.array([(value, *TAIL) for value in cgm], dtype=np.float64)
    actions = ((1, 0),) * (len(cgm) - 1) if actions is None else actions
    return observations, np.asarray(actions, dtype=np.int64)


def context(observations, actions, t, *, candidates=None, count=2):
    return build_context_query(observations, actions, query_index=t, history_length=3,
                               horizon_steps=2, context_size=count, candidate_actions=candidates)


@pytest.mark.parametrize('cgm, actions, targets', [
    (RAMP, None, ((10, 20), (10, 20), (10, 20), (10, 20), (10, 20))),
    (IRREGULAR, IRREGULAR_ACTIONS, ((7, 5), (-2, 10), (12, 9), (-3, 5), (8, 5))),
    (ACCEPTED_CGM, ACCEPT_ACTIONS, ((-31, -33), (-2, 2), (4, 7))),
    (REJECTED_CGM, ACCEPT_ACTIONS, ((14, 19), (5, 3), (-2, 2))),
    (BOUNDARY_A, None, ((2, 4),)),
    (BOUNDARY_B, None, ((-3, -1), (2, 4), (2, -1))),
    (HELDOUT_EXTREME, None, ((490, 450), (-40, -100), (-60, -100))),
    (LEAK, None, ((10, 25), (15, 25))),
])
def test_independent_signed_cumulative_targets_and_query_edges(cgm, actions, targets):
    observations, recommendations = raw(cgm, actions)
    inputs, output, anchors = causal_windows(observations, recommendations, history_length=3, horizon_steps=2)
    np.testing.assert_array_equal(output, targets)
    np.testing.assert_array_equal(anchors, np.arange(2, len(cgm) - 2))
    # Pinned first input range; post-action rows would shift every element.
    np.testing.assert_array_equal(inputs[0, :, 0], cgm[:3])
    assert all(inputs[i, -1, 0] == cgm[t] for i, t in enumerate(anchors))


@pytest.mark.parametrize('t', [2, 3, 4])
def test_independent_context_shortfall_is_reported_without_padding(t):
    observations, actions = raw(RAMP)
    with pytest.raises(ValueError, match='Insufficient completed history'):
        context(observations, actions, t)
    if t == 4:
        one = context(observations, actions, t, count=1)
        np.testing.assert_array_equal(one.context_x[:, :, 0], [[100, 110, 120]])
        np.testing.assert_array_equal(one.context_y, [[10, 20]])


@pytest.mark.parametrize('t, expected_context_x, expected_context_y, expected_query, expected_target', [
    (5, ((120, 125, 123), (125, 123, 130)), ((7, 5), (-2, 10)), (130, 128, 140), (-3, 5)),
    (6, ((125, 123, 130), (123, 130, 128)), ((-2, 10), (12, 9)), (128, 140, 137), (8, 5)),
])
def test_independent_latest_fully_observed_context(t, expected_context_x, expected_context_y, expected_query, expected_target):
    observations, actions = raw(IRREGULAR, IRREGULAR_ACTIONS)
    value = context(observations, actions, t)
    np.testing.assert_array_equal(value.context_x[:, :, 0], expected_context_x)
    np.testing.assert_array_equal(value.context_y, expected_context_y)
    np.testing.assert_array_equal(value.query_x[0, :, 0], expected_query)
    np.testing.assert_array_equal(value.query_y, expected_target)


def test_recommendations_are_identical_before_different_acceptance_outcomes():
    accepted_obs, recommendations = raw(ACCEPTED_CGM, ACCEPT_ACTIONS)
    rejected_obs, _ = raw(REJECTED_CGM, ACCEPT_ACTIONS)
    accepted_x, accepted_y, _ = causal_windows(accepted_obs, recommendations, history_length=3, horizon_steps=2)
    rejected_x, rejected_y, _ = causal_windows(rejected_obs, recommendations, history_length=3, horizon_steps=2)
    np.testing.assert_array_equal(accepted_x[0], rejected_x[0])
    np.testing.assert_array_equal(accepted_y[0], [-31, -33])
    np.testing.assert_array_equal(rejected_y[0], [14, 19])
    # The proposal remains bolus3 despite different future acceptance.
    np.testing.assert_array_equal(rejected_x[0, -1, 14:19], [0, 0, 0, 1, 0])


def test_online_parity_preserves_proposals_separately_from_executed_indices():
    observations, actions = raw(IRREGULAR, IRREGULAR_ACTIONS)
    t = 5
    transitions = tuple(ObservedTransition(tuple(observations[i]), (0, 0), tuple(observations[i + 1]),
                                           tuple(actions[i]), (False, False)) for i in range(t))
    candidates = ((2, 1), (4, 3))
    offline = context(observations, actions, t, candidates=candidates)
    online = online_context_query(transitions, tuple(observations[t]), candidates,
                                 history_length=3, horizon_steps=2, context_size=2)
    for name in ('context_x', 'context_y', 'query_x'):
        np.testing.assert_array_equal(getattr(online, name), getattr(offline, name))
    assert online.query_y is None  # an unobserved candidate has no observed label
    np.testing.assert_array_equal(online.query_x[1, -1, 14:], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    # Last past proposal is meal2 even though every nominal execution was0.
    np.testing.assert_array_equal(online.query_x[0, -2, 19:], [0, 0, 1, 0, 0])


def test_episode_boundary_never_supplies_another_episodes_context_or_warmup():
    a_obs, a_actions = raw(BOUNDARY_A)
    b_obs, b_actions = raw(BOUNDARY_B)
    samples = [SimpleNamespace(observations=obs, recommended_actions=actions)
               for obs, actions in ((a_obs, a_actions), (b_obs, b_actions))]
    config = ModelConfig(history_length=3, horizon_steps=2, context_size=2)
    with pytest.raises(ValueError, match='no complete causal'):
        prepare_windows(samples, config)
    one = context(b_obs, b_actions, 4, count=1)
    np.testing.assert_array_equal(one.context_x[0, :, 0], [200, 198, 202])
    for t in (0, 1):
        with pytest.raises(ValueError, match='Insufficient completed history'):
            context(b_obs, b_actions, t)
    cross_episode = tuple(ObservedTransition(tuple(obs[i]), (0, 0), tuple(obs[i + 1]), (1, 0))
                          for obs, indices in ((a_obs, (2, 3)), (b_obs, (0, 1, 2))) for i in indices)
    with pytest.raises(ValueError, match='not contiguous'):
        online_context_query(cross_episode, tuple(b_obs[3]), ((1, 0),),
                             history_length=3, horizon_steps=2, context_size=2)


def test_future_sentinel_changes_targets_but_not_model_inputs_or_forecasts():
    observations, actions = raw(IRREGULAR, IRREGULAR_ACTIONS)
    changed = observations.copy()
    changed[6:, 0] = 777
    changed_actions = actions.copy()
    changed_actions[6:] = (4, 4)
    original = context(observations, actions, 5)
    perturbed = context(changed, changed_actions, 5)
    for name in ('context_x', 'context_y', 'query_x'):
        np.testing.assert_array_equal(getattr(original, name), getattr(perturbed, name))
    np.testing.assert_array_equal(original.query_y, [-3, 5])
    np.testing.assert_array_equal(perturbed.query_y, [637, 637])
    # Exercise actual eager BA-NODE with fixed weights, no training or data files.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(841)
            model = DynamicsModel(ModelConfig(history_length=3, horizon_steps=2, context_size=2,
                                               hidden_size=8, n_basis=2), np.zeros(14), np.ones(14)).eval()
        def predict(value):
            tensors = [torch.tensor(getattr(value, key), dtype=torch.float32)[None]
                       for key in ('context_x', 'context_y', 'query_x')]
            with torch.no_grad():
                return model(*tensors)[0]
        torch.testing.assert_close(predict(original), predict(perturbed), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_known_leaky_rows_fail_the_future_sentinel_and_trivial_predictor_oracles():
    observations, actions = raw(LEAK)
    correct_x, target, _ = causal_windows(observations, actions, history_length=3, horizon_steps=2)
    correct_row = correct_x[0, :, 0]
    np.testing.assert_array_equal(correct_row, [100, 108, 115])
    leaky_row = observations[1:4, 0]  # deliberately reproduce legacy post-action alignment
    np.testing.assert_array_equal(leaky_row, [108, 115, 125])
    correct_prediction = correct_row[-1] - 115
    leaky_prediction = leaky_row[-1] - 115
    assert abs(correct_prediction - target[0, 0]) == 10
    assert abs(leaky_prediction - target[0, 0]) == 0
    assert abs(leaky_prediction - target[0, 1]) == 15  # only the first-step answer was leaked
    changed = observations.copy()
    changed[3:, 0] = 777
    changed_x, _, _ = causal_windows(changed, actions, history_length=3, horizon_steps=2)
    np.testing.assert_array_equal(changed_x[0], correct_x[0])
    assert not np.array_equal(changed[1:4, 0], leaky_row)


def test_train_only_normalization_uses_the_independently_corrected16_row_pool():
    ramp_obs, _ = raw(RAMP)
    irregular_obs, _ = raw(IRREGULAR)
    heldout_obs, _ = raw(HELDOUT_EXTREME)
    training = [SimpleNamespace(metadata={'split': 'train'}, observations=value)
                for value in (ramp_obs, irregular_obs)]
    mean, std = fit_normalization(training)
    # Independent integer arithmetic: n16,sum2128,sum_of_squares287832.
    assert 2128 / 16 == 133
    expected_std = math.sqrt(287832 / 16 - 133 ** 2)
    assert expected_std == math.sqrt(300.5)
    assert mean[0].item() == 133
    assert std[0].item() == pytest.approx(expected_std, abs=1e-6)
    contaminated_mean = 3683 / 22
    contaminated_std = math.sqrt(971557 / 22 - contaminated_mean ** 2)
    assert abs(contaminated_mean - mean[0].item()) > 34
    assert contaminated_std > 7 * std[0].item()
    before = mean.clone(), std.clone()
    heldout = SimpleNamespace(metadata={'split': 'validation'}, observations=heldout_obs)
    with pytest.raises(ValueError, match='training episodes only'):
        fit_normalization([*training, heldout])
    after = fit_normalization(training)
    for old, new in zip(before, after):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
