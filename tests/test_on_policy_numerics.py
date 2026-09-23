"""Exercise production update methods with small analytic CPU fixtures.

Loading methods directly avoids importing the optional simulator. Their bodies
are compiled unchanged; actor plumbing and distributed collectives are fixtures.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ROOT / 'omnisafe/algorithms/on_policy'


def method(relative, cls, name, **extra):
    path = ALGORITHMS / relative
    tree = ast.parse(path.read_text())
    node = next(
        node for item in tree.body if isinstance(item, ast.ClassDef) and item.name == cls
        for node in item.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = {
        'torch': torch,
        'distributed': NS(dist_avg=lambda value: value, avg_grads=lambda actor: None),
        'DataLoader': DataLoader,
        'TensorDataset': TensorDataset,
        'track': lambda values, **kwargs: values,
        **extra,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


_spec = importlib.util.spec_from_file_location(
    'numerics_distributions', ROOT / 'omnisafe/utils/distributions.py',
)
_distributions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distributions)


class Logger:
    def __init__(self, cost=0.0):
        self.values = {}
        self.cost = cost

    def log(self, *args):
        pass

    def store(self, values):
        self.values.update(values)

    def get_stats(self, key):
        return [self.cost]


class Actor:
    std = 0.0

    def __init__(self, distribution):
        self.distribution = distribution

    def __call__(self, *args, **kwargs):
        return self.distribution

    def zero_grad(self):
        pass

    def log_prob(self, actions):
        value = self.distribution.log_prob(actions)
        return value.sum(-1) if value.ndim > 1 else value


def distributions(kind, repeat=1):
    old = torch.zeros(4, 4).repeat(repeat, 1)
    new = torch.tensor([[.01, -.01, .01, -.01], [2., -2., 1., -1.],
                        [.02, -.02, .01, -.01], [-2., 2., -1., 1.]]).repeat(repeat, 1)
    new.requires_grad_()
    if kind == 'gaussian':
        factory = lambda value: torch.distributions.Normal(value, torch.ones_like(value))
        actions = torch.zeros(4 * repeat, 4)
    elif kind == 'multicategorical':
        factory = lambda value: _distributions.MultiCategoricalDistribution(value, [2, 2])
        actions = torch.zeros(4 * repeat, 2, dtype=torch.long)
    else:
        factory = _distributions.CategoricalDistribution
        actions = torch.zeros(4 * repeat, 1, dtype=torch.long)
    return factory(old), factory(new), actions, new, factory


@pytest.mark.parametrize('kind', ['categorical', 'multicategorical', 'gaussian'])
def test_epoch_kl_is_batch_invariant(kind):
    values = []
    for repeat in (1, 2):
        old, new, actions, _, _ = distributions(kind, repeat)
        n = actions.shape[0]
        observations = torch.zeros(n, 2)
        data = {key: torch.zeros(n) for key in (
            'logp', 'target_value_r', 'target_value_c', 'adv_r', 'adv_c',
        )}
        data.update(obs=observations, original_obs=observations, act=actions)
        sequence = iter((old, new))
        obj = NS(
            _buf=NS(get=lambda: data), _logger=Logger(),
            _actor_critic=NS(actor=lambda *args, **kwargs: next(sequence)),
            _cfgs=NS(algo_cfgs=NS(batch_size=n, update_iters=1, use_cost=False, kl_early_stop=False)),
            _update_actor=lambda *args, **kwargs: None,
            _update_reward_critic=lambda *args: None,
        )
        method('base/policy_gradient.py', 'PolicyGradient', '_update')(obj)
        expected = torch.distributions.kl_divergence(old, new)
        if expected.ndim > 1:
            expected = expected.sum(-1)
        values.append(obj._logger.values['Train/KL'])
        assert values[-1] == pytest.approx(expected.mean().item())
    assert values[0] == pytest.approx(values[1])


@pytest.mark.parametrize('kind', ['categorical', 'multicategorical', 'gaussian'])
@pytest.mark.parametrize('algorithm', ['FOCOPS', 'CUP'])
def test_losses_and_gradients_match_independent_single_sample_oracle(kind, algorithm):
    old, new, actions, parameters, factory = distributions(kind)
    adv = torch.tensor([2., -3., .5, 1.5])
    logp = torch.tensor([-.5, -1., -.2, -.8])
    cfg = NS(focops_lam=10., focops_eta=.02, entropy_coef=0., gamma=.99, lam=.95)
    obj = NS(_actor_critic=NS(actor=Actor(new)), _p_dist=old, _logger=Logger(),
             _cfgs=NS(algo_cfgs=cfg), _lagrange=NS(lagrangian_multiplier=1.5))
    name = '_loss_pi' if algorithm == 'FOCOPS' else '_loss_pi_cost'
    function = method(f'first_order/{algorithm.lower()}.py', algorithm, name)
    obs = torch.zeros(4, 2)
    actual = function(obj, obs, actions, logp, adv, original_obs=obs)
    terms = []
    for index in range(4):
        new_i = factory(parameters[index:index + 1])
        old_i = factory(torch.zeros_like(parameters[index:index + 1]))
        kl = torch.distributions.kl_divergence(new_i, old_i).sum()
        ratio = torch.exp(new_i.log_prob(actions[index:index + 1]).sum() - logp[index])
        if algorithm == 'FOCOPS':
            terms.append((kl - ratio * adv[index] / 10.) * (kl.detach() <= .02))
        else:
            coefficient = (1 - cfg.gamma * cfg.lam) / (1 - cfg.gamma)
            terms.append(kl + 1.5 * coefficient * ratio * adv[index])
    expected = torch.stack(terms).mean()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, parameters)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def search(kl_value, reward_value=lambda theta: 0., cost_value=lambda theta: 0., steps=10):
    state = {'theta': torch.zeros(1), 'candidates': []}
    logger = Logger()

    def set_params(actor, theta):
        state['theta'] = theta.clone()

    def kl(old, new):
        state['candidates'].append(new.item())
        return torch.tensor(kl_value(new.item()))

    obj = NS(
        _actor_critic=NS(actor=lambda *args, **kwargs: state['theta'].clone()),
        _cfgs=NS(algo_cfgs=NS(target_kl=.1)), _logger=logger,
        _loss_pi=lambda **kwargs: torch.tensor(reward_value(state['theta'].item())),
        _loss_pi_cost=lambda **kwargs: torch.tensor(cost_value(state['theta'].item())),
    )
    function = method('second_order/cpo.py', 'CPO', '_cpo_search_step',
                      get_flat_params_from=lambda actor: state['theta'].clone(),
                      set_param_values_to_model=set_params)
    with patch('torch.distributions.kl.kl_divergence', side_effect=kl):
        try:
            result = function(obj, torch.ones(1), torch.ones(1), torch.zeros(1),
                              *[torch.zeros(1)] * 5, torch.tensor(0.), torch.tensor(0.),
                              torch.zeros(1), total_steps=steps)
        finally:
            assert state['theta'].item() == 0., 'Line search must restore the actor'
    return result, state, logger.values


def test_nonfinite_kl_backtracks_until_finite_candidate():
    (step, accepted), state, values = search(lambda theta: float('inf') if theta > .5 else .01)
    assert accepted == 5
    assert step.item() == pytest.approx(.8 ** 4)
    assert state['candidates'] == pytest.approx([.8 ** i for i in range(5)])
    assert values['Train/KL'].item() == pytest.approx(.01)


@pytest.mark.parametrize('loss', ['reward', 'cost'])
def test_nonfinite_losses_cannot_be_accepted(loss):
    kwargs = {f'{loss}_value': lambda theta: float('nan') if theta > .5 else 0.}
    (step, accepted), _, _ = search(lambda theta: .01, **kwargs)
    assert accepted == 5
    assert step.item() == pytest.approx(.8 ** 4)


def test_rejected_search_logs_restored_policy_kl():
    (step, accepted), _, values = search(lambda theta: float('inf'))
    assert accepted == 0
    assert step.item() == 0.
    assert values['Train/KL'].item() == 0.


def test_unexpected_exception_restores_actor():
    def fail(theta):
        raise RuntimeError('loss failure')
    with pytest.raises(RuntimeError, match='loss failure'):
        search(lambda theta: .01, cost_value=fail)


def proposal(reward_gradient, cost_gradient, violation):
    fisher = torch.tensor([2., 4.])
    actor = Actor(object())
    logger = Logger(violation)
    gradients = iter((-torch.tensor(reward_gradient), torch.tensor(cost_gradient)))
    captured = {}

    def search_step(**kwargs):
        captured['step'] = kwargs['step_direction'].clone()
        return torch.zeros(2), 0

    obj = NS(
        _actor_critic=NS(actor=actor, zero_grad=lambda: None), _logger=logger,
        _cfgs=NS(algo_cfgs=NS(fvp_sample_freq=1, cg_iters=10, target_kl=.1, cost_limit=0.)),
        _fvp=lambda vector: fisher * vector,
        _loss_pi=lambda *args, **kwargs: torch.tensor(0., requires_grad=True),
        _loss_pi_cost=lambda *args, **kwargs: torch.tensor(0., requires_grad=True),
        _cpo_search_step=search_step,
    )
    function = method('second_order/pcpo.py', 'PCPO', '_update_actor',
                      get_flat_params_from=lambda actor: torch.zeros(2),
                      get_flat_gradients_from=lambda actor: next(gradients),
                      set_param_values_to_model=lambda *args: None,
                      conjugate_gradients=lambda function, vector, iterations: vector / fisher)
    function(obj, *[torch.zeros(2)] * 6)
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in logger.values.values())
    return captured['step']


def test_pcpo_natural_reward_step_obeys_fisher_budget():
    step = proposal([1., 2.], [0., 1.], -100.)
    assert step[0].item() == pytest.approx(step[1].item())
    assert (.5 * (step.square() * torch.tensor([2., 4.])).sum()).item() == pytest.approx(.1)


def test_pcpo_projection_satisfies_active_linearized_cost_boundary():
    step = proposal([1., 2.], [0., 1.], .1)
    assert step[1].item() == pytest.approx(-.1)
    assert step[0].item() > 0.


@pytest.mark.parametrize('reward,cost,violation,expected', [
    ([0., 0.], [0., 0.], -1., [0., 0.]),
    ([0., 0.], [0., 0.], 1., [0., 0.]),
    ([0., 0.], [0., 1.], .1, [0., -.1]),
    ([1., 2.], [0., 0.], 1., [0., 0.]),
])
def test_pcpo_degenerate_gradients_have_finite_defined_steps(reward, cost, violation, expected):
    torch.testing.assert_close(proposal(reward, cost, violation), torch.tensor(expected))


def test_pcpo_feasible_zero_cost_gradient_retains_reward_step():
    step = proposal([1., 2.], [0., 0.], -1.)
    assert step.norm() > 0.
    assert step[0].item() == pytest.approx(step[1].item())


@pytest.mark.parametrize('cost', [0., 2.])
def test_crpo_logs_both_counters_on_every_branch(cost):
    obj = NS(_logger=Logger(cost), _cfgs=NS(algo_cfgs=NS(cost_limit=1., distance=0.)),
             _rew_update=0, _cost_update=0)
    reward, penalty = torch.ones(2), torch.ones(2) * 2
    result = method('primal/crpo.py', 'OnCRPO', '_compute_adv_surrogate')(obj, reward, penalty)
    assert obj._logger.values == {'Misc/RewUpdate': int(cost <= 1.), 'Misc/CostUpdate': int(cost > 1.)}
    torch.testing.assert_close(result, reward if cost <= 1. else -penalty)


@pytest.mark.parametrize('algorithm,relative', [
    ('PolicyGradient', 'base/policy_gradient.py'), ('PPO', 'base/ppo.py'),
])
def test_policy_std_accumulator_is_populated(algorithm, relative):
    _, distribution, actions, _, _ = distributions('categorical')
    obj = NS(_actor_critic=NS(actor=Actor(distribution)), _logger=Logger(),
             _cfgs=NS(algo_cfgs=NS(clip=.2, entropy_coef=0.)))
    observations = torch.zeros(4, 2)
    method(relative, algorithm, '_loss_pi')(
        obj, observations, actions, torch.zeros(4), torch.ones(4), original_obs=observations,
    )
    assert obj._logger.values['Train/PolicyStd'] == 0.


def test_cup_never_constructs_cross_sample_loss_matrix():
    old, new, actions, _, _ = distributions('categorical')
    obj = NS(_actor_critic=NS(actor=Actor(new)), _p_dist=old, _logger=Logger(),
             _cfgs=NS(algo_cfgs=NS(gamma=.99, lam=.95)),
             _lagrange=NS(lagrangian_multiplier=1.))
    shapes = []
    original_mean = torch.Tensor.mean

    def mean(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return original_mean(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, 'mean', mean):
        method('first_order/cup.py', 'CUP', '_loss_pi_cost')(
            obj, torch.zeros(4, 2), actions, torch.zeros(4), torch.ones(4),
            original_obs=torch.zeros(4, 2),
        )
    assert (4, 4) not in shapes
    assert (4,) in shapes
