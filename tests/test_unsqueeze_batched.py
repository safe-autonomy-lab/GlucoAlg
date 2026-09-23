"""Regression: Unsqueeze must not double-batch already-vectorized single envs.

GlucoSim's DiabetesEnvs returns (1, obs_dim)/(1,) tensors even with num_envs=1,
while classic single-env CMDPs return unbatched (obs_dim,)/scalar values.
Unsqueeze must map both conventions to the canonical (1, obs_dim)/(1,) form.
"""
import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP
from omnisafe.envs.wrapper import Unsqueeze


class _StubCMDP(CMDP):
    need_time_limit_wrapper = False
    need_auto_reset_wrapper = False
    need_evaluation = False

    def __init__(self, vectorized: bool):
        self._vectorized = vectorized
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype='float32')
        self._action_space = spaces.MultiDiscrete([5, 5])

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self):
        return 1

    def _out(self):
        if self._vectorized:
            obs = torch.zeros(1, 4)
            scalar = torch.zeros(1)
            info = {'original_obs': torch.zeros(1, 4), 'original_reward': torch.zeros(1)}
        else:
            obs = torch.zeros(4)
            scalar = torch.zeros(())
            info = {'original_obs': torch.zeros(4), 'original_reward': torch.zeros(())}
        return obs, scalar, info

    def reset(self, seed=None, options=None):
        obs, _, info = self._out()
        return obs, info

    def step(self, action):
        self.last_action_shape = tuple(action.shape)
        self.last_action_dtype = action.dtype
        obs, scalar, info = self._out()
        return obs, scalar, scalar.clone(), scalar.clone(), scalar.clone(), info

    def set_seed(self, seed):
        return None

    def close(self):
        return None

    def render(self, *args, **kwargs):
        return None


def _check_canonical(obs, reward, cost, term, trunc, info):
    assert tuple(obs.shape) == (1, 4), obs.shape
    for name, tensor in [('reward', reward), ('cost', cost), ('term', term), ('trunc', trunc)]:
        assert tuple(tensor.shape) == (1,), (name, tensor.shape)
    assert tuple(info['original_obs'].shape) == (1, 4), info['original_obs'].shape
    assert tuple(info['original_reward'].shape) == (1,), info['original_reward'].shape


def test_unsqueeze_classic_unbatched_env():
    inner = _StubCMDP(vectorized=False)
    env = Unsqueeze(inner, device=torch.device('cpu'))
    obs, info = env.reset(seed=0)
    assert tuple(obs.shape) == (1, 4), obs.shape
    out = env.step(torch.zeros(1, 2, dtype=torch.long))
    _check_canonical(*out)
    assert inner.last_action_shape == (2,), inner.last_action_shape
    assert inner.last_action_dtype == torch.int64, inner.last_action_dtype


def test_unsqueeze_already_vectorized_env():
    inner = _StubCMDP(vectorized=True)
    env = Unsqueeze(inner, device=torch.device('cpu'))
    obs, info = env.reset(seed=0)
    assert tuple(obs.shape) == (1, 4), obs.shape
    out = env.step(torch.zeros(1, 2, dtype=torch.long))
    _check_canonical(*out)
    assert inner.last_action_shape == (1, 2), inner.last_action_shape
    assert inner.last_action_dtype == torch.int64, inner.last_action_dtype
