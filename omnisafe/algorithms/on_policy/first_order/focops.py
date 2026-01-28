# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the FOCOPS algorithm with discrete action support."""

from __future__ import annotations

from typing import Union
import torch
from gymnasium import spaces
from rich.progress import track
from torch.distributions import Normal, Distribution
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from omnisafe.utils.distributions import MultiCategoricalDistribution, CategoricalDistribution


@registry.register
class FOCOPS(PolicyGradient):
    """The First Order Constrained Optimization in Policy Space (FOCOPS) algorithm.

    References:
        - Title: First Order Constrained Optimization in Policy Space
        - Authors: Yiming Zhang, Quan Vuong, Keith W. Ross.
        - URL: `FOCOPS <https://arxiv.org/abs/2002.06506>`_
    
    Note: Extended to support discrete action spaces (Discrete/MultiDiscrete).
    For discrete actions, KL divergence is computed directly between categorical
    distributions rather than using the Normal distribution approximation.
    """

    _p_dist: Union[Normal, MultiCategoricalDistribution, CategoricalDistribution]
    _is_discrete: bool = False
    _action_dims: list = None

    def _init(self) -> None:
        """Initialize the FOCOPS specific model.

        The FOCOPS algorithm uses a Lagrange multiplier to balance the cost and reward.
        Extended to detect and handle discrete action spaces.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        
        # Detect if action space is discrete
        act_space = self._env.action_space
        if isinstance(act_space, spaces.Discrete):
            self._is_discrete = True
            self._action_dims = [act_space.n]
        elif isinstance(act_space, spaces.MultiDiscrete):
            self._is_discrete = True
            self._action_dims = act_space.nvec.tolist()
        else:
            self._is_discrete = False
            self._action_dims = None

    def _init_log(self) -> None:
        """Log the FOCOPS specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        original_obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute pi/actor loss.

        In FOCOPS, the loss is defined as:

        .. math::
            :nowrap:

            \begin{eqnarray}
                L = \nabla_{\theta} D_{K L} \left( \pi_{\theta}^{'} \| \pi_{\theta} \right)[s]
                - \frac{1}{\eta} \underset{a \sim \pi_{\theta}}{\mathbb{E}} \left[
                    \frac{\nabla_{\theta} \pi_{\theta} (a \mid s)}{\pi_{\theta}(a \mid s)}
                    \left( A^{R}_{\pi_{\theta}} (s, a) - \lambda A^C_{\pi_{\theta}} (s, a) \right)
                \right]
            \end{eqnarray}

        where :math:`\eta` is a hyperparameter, :math:`\lambda` is the Lagrange multiplier,
        :math:`A_{\pi_{\theta_k}}(s, a)` is the advantage function,
        :math:`A^C_{\pi_{\theta_k}}(s, a)` is the cost advantage function,
        :math:`\pi^*` is the optimal policy, and :math:`\pi_{\theta}` is the current policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs, original_obs=original_obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)

        # Compute KL divergence - handle both discrete and continuous
        kl = torch.distributions.kl_divergence(distribution, self._p_dist)
        # Handle different output shapes
        if kl.dim() > 1:
            kl = kl.sum(-1, keepdim=True)
        elif kl.dim() == 1:
            kl = kl.unsqueeze(-1)
            
        loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) * (
            kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        ).type(torch.float32)
        loss = loss.mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        FOCOPS uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        return (adv_r - self._lagrange.lagrangian_multiplier * adv_c) / (
            1 + self._lagrange.lagrangian_multiplier
        )

    def _update(self) -> None:
        r"""Update actor, critic, and Lagrange multiplier parameters.

        In FOCOPS, the Lagrange multiplier is updated as the naive lagrange multiplier update.

        Then in each iteration of the policy update, FOCOPS calculates current policy's
        distribution, which used to calculate the policy loss.
        
        For discrete action spaces, we store logits instead of mean/std.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)

        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, original_obs = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['original_obs'],
        )
        before_update_obs = obs
        before_update_original_obs = original_obs
        
        
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs, original_obs=original_obs)
            if self._is_discrete:
                # For discrete actions, store logits to reconstruct distribution
                old_logits = old_distribution.logits
            else:
                # For continuous actions, store mean and std
                old_mean = old_distribution.mean
                old_std = old_distribution.stddev

        # Create dataloader with appropriate data
        if self._is_discrete:
            dataloader = DataLoader(
                dataset=TensorDataset(
                    obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
                    old_logits,
                    original_obs,
                ),
                batch_size=self._cfgs.algo_cfgs.batch_size,
                shuffle=True,
            )
        else:
            dataloader = DataLoader(
                dataset=TensorDataset(
                    obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
                    old_mean,
                    old_std,
                ),
                batch_size=self._cfgs.algo_cfgs.batch_size,
                shuffle=True,
            )

        final_steps = self._cfgs.algo_cfgs.update_iters
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            if self._is_discrete:
                # Discrete action space iteration
                for (
                    obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
                    old_logits_batch,
                    original_obs,
                ) in dataloader:
                    self._update_reward_critic(obs, target_value_r)
                    if self._cfgs.algo_cfgs.use_cost:
                        self._update_cost_critic(obs, target_value_c)

                    # Reconstruct old distribution from stored logits
                    if len(self._action_dims) > 1:
                        self._p_dist = MultiCategoricalDistribution(old_logits_batch, self._action_dims)
                    else:
                        self._p_dist = CategoricalDistribution(old_logits_batch)
                    self._update_actor(obs, act, logp, adv_r, adv_c, original_obs=original_obs)
            else:
                # Continuous action space iteration (original logic)
                for (
                    obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
                    old_mean,
                    old_std,
                ) in dataloader:
                    self._update_reward_critic(obs, target_value_r)
                    if self._cfgs.algo_cfgs.use_cost:
                        self._update_cost_critic(obs, target_value_c)

                    self._p_dist = Normal(old_mean, old_std)
                    self._update_actor(obs, act, logp, adv_r, adv_c, original_obs=original_obs)

            new_distribution = self._actor_critic.actor(before_update_obs, original_obs=before_update_original_obs)

            kl = torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
            # Handle different output shapes
            if kl.dim() > 1:
                kl = kl.sum(-1, keepdim=True).mean()
            else:
                kl = kl.mean()
            kl = distributed.dist_avg(kl)

            self._logger.store({'Train/KL': kl.item()})
            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                final_steps = i + 1
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': final_steps,
                'Value/Adv': adv_r.mean().item(),
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )
