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
"""Implementation of the CUP algorithm with discrete action support."""

from typing import Union
import torch
from gymnasium import spaces
from rich.progress import track
from torch.distributions import Normal, Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from omnisafe.utils.distributions import MultiCategoricalDistribution, CategoricalDistribution


@registry.register
class CUP(PPO):
    """The Constrained Update Projection (CUP) Approach to Safe Policy Optimization.

    References:
        - Title: Constrained Update Projection Approach to Safe Policy Optimization
        - Authors: Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li,
            Yaodong Yang, Gang Pan.
        - URL: `CUP <https://arxiv.org/abs/2209.07089>`_
    
    Note: Extended to support discrete action spaces (Discrete/MultiDiscrete).
    """

    _p_dist: Union[Normal, MultiCategoricalDistribution, CategoricalDistribution]
    _is_discrete: bool = False
    _action_dims: list = None

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
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
        """Log the CUP specific information.

        +-----------------------------+----------------------------------------------------------+
        | Things to log               | Description                                              |
        +=============================+==========================================================+
        | Metrics/LagrangeMultiplier  | The Lagrange multiplier.                                 |
        +-----------------------------+----------------------------------------------------------+
        | Loss/Loss_pi_c              | The loss of the cost performance.                        |
        +-----------------------------+----------------------------------------------------------+
        | Train/SecondStepStopIter    | The number of iterations to stop the second step.        |
        +-----------------------------+----------------------------------------------------------+
        | Train/SecondStepEntropy     | The entropy of the current policy.                       |
        +-----------------------------+----------------------------------------------------------+
        | Train/SecondStepPolicyRatio | The ratio between the current policy and the old policy. |
        +-----------------------------+----------------------------------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Loss/Loss_pi_c', delta=True)
        self._logger.register_key('Train/SecondStepStopIter')
        self._logger.register_key('Train/SecondStepEntropy')
        self._logger.register_key('Train/SecondStepPolicyRatio', min_and_max=True)

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
        original_obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the performance of cost on this moment.

        We compute the KL divergence between the current policy and the old policy, the entropy of
        the current policy, and the ratio between the current policy and the old policy.

        The loss of the cost performance is defined as:

        .. math::

            L = \underset{a \sim \pi_{\theta}}{\mathbb{E}} [
                \lambda \frac{1 - \gamma \nu}{1 - \gamma}
                    \frac{\pi_{\theta}^{'} (a|s)}{\pi_{\theta} (a|s)} A^{C}_{\pi_{\theta}}
                + KL (\pi_{\theta}^{'} (a|s) || \pi_{\theta} (a|s))
            ]

        where :math:`\lambda` is the Lagrange multiplier, :math:`\frac{1 - \gamma \nu}{1 - \gamma}`
        is the coefficient value, :math:`\pi_{\theta}^{'} (a_t|s_t)` is the current policy,
        :math:`\pi_{\theta} (a_t|s_t)` is the old policy, :math:`A^{C}_{\pi_{\theta}}` is the cost
        advantage, :math:`KL (\pi_{\theta}^{'} (a_t|s_t) || \pi_{\theta} (a_t|s_t))` is the KL
        divergence between the current policy and the old policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The loss of the cost performance.
        """
        distribution = self._actor_critic.actor(obs, original_obs=original_obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)

        # Compute KL divergence - handle both discrete and continuous
        kl = torch.distributions.kl_divergence(distribution, self._p_dist)
        # Handle different output shapes
        if kl.dim() > 1:
            kl = kl.sum(-1, keepdim=True)
        elif kl.dim() == 1:
            kl = kl.unsqueeze(-1)

        coef = (1 - self._cfgs.algo_cfgs.gamma * self._cfgs.algo_cfgs.lam) / (
            1 - self._cfgs.algo_cfgs.gamma
        )
        loss = (self._lagrange.lagrangian_multiplier * coef * ratio * adv_c + kl).mean()

        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Loss/Loss_pi_c': loss.item(),
                'Train/SecondStepEntropy': entropy,
                'Train/SecondStepPolicyRatio': ratio,
            },
        )
        return loss

    def _update(self) -> None:
        r"""Update actor, critic, and Lagrange multiplier parameters.

        In CUP, the Lagrange multiplier is updated as the naive lagrange multiplier update.

        Then in each iteration of the policy update, CUP calculates current policy's distribution,
        which used to calculate the policy loss.
        
        For discrete action spaces, we store logits instead of mean/std.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)

        super()._update()

        data = self._buf.get()
        obs, act, logp, adv_c, original_obs = (
            data['obs'],
            data['act'],
            data['logp'],
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
                dataset=TensorDataset(obs, act, logp, adv_c, old_logits, original_obs),
                batch_size=self._cfgs.algo_cfgs.batch_size,
                shuffle=True,
            )
        else:
            dataloader = DataLoader(
                dataset=TensorDataset(obs, act, logp, adv_c, old_mean, old_std),
                batch_size=self._cfgs.algo_cfgs.batch_size,
                shuffle=True,
            )

        final_steps = self._cfgs.algo_cfgs.update_iters
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            if self._is_discrete:
                # Discrete action space iteration
                for obs, act, logp, adv_c, old_logits_batch, original_obs in dataloader:
                    # Reconstruct old distribution from stored logits
                    if len(self._action_dims) > 1:
                        self._p_dist = MultiCategoricalDistribution(old_logits_batch, self._action_dims)
                    else:
                        self._p_dist = CategoricalDistribution(old_logits_batch)
                    
                    loss_cost = self._loss_pi_cost(obs, act, logp, adv_c, original_obs=original_obs)
                    self._actor_critic.actor_optimizer.zero_grad()
                    loss_cost.backward()
                    if self._cfgs.algo_cfgs.max_grad_norm is not None:
                        clip_grad_norm_(
                            self._actor_critic.actor.parameters(),
                            self._cfgs.algo_cfgs.max_grad_norm,
                        )
                    distributed.avg_grads(self._actor_critic.actor)
                    self._actor_critic.actor_optimizer.step()
            else:
                # Continuous action space iteration (original logic)
                for obs, act, logp, adv_c, old_mean, old_std in dataloader:
                    self._p_dist = Normal(old_mean, old_std)
                    loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
                    self._actor_critic.actor_optimizer.zero_grad()
                    loss_cost.backward()
                    if self._cfgs.algo_cfgs.max_grad_norm is not None:
                        clip_grad_norm_(
                            self._actor_critic.actor.parameters(),
                            self._cfgs.algo_cfgs.max_grad_norm,
                        )
                    distributed.avg_grads(self._actor_critic.actor)
                    self._actor_critic.actor_optimizer.step()

            new_distribution = self._actor_critic.actor(before_update_obs, original_obs=before_update_original_obs)

            kl = torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
            # Handle different output shapes
            if kl.dim() > 1:
                kl = kl.sum(-1, keepdim=True).mean()
            else:
                kl = kl.mean()
            kl = distributed.dist_avg(kl)

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                final_steps = i + 1
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.item(),
                'Train/SecondStepStopIter': final_steps,  # pylint: disable=undefined-loop-variable
            },
        )
