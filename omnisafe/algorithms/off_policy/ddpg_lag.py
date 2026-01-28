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
"""Implementation of the Lagrangian version of Deep Deterministic Policy Gradient algorithm."""


import torch
import torch.nn.functional as F
from gymnasium import spaces
import logging

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.common.lagrange import Lagrange

logger = logging.getLogger(__name__)


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class DDPGLag(DDPG):
    """The Lagrangian version of Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
            Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the DDPGLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _update_actor(self, obs: torch.Tensor) -> None:
        """Update actor with Lagrangian constraints.
        
        Override the parent method to use discrete-specific updates when needed.
        
        Args:
            obs (torch.Tensor): The observation sampled from buffer.
        """
        # Check if we have discrete action space and use appropriate update method
        if isinstance(self._env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            self._update_actor_discrete(obs)
        else:
            # Use the original continuous action update from parent DDPG
            loss = self._loss_pi(obs)
            self._actor_critic.actor_optimizer.zero_grad()
            loss.backward()
            if self._cfgs.algo_cfgs.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self._actor_critic.actor.parameters(),
                    self._cfgs.algo_cfgs.max_grad_norm,
                )
            self._actor_critic.actor_optimizer.step()
            self._logger.store(
                {
                    'Loss/Loss_pi': loss.mean().item(),
                },
            )

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

    def _update_actor_discrete(self, obs: torch.Tensor) -> None:
        """Update actor for discrete action spaces using Q-learning style updates with Lagrangian constraints.
        
        This combines the discrete action update from DDPG with the Lagrangian constraint handling.
        
        Args:
            obs (torch.Tensor): The observation sampled from buffer.
        """
        # Get current policy distribution
        policy_dist = self._actor_critic.actor(obs)
        
        # Get Q-values for all possible actions from both reward and cost critics
        if hasattr(self._actor_critic.reward_critic, 'get_all_q_values'):
            all_q_values_r = self._actor_critic.reward_critic.get_all_q_values(obs)[0]
            all_q_values_c = self._actor_critic.cost_critic.get_all_q_values(obs)[0]
        else:
            # Fallback: evaluate all possible actions manually
            all_q_values_r = self._get_all_q_values_manual(obs, critic_type='reward')
            all_q_values_c = self._get_all_q_values_manual(obs, critic_type='cost')
        
        # Combine reward and cost Q-values using Lagrangian multiplier
        lagrange_mult = self._lagrange.lagrangian_multiplier.item()
        combined_q_values = all_q_values_r - lagrange_mult * all_q_values_c
        
        # Create target action probabilities from combined Q-values using softmax
        temperature = getattr(self._cfgs.algo_cfgs, 'policy_temperature', 1.0)
        target_probs = F.softmax(combined_q_values / temperature, dim=-1)
        
        # Compute cross-entropy loss between current policy and target probabilities
        if hasattr(policy_dist, 'logits'):
            log_probs = F.log_softmax(policy_dist.logits, dim=-1)
        else:
            # Get log probabilities for all actions
            if isinstance(self._env.action_space, spaces.Discrete):
                n_actions = self._env.action_space.n
                all_actions = torch.arange(n_actions, device=obs.device).unsqueeze(0).expand(obs.shape[0], -1)
                log_probs = policy_dist.log_prob(all_actions)
            else:
                raise NotImplementedError("Multi-discrete action spaces not fully implemented yet")
        
        # Cross-entropy loss: -sum(target_probs * log_probs)
        loss = -torch.sum(target_probs * log_probs, dim=-1).mean()
        
        # Add entropy regularization
        entropy_coeff = getattr(self._cfgs.algo_cfgs, 'entropy_coeff', 0.01)
        if hasattr(policy_dist, 'entropy'):
            entropy_loss = -entropy_coeff * policy_dist.entropy().mean()
            loss += entropy_loss
        
        # Normalize by Lagrangian factor (similar to continuous case)
        loss = loss / (1 + lagrange_mult)
        
        # Update actor
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        
        if self._cfgs.algo_cfgs.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        
        self._actor_critic.actor_optimizer.step()
        
        # Log the loss
        self._logger.store({'Loss/Loss_pi': loss.item()})

    def _get_all_q_values_manual(self, obs: torch.Tensor, critic_type: str = 'reward') -> torch.Tensor:
        """Manually compute Q-values for all possible actions for a specific critic.
        
        Args:
            obs (torch.Tensor): Observations
            critic_type (str): Either 'reward' or 'cost'
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Select the appropriate critic
        if critic_type == 'reward':
            critic = self._actor_critic.reward_critic
        elif critic_type == 'cost':
            critic = self._actor_critic.cost_critic
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}")
        
        if isinstance(self._env.action_space, spaces.Discrete):
            n_actions = self._env.action_space.n
            all_q_values = torch.zeros(batch_size, n_actions, device=device)
            
            for action_idx in range(n_actions):
                action_tensor = torch.full((batch_size,), action_idx, device=device, dtype=torch.long)
                q_value = critic(obs, action_tensor)[0]
                all_q_values[:, action_idx] = q_value
        else:
            raise NotImplementedError("Multi-discrete action spaces not fully implemented yet")
        
        return all_q_values

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in DDPGLag is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        # Check if we have discrete action space and use appropriate loss calculation
        if isinstance(self._env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            return self._loss_pi_discrete(obs)
        else:
            # Original continuous action loss
            action = self._actor_critic.actor.predict(obs, deterministic=True)
            loss_r = -self._actor_critic.reward_critic(obs, action)[0]
            loss_c = (
                self._lagrange.lagrangian_multiplier.item()
                * self._actor_critic.cost_critic(obs, action)[0]
            )
            return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())

    def _loss_pi_discrete(self, obs: torch.Tensor) -> torch.Tensor:
        """Computing policy loss for discrete actions with Lagrangian constraints.
        
        Args:
            obs (torch.Tensor): The observation sampled from buffer.
            
        Returns:
            The loss of pi/actor.
        """
        # Get current policy distribution
        policy_dist = self._actor_critic.actor(obs)
        
        # Get Q-values for all possible actions
        if hasattr(self._actor_critic.reward_critic, 'get_all_q_values'):
            all_q_values_r = self._actor_critic.reward_critic.get_all_q_values(obs)[0]
            all_q_values_c = self._actor_critic.cost_critic.get_all_q_values(obs)[0]
        else:
            all_q_values_r = self._get_all_q_values_manual(obs, 'reward')
            all_q_values_c = self._get_all_q_values_manual(obs, 'cost')
        
        # Compute expected Q-values under current policy
        if hasattr(policy_dist, 'probs'):
            action_probs = policy_dist.probs
        else:
            action_probs = F.softmax(policy_dist.logits, dim=-1)
        
        expected_q_r = torch.sum(action_probs * all_q_values_r, dim=-1)
        expected_q_c = torch.sum(action_probs * all_q_values_c, dim=-1)
        
        # Combine with Lagrangian multiplier
        lagrange_mult = self._lagrange.lagrangian_multiplier.item()
        combined_loss = -expected_q_r + lagrange_mult * expected_q_c
        
        return combined_loss.mean() / (1 + lagrange_mult)

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )
