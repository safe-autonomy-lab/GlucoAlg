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
"""Implementation of Discrete Q Critic for discrete action spaces."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
import logging

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network

logger = logging.getLogger(__name__)


class DiscreteQCritic(Critic):
    """Implementation of Q Critic for discrete action spaces.
    
    This Q-function approximator is designed specifically for discrete action spaces.
    Instead of concatenating continuous action values with observations, it either:
    1. Converts discrete action indices to one-hot encodings, or  
    2. Uses separate Q-values for each discrete action (like DQN)
    
    The critic supports both single discrete actions (Discrete space) and 
    multi-discrete actions (MultiDiscrete space).
    
    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space (must be Discrete or MultiDiscrete).
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. 
            Defaults to ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_one_hot (bool, optional): Whether to use one-hot encoding for actions.
            If False, uses separate Q-values for each action. Defaults to True.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_one_hot: bool = True,
    ) -> None:
        """Initialize an instance of :class:`DiscreteQCritic`."""
        # Initialize base critic but skip parent's action space validation
        nn.Module.__init__(self)
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes
        self._num_critics: int = num_critics
        self._use_one_hot: bool = use_one_hot
        
        # Handle observation space
        if isinstance(self._obs_space, spaces.Box) and len(self._obs_space.shape) == 1:
            self._obs_dim: int = self._obs_space.shape[0]
        elif isinstance(self._obs_space, spaces.Dict):
            self._obs_dim: int = 1000  # Default feature dim
        else:
            raise NotImplementedError(f"Observation space {type(self._obs_space)} not supported")
        
        # Handle discrete action space
        if isinstance(self._act_space, spaces.Discrete):
            self._action_dims = [self._act_space.n]
            self._total_action_dim = self._act_space.n
            self._is_multi_discrete = False
        elif isinstance(self._act_space, spaces.MultiDiscrete):
            self._action_dims = self._act_space.nvec.tolist()
            self._total_action_dim = sum(self._action_dims)
            self._is_multi_discrete = True
        else:
            raise NotImplementedError(f"Action space {type(self._act_space)} not supported for DiscreteQCritic")
        
        logger.info(f"DiscreteQCritic: action_dims={self._action_dims}, use_one_hot={use_one_hot}")
        
        # Build critic networks
        self.net_lst: list[nn.Sequential] = []
        for idx in range(self._num_critics):
            if self._use_one_hot:
                # Use one-hot encoding approach: concatenate obs with one-hot action
                net = build_mlp_network(
                    [self._obs_dim + self._total_action_dim, *hidden_sizes, 1],
                    activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
            else:
                # Use separate Q-values approach: output Q-value for each action
                net = build_mlp_network(
                    [self._obs_dim, *hidden_sizes, self._total_action_dim],
                    activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
            
            critic = nn.Sequential(net)
            self.net_lst.append(critic)
            self.add_module(f'critic_{idx}', critic)

    def _action_to_one_hot(self, act: torch.Tensor) -> torch.Tensor:
        """Convert discrete action indices to one-hot encoding.
        
        Args:
            act (torch.Tensor): Discrete action indices, shape (batch_size, action_dims)
            
        Returns:
            torch.Tensor: One-hot encoded actions, shape (batch_size, total_action_dim)
        """
        batch_size = act.shape[0]
        device = act.device
        
        if self._is_multi_discrete:
            # Handle multi-discrete actions
            one_hot_list = []
            for i, (action_dim, n_actions) in enumerate(zip(act.unbind(-1), self._action_dims)):
                one_hot = F.one_hot(action_dim.long(), num_classes=n_actions).float()
                one_hot_list.append(one_hot)
            return torch.cat(one_hot_list, dim=-1)
        else:
            # Handle single discrete actions
            if act.dim() == 1:
                # Convert 1D tensor to 2D: (batch_size,) -> (batch_size, 1)
                act_indices = act.long()
            else:
                # Already 2D, squeeze last dimension if needed
                act_indices = act.squeeze(-1).long()
            
            # Create one-hot encoding, ensuring output is 2D
            one_hot = F.one_hot(act_indices, num_classes=self._total_action_dim).float()
            return one_hot

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward function.
        
        Args:
            obs (torch.Tensor): Observation from environments.
            act (torch.Tensor): Discrete action indices from actor.
            
        Returns:
            A list of Q critic values of action and observation pair.
        """
        res = []
        
        for critic in self.net_lst:
            if self._use_one_hot:
                # Convert discrete actions to one-hot and concatenate with observations
                act_one_hot = self._action_to_one_hot(act)
                critic_input = torch.cat([obs, act_one_hot], dim=-1)
                q_value = torch.squeeze(critic(critic_input), -1)
            else:
                # Get Q-values for all actions, then select the ones corresponding to taken actions
                all_q_values = critic(obs)  # Shape: (batch_size, total_action_dim)
                
                if self._is_multi_discrete:
                    # For multi-discrete, we need to sum Q-values across action dimensions
                    # This is a simplification - in practice you might want different handling
                    q_value = torch.zeros(obs.shape[0], device=obs.device)
                    start_idx = 0
                    for i, (action_indices, n_actions) in enumerate(zip(act.unbind(-1), self._action_dims)):
                        q_vals_for_dim = all_q_values[:, start_idx:start_idx + n_actions]
                        selected_q = q_vals_for_dim.gather(1, action_indices.long().unsqueeze(-1)).squeeze(-1)
                        q_value += selected_q
                        start_idx += n_actions
                else:
                    # For single discrete, gather Q-values for selected actions
                    if act.dim() == 1:
                        act_indices = act.long()
                    else:
                        act_indices = act.squeeze(-1).long()
                    q_value = all_q_values.gather(1, act_indices.unsqueeze(-1)).squeeze(-1)
            
            res.append(q_value)
        
        return res
        
    def get_all_q_values(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Get Q-values for all possible actions (useful for action selection).
        
        Args:
            obs (torch.Tensor): Observation from environments.
            
        Returns:
            A list of Q-values for all actions, shape (batch_size, total_action_dim)
        """
        if self._use_one_hot:
            # For one-hot approach, we need to evaluate all possible actions
            batch_size = obs.shape[0]
            device = obs.device
            
            # Create all possible action indices
            if self._is_multi_discrete:
                # This is complex for multi-discrete - simplified version
                all_actions = torch.arange(self._total_action_dim, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                all_actions = torch.arange(self._total_action_dim, device=device).unsqueeze(0).expand(batch_size, -1)
            
            res = []
            for critic in self.net_lst:
                q_values = torch.zeros(batch_size, self._total_action_dim, device=device)
                for action_idx in range(self._total_action_dim):
                    action_tensor = torch.full((batch_size,), action_idx, device=device, dtype=torch.long)
                    act_one_hot = self._action_to_one_hot(action_tensor)
                    
                    # Debug logging - use print for immediate visibility
                    if action_idx == 0:  # Only print for first action to avoid spam
                        print(f"DEBUG: obs shape: {obs.shape}, act_one_hot shape: {act_one_hot.shape}")
                        print(f"DEBUG: obs dims: {obs.dim()}, act_one_hot dims: {act_one_hot.dim()}")
                    
                    # Ensure both tensors have the same number of dimensions
                    if obs.dim() != act_one_hot.dim():
                        if obs.dim() == 2 and act_one_hot.dim() == 1:
                            # Expand act_one_hot to match obs dimensions
                            act_one_hot = act_one_hot.unsqueeze(0).expand(batch_size, -1)
                        elif obs.dim() == 1 and act_one_hot.dim() == 2:
                            # Expand obs to match act_one_hot dimensions
                            obs = obs.unsqueeze(0)
                    
                    critic_input = torch.cat([obs, act_one_hot], dim=-1)
                    q_values[:, action_idx] = torch.squeeze(critic(critic_input), -1)
                res.append(q_values)
            return res
        else:
            # For separate Q-values approach, directly return network outputs
            res = []
            for critic in self.net_lst:
                res.append(critic(obs))
            return res

    @property
    def action_dims(self) -> list[int]:
        """Get action dimensions."""
        return self._action_dims
    
    @property
    def is_multi_discrete(self) -> bool:
        """Check if action space is multi-discrete."""
        return self._is_multi_discrete
    
    @property
    def use_one_hot(self) -> bool:
        """Check if using one-hot encoding."""
        return self._use_one_hot
