# authors: Minjae Kwon
# ==============================================================================
"""Implementation of CategoricalActor for discrete action spaces."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Distribution
from gymnasium import spaces
from typing import Union, List

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network
from omnisafe.utils.distributions import MultiCategoricalDistribution, CategoricalDistribution
from shield.predictive_shield import Shield


class CategoricalActor(Actor):
    """Implementation of CategoricalActor for discrete action spaces.
    
    This actor handles both single discrete actions (Discrete space) and 
    multi-discrete actions (MultiDiscrete space) using categorical distributions.
    
    Args:
        obs_space: Observation space
        act_space: Action space (must be Discrete or MultiDiscrete)
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        weight_initialization_mode: Weight initialization mode
    """
    
    _current_dist: Union[CategoricalDistribution, MultiCategoricalDistribution]
    
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        shield_type: str = 'none',
    ) -> None:
        """Initialize CategoricalActor."""
        # Override parent's action space validation
        nn.Module.__init__(self)
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: List[int] = hidden_sizes
        self._after_inference: bool = False
        self._shield_type = shield_type
        if self._shield_type not in ['child', 'adult', 'adolescent', 'none']:
            raise ValueError("Invalid shield type. Must be one of: child, adult, adolescent, none")

        if self._shield_type != 'none':
            self.shield = Shield(shield_type=self._shield_type)
        
        else:
            print("No shield is used")
            self.shield = None
        
        # Handle observation space
        if isinstance(self._obs_space, spaces.Box) and len(self._obs_space.shape) == 1:
            self._obs_dim: int = self._obs_space.shape[0]
        elif isinstance(self._obs_space, spaces.Dict):
            self._obs_dim: int = 1000  # Default feature dim
        else:
            raise NotImplementedError(f"Observation space {type(self._obs_space)} not supported")
        
        # Handle action space - support both Discrete and MultiDiscrete
        if isinstance(self._act_space, spaces.Discrete):
            self._action_dims = [self._act_space.n]
            self._total_action_dim = self._act_space.n
            self._is_multi_discrete = False
        elif isinstance(self._act_space, spaces.MultiDiscrete):
            self._action_dims = self._act_space.nvec.tolist()
            self._total_action_dim = sum(self._action_dims)
            self._is_multi_discrete = True
        else:
            raise NotImplementedError(f"Action space {type(self._act_space)} not supported for CategoricalActor")
        
        # Build the network that outputs logits for all actions
        self.logits_net: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._total_action_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='identity',  # No activation on logits
        )
    
    def _distribution(self, obs: torch.Tensor, original_obs: torch.Tensor | None = None) -> Union[CategoricalDistribution, MultiCategoricalDistribution]:
        """Get the distribution of the actor.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Categorical distribution(s) for the action space
        """
        logits = self.logits_net(obs)
        
        # ======================================================================
        # DIFFERENTIABLE LTL SHIELD (Soft Logic)
        # Specific to Diabetes Environment (Obs Dim = 12)
        # Enforces temporal spacing + daily cap constraints on policy logits
        # Activated by self._use_shield flag
        # ======================================================================
        source_obs = original_obs if original_obs is not None else obs
        logits = self.shield.apply(source_obs, logits, self._action_dims) if self.shield is not None else logits

        if self._is_multi_discrete:
            return MultiCategoricalDistribution(logits, self._action_dims)
        else:
            return CategoricalDistribution(logits)
    
    def forward(self, obs: torch.Tensor, original_obs: torch.Tensor=None) -> Distribution:
        """Forward pass returning distribution.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Action distribution
        """
        self._current_dist = self._distribution(obs, original_obs)
        self._after_inference = True
        return self._current_dist
    
    def predict(self, obs: torch.Tensor, deterministic: bool = False, original_obs: torch.Tensor = None) -> torch.Tensor:
        """Predict action given observation.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mode; if False, sample from distribution
            
        Returns:
            Action tensor
        """
        self._current_dist = self._distribution(obs, original_obs)
        self._after_inference = True
        
        if deterministic:
            action = self._current_dist.mode()
        else:
            action = self._current_dist.sample()
        
        # Ensure proper shape for single discrete actions
        if not self._is_multi_discrete:
            action = action.squeeze(-1)
            
        return action
    
    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions.
        
        Args:
            act: Action tensor
            
        Returns:
            Log probability tensor
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        
        # Handle single discrete actions
        if not self._is_multi_discrete and act.dim() == 1:
            act = act.unsqueeze(-1)
            
        return self._current_dist.log_prob(act)
    
    def sample(self, obs: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Sample multiple actions given observation.
        
        Args:
            obs: Observation tensor
            n_samples: Number of samples to draw
            
        Returns:
            Sampled actions tensor
        """
        self._current_dist = self._distribution(obs, obs)
        self._after_inference = True
        
        samples = self._current_dist.sample(torch.Size([n_samples]))
        
        # Handle single discrete actions
        if not self._is_multi_discrete:
            samples = samples.squeeze(-1)
            
        return samples
    
    @property
    def action_dims(self) -> List[int]:
        """Get action dimensions."""
        return self._action_dims
    
    @property
    def is_multi_discrete(self) -> bool:
        """Check if action space is multi-discrete."""
        return self._is_multi_discrete
    
    @property
    def std(self) -> float:
        """Standard deviation property for compatibility with continuous actors.
        
        For discrete actions, this returns a dummy value since std doesn't apply.
        This is for compatibility with algorithms that expect this property.
        
        Returns:
            Always returns 0.0 for discrete actions.
        """
        return 0.0
