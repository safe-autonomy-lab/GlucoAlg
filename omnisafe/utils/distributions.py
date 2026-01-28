# authors: Minjae Kwon
# ==============================================================================
"""Implementation of categorical distributions for discrete action spaces."""

from __future__ import annotations

import torch
from torch.distributions import Categorical, Distribution, constraints
from torch.distributions.kl import register_kl
from typing import List


class MultiCategoricalDistribution(Distribution):
    """Multi-categorical distribution for discrete action spaces.
    
    This distribution handles both single discrete actions and multi-discrete actions
    by using multiple categorical distributions.
    
    Args:
        logits: Tensor of logits for each categorical distribution
        action_dims: List of action dimensions for each categorical distribution
    """
    
    # Define argument constraints as class attribute
    arg_constraints = {}
    
    def __init__(self, logits: torch.Tensor, action_dims: List[int]):
        """Initialize multi-categorical distribution.
        
        Args:
            logits: Tensor of shape (batch_size, sum(action_dims)) containing logits
            action_dims: List of integers specifying the number of actions for each discrete dimension
        """
        # Check for invalid values in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError(f"Logits contain NaN or Inf values: {logits}")
            
        self.action_dims = action_dims
        
        # Split logits and create categorical distributions
        logit_splits = torch.split(logits, action_dims, dim=-1)
        # Defensive logging for debugging action logits; avoid shape assumptions that can crash
        # print("Logit splits bolus", logit_splits[0] if logit_splits[0].dim() == 1 else logit_splits[0][0])
        # print("Logit splits meal", logit_splits[1] if logit_splits[1].dim() == 1 else logit_splits[1][0])
        self.categoricals = [Categorical(logits=logit) for logit in logit_splits]
        
        super().__init__(batch_shape=logits.shape[:-1], event_shape=torch.Size([len(action_dims)]))
        
        # Validate action dimensions
        if sum(action_dims) != logits.shape[-1]:
            raise ValueError(f"Sum of action_dims ({sum(action_dims)}) must equal last dimension of logits ({logits.shape[-1]})")
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the distribution.
        
        Args:
            sample_shape: Shape of samples to draw
            
        Returns:
            Tensor of shape (*sample_shape, *batch_shape, len(action_dims))
        """
        samples = [cat.sample(sample_shape) for cat in self.categoricals]
        return torch.stack(samples, dim=-1)
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sample (same as sample for discrete distributions)."""
        return self.sample(sample_shape)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions.
        
        Args:
            value: Actions of shape (*batch_shape, len(action_dims))
            
        Returns:
            Log probabilities of shape (*batch_shape,)
        """
        if value.shape[-1] != len(self.action_dims):
            raise ValueError(f"Expected {len(self.action_dims)} actions, got {value.shape[-1]}")
        
        log_probs = []
        for i, cat in enumerate(self.categoricals):
            log_probs.append(cat.log_prob(value[..., i]))
        
        return torch.stack(log_probs, dim=-1).sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """Compute entropy of the distribution.
        
        Returns:
            Entropy tensor of shape (*batch_shape,)
        """
        entropies = [cat.entropy() for cat in self.categoricals]
        return torch.stack(entropies, dim=-1).sum(dim=-1)
    
    def mode(self) -> torch.Tensor:
        """Return the mode (most likely action) for each categorical.
        
        Returns:
            Mode actions of shape (*batch_shape, len(action_dims))
        """
        modes = [torch.argmax(cat.probs, dim=-1) for cat in self.categoricals]
        return torch.stack(modes, dim=-1)
    
    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution (same as mode for categorical)."""
        return self.mode().float()
    
    @property
    def probs(self) -> torch.Tensor:
        """Probabilities for all actions."""
        return torch.cat([cat.probs for cat in self.categoricals], dim=-1)
    
    @property
    def logits(self) -> torch.Tensor:
        """Logits for all actions."""
        return torch.cat([cat.logits for cat in self.categoricals], dim=-1)
    
    @property
    def variance(self) -> torch.Tensor:
        """Variance of the distribution.
        
        For discrete distributions, we compute the variance of each categorical
        and sum them. This is used for compatibility with algorithms like FOCOPS
        that expect continuous distribution properties.
        
        Returns:
            Variance tensor of shape (*batch_shape,)
        """
        # For each categorical, variance = sum_i p_i * (i - mean)^2
        variances = []
        for cat in self.categoricals:
            probs = cat.probs  # shape: (*batch_shape, n_actions)
            n_actions = probs.shape[-1]
            # Create action indices: 0, 1, 2, ..., n_actions-1
            indices = torch.arange(n_actions, dtype=probs.dtype, device=probs.device)
            # Mean = sum_i p_i * i
            mean = (probs * indices).sum(dim=-1, keepdim=True)
            # Variance = sum_i p_i * (i - mean)^2
            var = (probs * (indices - mean) ** 2).sum(dim=-1)
            variances.append(var)
        return torch.stack(variances, dim=-1).sum(dim=-1)
    
    @property
    def stddev(self) -> torch.Tensor:
        """Standard deviation of the distribution.
        
        Returns:
            Standard deviation tensor of shape (*batch_shape,)
        """
        return self.variance.sqrt()


class CategoricalDistribution(Distribution):
    """Single categorical distribution wrapper for consistency."""
    
    # Define argument constraints as class attribute
    arg_constraints = {}
    
    def __init__(self, logits: torch.Tensor):
        """Initialize single categorical distribution.
        
        Args:
            logits: Tensor of logits for the categorical distribution
        """
        self.categorical = Categorical(logits=logits)
        super().__init__(batch_shape=self.categorical.batch_shape, event_shape=torch.Size([1]))
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the distribution."""
        return self.categorical.sample(sample_shape).unsqueeze(-1)
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sample (same as sample for discrete)."""
        return self.sample(sample_shape)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        if value.dim() > 1:
            value = value.squeeze(-1)
        return self.categorical.log_prob(value)
    
    def entropy(self) -> torch.Tensor:
        """Compute entropy."""
        return self.categorical.entropy()
    
    def mode(self) -> torch.Tensor:
        """Return the mode."""
        return torch.argmax(self.categorical.probs, dim=-1).unsqueeze(-1)
    
    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution (same as mode)."""
        return self.mode().float()
    
    @property
    def probs(self) -> torch.Tensor:
        """Probabilities."""
        return self.categorical.probs
    
    @property
    def logits(self) -> torch.Tensor:
        """Logits."""
        return self.categorical.logits
    
    @property
    def variance(self) -> torch.Tensor:
        """Variance of the distribution.
        
        For discrete distributions, we compute the variance of the categorical.
        This is used for compatibility with algorithms like FOCOPS.
        
        Returns:
            Variance tensor of shape (*batch_shape,)
        """
        probs = self.categorical.probs  # shape: (*batch_shape, n_actions)
        n_actions = probs.shape[-1]
        # Create action indices: 0, 1, 2, ..., n_actions-1
        indices = torch.arange(n_actions, dtype=probs.dtype, device=probs.device)
        # Mean = sum_i p_i * i
        mean = (probs * indices).sum(dim=-1, keepdim=True)
        # Variance = sum_i p_i * (i - mean)^2
        var = (probs * (indices - mean) ** 2).sum(dim=-1)
        return var
    
    @property
    def stddev(self) -> torch.Tensor:
        """Standard deviation of the distribution.
        
        Returns:
            Standard deviation tensor of shape (*batch_shape,)
        """
        return self.variance.sqrt()


# Register KL divergence implementations
@register_kl(MultiCategoricalDistribution, MultiCategoricalDistribution)
def _kl_multi_categorical_multi_categorical(p: MultiCategoricalDistribution, q: MultiCategoricalDistribution) -> torch.Tensor:
    """Compute KL divergence between two MultiCategoricalDistribution instances.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        KL divergence tensor
    """
    if len(p.action_dims) != len(q.action_dims):
        raise ValueError("Action dimensions must match for KL divergence computation")
    
    # Compute KL divergence for each categorical distribution and sum them
    kl_divs = []
    for p_cat, q_cat in zip(p.categoricals, q.categoricals):
        kl_divs.append(torch.distributions.kl.kl_divergence(p_cat, q_cat))
    
    return torch.stack(kl_divs, dim=-1).sum(dim=-1)


@register_kl(CategoricalDistribution, CategoricalDistribution)
def _kl_categorical_categorical(p: CategoricalDistribution, q: CategoricalDistribution) -> torch.Tensor:
    """Compute KL divergence between two CategoricalDistribution instances.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        KL divergence tensor
    """
    return torch.distributions.kl.kl_divergence(p.categorical, q.categorical)
