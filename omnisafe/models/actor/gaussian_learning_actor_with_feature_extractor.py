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
"""Implementation of GaussianLearningActor."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from omnisafe.models.actor.gaussian_actor import GaussianActor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network
from omnisafe.utils.feature_extractor import FeatureExtractor


# pylint: disable-next=too-many-instance-attributes
class GaussianLearningActorWithFeatureExtractor(GaussianActor):
    """Implementation of GaussianLearningActor.

    GaussianLearningActor is a Gaussian actor with a learnable standard deviation. It is used in
    on-policy algorithms such as ``PPO``, ``TRPO`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    _current_dist: Normal

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        feature_dim: int = 1000,
    ) -> None:
        """Initialize an instance of :class:`GaussianLearningActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode, feature_dim)

        hidden_dim = hidden_sizes[0]
        self.feature_extractor = FeatureExtractor(hidden_dim=hidden_dim)
        self.mean: nn.Module = build_mlp_network(
            sizes=[hidden_dim, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(self._act_dim), requires_grad=True)

    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(obs)
        return features
    
    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        features = self._extract_features(obs)
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()
    
    def predict_with_multiple_samples(self, obs: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Predict the action given observation with multiple samples.

        Args:
            obs (torch.Tensor): Observation from environments.
            n_samples (int, optional): Number of samples. Defaults to 10.

        Returns:
            The action sampled from the distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist.rsample((n_samples,))
    
        
    def predict_with_multiple_samples_with_seed(self, obs: torch.Tensor, n_samples: int = 10, seed: int = 0) -> torch.Tensor:
        """Predict the action given observation with multiple samples.

        Args:
            obs (torch.Tensor): Observation from environments.
            n_samples (int, optional): Number of samples. Defaults to 10.

        Returns:
            The action sampled from the distribution.
        """
        self._current_dist = self._distribution(obs)
        generator = torch.Generator(device=self.log_std.device)
        generator.manual_seed(seed)
        self._after_inference = True
        return self._current_dist.rsample((n_samples,), generator=generator)

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return self._current_dist.log_prob(act).sum(axis=-1)

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return torch.exp(self.log_std).mean().item()

    @std.setter
    def std(self, std: float) -> None:
        device = self.log_std.device
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=device)))
