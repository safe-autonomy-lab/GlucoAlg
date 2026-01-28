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
"""Implementation of ActorBuilder."""

from __future__ import annotations

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_learning_actor_with_feature_extractor import GaussianLearningActorWithFeatureExtractor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.actor.perturbation_actor import PerturbationActor
from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.actor.categorical_actor import CategoricalActor
from omnisafe.models.base import Actor

from omnisafe.typing import Activation, ActorType, InitFunction, OmnisafeSpace
import gymnasium as gym


# pylint: disable-next=too-few-public-methods
class ActorBuilder:
    """Class for building actor networks.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',        
        shield_type: str = 'none',
    ) -> None:
        """Initialize an instance of :class:`ActorBuilder`."""
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes
        self._shield_type: str = shield_type
    # pylint: disable-next=too-many-return-statements
    def build_actor(
        self,
        actor_type: ActorType,
    ) -> Actor:
        """Build actor network.

        Currently, we support the following actor types:
            - ``gaussian_learning``: Gaussian actor with learnable standard deviation parameters.
            - ``gaussian_sac``: Gaussian actor with learnable standard deviation network.
            - ``mlp``: Multi-layer perceptron actor, used in ``DDPG`` and ``TD3``.
            - ``categorical``: Categorical actor for discrete action spaces.

        Args:
            actor_type (ActorType): Type of actor network, e.g. ``gaussian_learning``.

        Returns:
            Actor network, ranging from GaussianLearningActor, GaussianSACActor to MLPActor.

        Raises:
            NotImplementedError: If the actor type is not implemented.
        """

        if actor_type == 'gaussian_learning':
            return GaussianLearningActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                output_activation=self._output_activation,
                init_log_std=self._init_log_std,
            )
        if actor_type == 'gaussian_sac':
            return GaussianSACActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                output_activation=self._output_activation,
            )
        if actor_type == 'mlp':
            return MLPActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                output_activation=self._output_activation,
            )
        if actor_type == 'vae':
            return VAE(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                output_activation=self._output_activation,
            )
        if actor_type == 'perturbation':
            return PerturbationActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                output_activation=self._output_activation,
            )
        if actor_type == 'categorical':
            return CategoricalActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                shield_type=self._shield_type,
            )
        raise NotImplementedError(
            f'Actor type {actor_type} is not implemented! '
            f'Available actor types are: gaussian_learning, gaussian_sac, mlp, vae, perturbation, categorical.',
        )
