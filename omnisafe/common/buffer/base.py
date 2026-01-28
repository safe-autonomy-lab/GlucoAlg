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
"""Abstract base class for buffer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from gymnasium.spaces import Box
import gymnasium as gym
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class BaseBuffer(ABC):
    r"""Abstract base class for buffer.

    .. warning::
        The buffer supports Box observation spaces and Box/Discrete/MultiDiscrete action spaces.

    In base buffer, we store the following data:

    +--------+---------------------------+------------------+-----------------------------------+
    | Name   | Shape                     | Dtype            | Description                       |
    +========+===========================+==================+===================================+
    | obs    | (size, \*obs_space.shape) | torch.float32    | The observation from environment. |
    +--------+---------------------------+------------------+-----------------------------------+
    | act    | (size, \*act_space.shape) | torch.float32    | The action from agent (Box).      |
    |        |                           | or torch.long    | The action from agent (Discrete). |
    +--------+---------------------------+------------------+-----------------------------------+
    | reward | (size,)                   | torch.float32    | Single step reward.               |
    +--------+---------------------------+------------------+-----------------------------------+
    | cost   | (size,)                   | torch.float32    | Single step cost.                 |
    +--------+---------------------------+------------------+-----------------------------------+
    | done   | (size,)                   | torch.float32    | Whether the episode is done.      |
    +--------+---------------------------+------------------+-----------------------------------+


    Args:
        obs_space (OmnisafeSpace): The observation space (must be Box).
        act_space (OmnisafeSpace): The action space (Box, Discrete, or MultiDiscrete).
        size (int): The size of the buffer.
        device (torch.device): The device of the buffer. Defaults to ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space is not Box.
        NotImplementedError: If the action space is not Box, Discrete, or MultiDiscrete.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`BaseBuffer`."""
        self._device: torch.device = device
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros((size, *obs_space.shape), dtype=torch.float32, device=device)
            original_obs_buf = torch.zeros((size, *obs_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError
        if isinstance(act_space, Box):
            act_buf = torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device)
        elif isinstance(act_space, gym.spaces.Discrete):
            # Single discrete action - store as integers
            act_buf = torch.zeros((size, 1), dtype=torch.long, device=device)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            # Multi-discrete actions - store as integers with shape (size, num_discrete_dims)
            act_buf = torch.zeros((size, len(act_space.nvec)), dtype=torch.long, device=device)
        else:
            raise NotImplementedError(f"Action space type {type(act_space)} not supported")

        self.data: dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
            'original_obs': original_obs_buf,
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            'cost': torch.zeros(size, dtype=torch.float32, device=device),
            'done': torch.zeros(size, dtype=torch.float32, device=device),
        }
        self._size: int = size

    @property
    def device(self) -> torch.device:
        """The device of the buffer."""
        return self._device

    @property
    def size(self) -> int:
        """The size of the buffer."""
        return self._size

    def __len__(self) -> int:
        """Return the length of the buffer."""
        return self._size

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros((self._size, *shape), dtype=dtype, device=self._device)

    @abstractmethod
    def store(self, **data: torch.Tensor) -> None:
        """Store a transition in the buffer.

        .. warning::
            This is an abstract method.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.store(obs=obs, act=act, reward=reward, cost=cost, done=done)

        Args:
            data (torch.Tensor): The data to store.
        """


class BaseDictBuffer(ABC):
    r"""Abstract base class for buffer supporting dictionary observation spaces.

    In this buffer, we store the following data:
    - 'obs': A dictionary of observation tensors, each with shape (size, *obs_shape[key]).
    - 'act': Action tensor with shape (size, *act_space.shape).
    - 'reward': Reward tensor with shape (size,).
    - 'cost': Cost tensor with shape (size,).
    - 'done': Done tensor with shape (size,).

    Args:
        obs_space (OmnisafeSpace): The observation space (Dict with Box and MultiBinary subspaces).
        act_space (OmnisafeSpace): The action space (Box, Discrete, or MultiDiscrete).
        size (int): The size of the buffer.
        device (torch.device): The device of the buffer. Defaults to CPU.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer, including nested observation dictionary.

    Raises:
        NotImplementedError: If the observation space is not a Dict or action space is not supported.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`BaseBuffer`."""
        self._device: torch.device = device
        self._size: int = size

        # Validate and initialize observation space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise NotImplementedError("Observation space must be a Dict")
        
        obs_buf = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                obs_buf[key] = torch.zeros((size, *space.shape), dtype=torch.float32, device=device)
            elif isinstance(space, gym.spaces.MultiBinary):
                obs_buf[key] = torch.zeros((size, *space.shape), dtype=torch.float32, device=device)
            else:
                raise NotImplementedError(f"Unsupported observation subspace type for '{key}': {type(space)}")

        # Validate and initialize action space
        if isinstance(act_space, gym.spaces.Box):
            act_buf = torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device)
        elif isinstance(act_space, gym.spaces.Discrete):
            act_buf = torch.zeros((size, 1), dtype=torch.long, device=device)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_buf = torch.zeros((size, len(act_space.nvec)), dtype=torch.long, device=device)
        else:
            raise NotImplementedError(f"Action space type {type(act_space)} not supported")

        # Initialize buffer data
        self.data: Dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            'cost': torch.zeros(size, dtype=torch.float32, device=device),
            'done': torch.zeros(size, dtype=torch.float32, device=device),
        }

    @property
    def device(self) -> torch.device:
        """The device of the buffer."""
        return self._device

    @property
    def size(self) -> int:
        """The size of the buffer."""
        return self._size

    def __len__(self) -> int:
        """Return the length of the buffer."""
        return self._size

    def add_field(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer outside of observations.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros((self._size, *shape), dtype=dtype, device=self._device)

    @abstractmethod
    def store(self, **data: torch.Tensor) -> None:
        """Store a transition in the buffer.

        .. warning::
            This is an abstract method.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.store(obs={'robot_node': obs1, 'ped_pos': obs2}, act=act, reward=reward, cost=cost, done=done)

        Args:
            data (torch.Tensor): The data to store, with 'obs' as a dictionary of tensors.
        """

