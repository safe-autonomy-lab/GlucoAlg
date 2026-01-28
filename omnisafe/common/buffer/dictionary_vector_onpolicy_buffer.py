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
"""Implementation of VectorOnPolicyBuffer."""

from __future__ import annotations

import torch

from omnisafe.common.buffer.dictionary_onpolicy_buffer import OnPolicyDictBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed

import torch
from gymnasium import spaces
from typing import Dict, Optional

# Placeholder for OmnisafeSpace (assumed to be a Gymnasium space)
OmnisafeSpace = spaces.Space
DEVICE_CPU = torch.device('cpu')

# Placeholder for AdvantageEstimator (assumed to be a string-based enum)
class AdvantageEstimator:
    GAE = 'gae'
    GAE_RTG = 'gae-rtg'
    VTRACE = 'vtrace'
    PLAIN = 'plain'

# Placeholder for distributed statistics (mock implementation)
class distributed:
    @staticmethod
    def dist_statistics_scalar(x: torch.Tensor) -> tuple[float, float]:
        return x.mean().item(), x.std().item() if x.std() > 0 else 1.0

class VectorOnPolicyDictBuffer(OnPolicyDictBuffer):
    """Vectorized on-policy buffer for dictionary observation spaces.

    This buffer manages multiple OnPolicyBuffer instances, each corresponding to one environment.
    It supports dictionary observation spaces and integrates with a modified OnPolicyBuffer.

    Args:
        obs_space (OmnisafeSpace): Observation space (expected to be a Dict space).
        act_space (OmnisafeSpace): Action space (expected to be Box).
        size (int): Size of each buffer.
        gamma (float): Discount factor.
        lam (float): Lambda for GAE.
        lam_c (float): Lambda for GAE for cost.
        advantage_estimator (str): Advantage estimator method (e.g., 'gae', 'vtrace').
        penalty_coefficient (float): Penalty coefficient for cost.
        standardized_adv_r (bool): Whether to standardize the advantage for reward.
        standardized_adv_c (bool): Whether to standardize the advantage for cost.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to CPU.

    Attributes:
        buffers (list[OnPolicyBuffer]): List of on-policy buffers, one per environment.
        _num_buffers (int): Number of environments/buffers.
        _standardized_adv_r (bool): Flag for reward advantage standardization.
        _standardized_adv_c (bool): Flag for cost advantage standardization.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: str,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of VectorOnPolicyBuffer."""
        self._num_buffers: int = num_envs
        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c

        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')

        # Initialize a list of OnPolicyBuffer instances, each for one environment
        self.buffers = [
            OnPolicyDictBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            ) for _ in range(num_envs)
        ]

    @property
    def num_buffers(self) -> int:
        """Number of buffers."""
        return self._num_buffers

    def store(self, **data: torch.Tensor) -> None:
        """Store vectorized data into the vectorized buffer.

        The data dictionary contains tensors with a leading dimension equal to the number of
        environments. For dictionary observations, the 'obs' key maps to a nested dictionary
        of tensors, each with shape (num_envs, ...).

        Args:
            **data (torch.Tensor): Data to store, where 'obs' is a dict of tensors.
        """
        for i, buffer in enumerate(self.buffers):
            # Extract the i-th slice for each key in the data dictionary
            env_data = {
                k: v[i] if isinstance(v, torch.Tensor) else {sub_k: sub_v[i] for sub_k, sub_v in v.items()}
                for k, v in data.items()
            }

            buffer.store(**env_data)

    def finish_path(
        self,
        last_value_r: Optional[torch.Tensor] = None,
        last_value_c: Optional[torch.Tensor] = None,
        idx: int = 0,
    ) -> None:
        """Finish the current path for a specific environment buffer.

        Args:
            last_value_r (torch.Tensor, optional): Last state's value for reward.
            last_value_c (torch.Tensor, optional): Last state's value for cost.
            idx (int): Index of the environment buffer to finish the path for.
        """
        self.buffers[idx].finish_path(last_value_r, last_value_c)

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the concatenated data from all buffers.

        Collects data from each buffer, concatenates along the batch dimension, and applies
        advantage standardization if specified. The 'obs' key in the returned dictionary
        contains a nested dictionary of concatenated tensors.

        Returns:
            Dict[str, torch.Tensor]: Concatenated data, with 'obs' as a dict of tensors.
        """
        # Get data from the first buffer to determine the structure
        data_pre = self.buffers[0].get()
        concatenated_data = {k: [v] for k, v in data_pre.items()}

        # Append data from the remaining buffers
        for buffer in self.buffers[1:]:
            buffer_data = buffer.get()
            for k, v in buffer_data.items():
                concatenated_data[k].append(v)

        # Concatenate the lists of tensors along the batch dimension (dim=0)
        final_data = {}
        for k, v_list in concatenated_data.items():
            if isinstance(v_list[0], dict):  # Handle dictionary observations (e.g., 'obs')
                final_data[k] = {
                    sub_k: torch.cat([v[sub_k] for v in v_list], dim=0)
                    for sub_k in v_list[0].keys()
                }
            else:
                final_data[k] = torch.cat(v_list, dim=0)

        # Standardize advantages if required
        adv_mean, adv_std = distributed.dist_statistics_scalar(final_data['adv_r'])
        cadv_mean, _ = distributed.dist_statistics_scalar(final_data['adv_c'])
        if self._standardized_adv_r:
            final_data['adv_r'] = (final_data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            final_data['adv_c'] = final_data['adv_c'] - cadv_mean

        return final_data

