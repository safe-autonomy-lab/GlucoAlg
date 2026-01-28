import torch
from abc import abstractmethod
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any
from omnisafe.common.buffer.base import BaseDictBuffer

# Placeholder for OmnisafeSpace and DEVICE_CPU
OmnisafeSpace = spaces.Space
DEVICE_CPU = torch.device('cpu')

# Placeholder for advantage estimation methods
class AdvantageEstimator:
    GAE = 'gae'
    GAE_RTG = 'gae-rtg'
    VTRACE = 'vtrace'
    PLAIN = 'plain'

# Placeholder for discount_cumsum function (assuming it's defined elsewhere)
def discount_cumsum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    return torch.zeros_like(x)  # Replace with actual implementation

# Placeholder for distributed statistics (assuming it's defined elsewhere)
class distributed:
    @staticmethod
    def dist_statistics_scalar(x: torch.Tensor) -> Tuple[float, float]:
        return 0.0, 1.0  # Replace with actual implementation


class OnPolicyDictBuffer(BaseDictBuffer):
    """On-policy buffer for dictionary observation spaces.

    This buffer extends the BaseBuffer to include additional data for on-policy algorithms,
    such as advantages and value targets, while supporting dictionary observation spaces.

    Attributes:
        ptr (int): The pointer of the buffer.
        path_start_idx (int): The start index of the current path.
        max_size (int): The maximum size of the buffer.
        data (dict): The data stored in the buffer, including nested observation dictionary.
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        device (torch.device): The device to store the data.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: str,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyBuffer`."""
        super().__init__(obs_space, act_space, size, device)

        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c
        self.data['adv_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discounted_ret'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['adv_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['logp'] = torch.zeros((size,), dtype=torch.float32, device=device)

        self._gamma: float = gamma
        self._lam: float = lam
        self._lam_c: float = lam_c
        self._penalty_coefficient: float = penalty_coefficient
        self._advantage_estimator: str = advantage_estimator
        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size: int = size

        assert self._penalty_coefficient >= 0, 'penalty_coefficient must be non-negative!'
        assert self._advantage_estimator in ['gae', 'gae-rtg', 'vtrace', 'plain']

    @property
    def standardized_adv_r(self) -> bool:
        return self._standardized_adv_r

    @property
    def standardized_adv_c(self) -> bool:
        return self._standardized_adv_c

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer.

        .. warning::
            The total size of the data must be less than the buffer size.

        Args:
            data (torch.Tensor): The data to store, including 'obs' as a dictionary.
        """
        assert self.ptr < self.max_size, 'No more space in the buffer!'
        for key, value in data.items():
            if key != 'obs':
                self.data[key][self.ptr] = value.to(self._device)
            else:
                for key, value in data['obs'].items():
                    self.data['obs'][key][self.ptr] = value.to(self._device)
        self.ptr += 1

    def finish_path(
        self,
        last_value_r: Optional[torch.Tensor] = None,
        last_value_c: Optional[torch.Tensor] = None,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs.

        Args:
            last_value_r (torch.Tensor, optional): The value of the last state for reward.
            last_value_c (torch.Tensor, optional): The value of the last state for cost.
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discounted_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, Any]:
        """Get the data in the buffer.

        Returns:
            The data stored and calculated in the buffer, with 'obs' as a dictionary.
        """
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': {key: val for key, val in self.data['obs'].items()},
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data

    def _calculate_adv_and_value_targets(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the estimated advantage.

        Supports 'gae', 'gae-rtg', 'vtrace', 'plain' methods.

        Args:
            values (torch.Tensor): The value of states.
            rewards (torch.Tensor): The reward of states.
            lam (float): The lambda parameter in GAE formula.

        Returns:
            adv (torch.Tensor): The estimated advantage.
            target_value (torch.Tensor): The target value for the value function.
        """
        if self._advantage_estimator == 'gae':
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, self._gamma * lam)
            target_value = adv + values[:-1]

        elif self._advantage_estimator == 'gae-rtg':
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, self._gamma * lam)
            target_value = discount_cumsum(rewards, self._gamma)[:-1]

        elif self._advantage_estimator == 'vtrace':
            action_probs = self.data['logp'].exp()
            behavior_action_probs = action_probs  # Assuming behavior policy is the same
            target_value, adv, _ = self._calculate_v_trace(
                policy_action_probs=action_probs,
                values=values,
                rewards=rewards,
                behavior_action_probs=behavior_action_probs,
                gamma=self._gamma,
                rho_bar=1.0,
                c_bar=1.0,
            )

        elif self._advantage_estimator == 'plain':
            adv = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            target_value = discount_cumsum(rewards, self._gamma)[:-1]

        else:
            raise NotImplementedError

        return adv, target_value

    @staticmethod
    def _calculate_v_trace(
        policy_action_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        behavior_action_probs: torch.Tensor,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate V-trace targets.

        Args:
            policy_action_probs (torch.Tensor): Action probabilities of the policy.
            values (torch.Tensor): The value of states.
            rewards (torch.Tensor): The reward of states.
            behavior_action_probs (torch.Tensor): Action probabilities of the behavior policy.
            gamma (float): The discount factor.
            rho_bar (float): The maximum value of importance weights.
            c_bar (float): The maximum value of clipped importance weights.

        Returns:
            V-trace targets, shape=(batch_size, sequence_length)
        """
        sequence_length = policy_action_probs.shape[0]
        rhos = torch.div(policy_action_probs, behavior_action_probs)
        clip_rhos = torch.min(rhos, torch.as_tensor(rho_bar))
        clip_cs = torch.min(rhos, torch.as_tensor(c_bar))
        v_s = values[:-1].clone()
        last_v_s = values[-1]

        for index in reversed(range(sequence_length)):
            delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
            v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
            last_v_s = v_s[index]

        v_s_plus_1 = torch.cat((v_s[1:], values[-1:]))
        policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

        return v_s, policy_advantage, clip_rhos