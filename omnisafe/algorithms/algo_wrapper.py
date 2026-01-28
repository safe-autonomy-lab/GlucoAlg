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
"""Implementation of the AlgoWrapper Class."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch

from omnisafe.algorithms import ALGORITHM2TYPE, ALGORITHMS, registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.envs import support_envs
from omnisafe.evaluator import Evaluator
from omnisafe.utils import distributed
from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.plotter import Plotter
from omnisafe.utils.tools import recursive_check_config
from omnisafe.envs.wrapper import Normalizer, ObsNormalize


class AlgoWrapper:
    """Algo Wrapper for algorithms.

    Args:
        algo (str): The algorithm name.
        env_id (str): The environment id.
        train_terminal_cfgs (dict[str, Any], optional): The configurations for training termination.
            Defaults to None.
        custom_cfgs (dict[str, Any], optional): The custom configurations. Defaults to None.

    Attributes:
        algo (str): The algorithm name.
        env_id (str): The environment id.
        train_terminal_cfgs (dict[str, Any]): The configurations for training termination.
        custom_cfgs (dict[str, Any]): The custom configurations.
        cfgs (Config): The configurations for the algorithm.
        algo_type (str): The algorithm type.
    """

    algo_type: str

    def __init__(
        self,
        algo: str,
        env_id: str,
        train_terminal_cfgs: dict[str, Any] | None = None,
        custom_cfgs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an instance of :class:`AlgoWrapper`."""
        self.algo: str = algo
        self.env_id: str = env_id
        # algo_type will set in _init_checks()
        self.train_terminal_cfgs: dict[str, Any] | None = train_terminal_cfgs
        self.custom_cfgs: dict[str, Any] | None = custom_cfgs
        self._evaluator: Evaluator | None = None
        self._plotter: Plotter | None = None
        self.cfgs: Config = self._init_config()
        self._init_checks()
        self._init_algo()

    def _init_config(self) -> Config:
        """Initialize config.

        Initialize the configurations for the algorithm, following the order of default
        configurations, custom configurations, and terminal configurations.

        Returns:
            The configurations for the algorithm.

        Raises:
            AssertionError: If the algorithm name is not in the supported algorithms.
        """
        assert (
            self.algo in ALGORITHMS['all']
        ), f"{self.algo} doesn't exist. Please choose from {ALGORITHMS['all']}."
        self.algo_type = ALGORITHM2TYPE.get(self.algo, '')
        if self.train_terminal_cfgs is not None:
            if self.algo_type in ['model-based', 'offline']:
                assert (
                    self.train_terminal_cfgs['vector_env_nums'] == 1
                ), 'model-based and offline only support vector_env_nums==1!'
            if self.algo_type in ['off-policy', 'model-based', 'offline']:
                assert (
                    self.train_terminal_cfgs['parallel'] == 1
                ), 'off-policy, model-based and offline only support parallel==1!'

        cfgs = get_default_kwargs_yaml(self.algo, self.env_id, self.algo_type)

        # update the cfgs from custom configurations
        if self.custom_cfgs:
            # avoid repeatedly record the env_id and algo
            if 'env_id' in self.custom_cfgs:
                self.custom_cfgs.pop('env_id')
            if 'algo' in self.custom_cfgs:
                self.custom_cfgs.pop('algo')
            # validate the keys of custom configuration
            recursive_check_config(self.custom_cfgs, cfgs)
            # update the cfgs from custom configurations
            cfgs.recurisve_update(self.custom_cfgs)
            # save configurations specified in current experiment
            cfgs.update({'exp_increment_cfgs': self.custom_cfgs})
        # update the cfgs from custom terminal configurations
        if self.train_terminal_cfgs:
            # avoid repeatedly record the env_id and algo
            if 'env_id' in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop('env_id')
            if 'algo' in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop('algo')
            # validate the keys of train_terminal_cfgs configuration
            recursive_check_config(self.train_terminal_cfgs, cfgs.train_cfgs)
            # update the cfgs.train_cfgs from train_terminal configurations
            cfgs.train_cfgs.recurisve_update(self.train_terminal_cfgs)
            # save configurations specified in current experiment
            cfgs.recurisve_update({'exp_increment_cfgs': {'train_cfgs': self.train_terminal_cfgs}})

        # the exp_name format is PPO-{SafetyPointGoal1-v0}
        exp_name = f'{self.algo}-{{{self.env_id}}}'
        cfgs.recurisve_update({'exp_name': exp_name, 'env_id': self.env_id, 'algo': self.algo})
        if hasattr(cfgs.train_cfgs, 'total_steps') and hasattr(cfgs.algo_cfgs, 'steps_per_epoch'):
            epochs = cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.steps_per_epoch
            cfgs.train_cfgs.recurisve_update(
                {'epochs': epochs},
            )
        return cfgs

    def _init_checks(self) -> None:
        """Initial checks."""
        assert isinstance(self.algo, str), 'algo must be a string!'
        assert isinstance(self.cfgs.train_cfgs.parallel, int), 'parallel must be an integer!'
        assert self.cfgs.train_cfgs.parallel > 0, 'parallel must be greater than 0!'
        assert (
            self.env_id in support_envs()
        ), f"{self.env_id} doesn't exist. Please choose from {support_envs()}."

    def _init_algo(self) -> None:
        """Initialize the algorithm."""
        check_all_configs(self.cfgs, self.algo_type)
        if distributed.fork(
            self.cfgs.train_cfgs.parallel,
            device=self.cfgs.train_cfgs.device,
        ):
            # re-launches the current script with workers linked by MPI
            sys.exit()
        if self.cfgs.train_cfgs.device == 'cpu':
            torch.set_num_threads(self.cfgs.train_cfgs.torch_threads)
        else:
            if self.cfgs.train_cfgs.parallel > 1 and os.getenv('MASTER_ADDR') is not None:
                ddp_local_rank = int(os.environ['LOCAL_RANK'])
                self.cfgs.train_cfgs.device = f'cuda:{ddp_local_rank}'
            torch.set_num_threads(1)
            torch.cuda.set_device(self.cfgs.train_cfgs.device)
        os.environ['OMNISAFE_DEVICE'] = self.cfgs.train_cfgs.device
        self.agent: BaseAlgo = registry.get(self.algo)(
            env_id=self.env_id,
            cfgs=self.cfgs,
        )

    def learn(self) -> tuple[float, float, float]:
        """Agent learning.

        Returns:
            ep_ret: The episode return of the final episode.
            ep_cost: The episode cost of the final episode.
            ep_len: The episode length of the final episode.
        """
        ep_ret, ep_cost, ep_len = self.agent.learn()

        self._init_statistical_tools()

        return ep_ret, ep_cost, ep_len

    def _init_statistical_tools(self) -> None:
        """Initialize statistical tools."""
        self._evaluator = Evaluator()
        self._plotter = Plotter()

    def plot(self, smooth: int = 1) -> None:
        """Plot the training curve.

        Args:
            smooth (int, optional): window size, for smoothing the curve. Defaults to 1.

        Raises:
            AssertionError: If the :meth:`learn` method has not been called.
        """
        assert self._plotter is not None, 'Please run learn() first!'
        self._plotter.make_plots(
            [self.agent.logger.log_dir],
            None,
            'Steps',
            'Rewards',
            False,
            self.agent.cost_limit,
            smooth,
            None,
            None,
            'mean',
            self.agent.logger.log_dir,
        )

    def evaluate(self, num_episodes: int = 10, cost_criteria: float = 1.0) -> None:
        """Agent Evaluation.

        Args:
            num_episodes (int, optional): number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): the cost criteria to evaluate. Defaults to 1.0.

        Raises:
            AssertionError: If the :meth:`learn` method has not been called.
        """
        assert self._evaluator is not None, 'Please run learn() first!'
        scan_dir = os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save'))
        for item in scan_dir:
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                self._evaluator.load_saved(save_dir=self.agent.logger.log_dir, model_name=item.name)
                self._evaluator.evaluate(num_episodes=num_episodes, cost_criteria=cost_criteria)
        scan_dir.close()

    def load_model(self, model_path: str, training_steps: int = 2_000_000, device: str = 'gpu') -> None:
        """Load the model and config."""
        checkpoint = torch.load(model_path, map_location=device)
        # Load observation normalizer if present and configured
        if self.agent._cfgs.algo_cfgs.obs_normalize and 'obs_normalizer' in checkpoint:
            if "obs_normalizer" in checkpoint:
                if isinstance(self.agent._env, ObsNormalize):
                    env = self.agent._env._env
                else:
                    env = self.agent._env._env._env
                normalizer = Normalizer(env.observation_space.shape)
                normalizer.count = torch.tensor(training_steps).to(device)
                normalizer.load_state_dict(checkpoint["obs_normalizer"])
                env._obs_normalizer = normalizer
                
        # Load actor and critics
        self.agent._actor_critic.actor.load_state_dict(checkpoint['pi'])
        self.agent._actor_critic.reward_critic.load_state_dict(checkpoint['v_reward'])
        self.agent._actor_critic.cost_v_critic.load_state_dict(checkpoint['v_cost'])
        self.agent._actor_critic.cost_q_critic.load_state_dict(checkpoint['q_cost'])
        
        # Load optimizers
        self.agent._actor_critic.reward_critic_optimizer.load_state_dict(checkpoint['reward_optimizers'])
        self.agent._actor_critic.cost_v_critic_optimizer.load_state_dict(checkpoint['cost_v_optimizers'])
        self.agent._actor_critic.cost_q_critic_optimizer.load_state_dict(checkpoint['cost_q_optimizers'])
        
        # Load scheduler
        self.agent._actor_critic.actor_scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load Lagrange if present
        if 'lagrange' in checkpoint:
            lagrange_data = checkpoint['lagrange']
            # Assuming self._lagrange is already initialized with same config
            self.agent._lagrange.lagrangian_multiplier.data = lagrange_data['lagrangian_multiplier']
            self.agent._lagrange.lambda_optimizer.load_state_dict(lagrange_data['lambda_optimizer'])
            self.agent._lagrange.lagrangian_multiplier.data.clamp_(
                0.0, self.agent._lagrange.lagrangian_upper_bound
            )
        
    # pylint: disable-next=too-many-arguments
    def render(
        self,
        num_episodes: int = 10,
        render_mode: str = 'rgb_array',
        camera_name: str = 'track',
        width: int = 256,
        height: int = 256,
    ) -> None:
        """Evaluate and render some episodes.

        Args:
            num_episodes (int, optional): The number of episodes to render. Defaults to 10.
            render_mode (str, optional): The render mode, can be 'rgb_array', 'depth_array' or
                'human'. Defaults to 'rgb_array'.
            camera_name (str, optional): the camera name, specify the camera which you use to
                capture images. Defaults to 'track'.
            width (int, optional): The width of the rendered image. Defaults to 256.
            height (int, optional): The height of the rendered image. Defaults to 256.

        Raises:
            AssertionError: If the :meth:`learn` method has not been called.
        """
        assert self._evaluator is not None, 'Please run learn() first!'
        scan_dir = os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save'))
        for item in scan_dir:
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                self._evaluator.load_saved(
                    save_dir=self.agent.logger.log_dir,
                    model_name=item.name,
                    render_mode=render_mode,
                    camera_name=camera_name,
                    width=width,
                    height=height,
                )
                self._evaluator.render(num_episodes=num_episodes)
        scan_dir.close()
