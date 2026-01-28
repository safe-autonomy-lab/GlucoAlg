import os
import argparse
import json
from typing import Dict, Tuple, List
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
import jax.numpy as jnp
from datetime import datetime, timedelta
from shield.predictive_shield import Shield
from shield.rule_based_shield import RuleBasedShield

# Import diabetes environment
from glucosim.diabetes_cmdp import DiabetesEnvs
from plot_utils import create_diabetes_animation
from glucosim.simglucose.evaluation.metrics import risk_index, glucose_variability_metrics, time_in_range

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_activation(activation: str) -> nn.Module:
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'relu':
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {activation}")

def _build_mlp(sizes: List[int], activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    for idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
        if idx < len(sizes) - 2:
            layers.append(_get_activation(activation))
    return nn.Sequential(*layers)

def _infer_policy_obs_dim(state_dict: Dict[str, torch.Tensor]) -> int | None:
    """Infer observation dimension from the first layer of the saved policy."""
    candidate_keys = [
        "logits_net.0.weight",  # categorical actor
        "mean.0.weight",        # gaussian actor
    ]
    for key in candidate_keys:
        if key in state_dict:
            return state_dict[key].shape[1]
    return None

class MultiCategoricalDistribution:
    """Lightweight wrapper to sample MultiDiscrete actions."""
    
    def __init__(self, dists: List[torch.distributions.Categorical]):
        self.dists = dists
    
    def sample(self):
        samples = [d.sample() for d in self.dists]
        return torch.stack(samples, dim=-1)
    
    def log_prob(self, actions: torch.Tensor):
        log_probs = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[..., i]))
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

class SimpleCategoricalActor(nn.Module):
    """Simplified categorical actor for discrete or multi-discrete actions."""
    
    def __init__(
        self,
        obs_dim: int,
        action_space: gym.spaces.Space,
        hidden_sizes: List[int] = [64, 64],
        activation: str = 'tanh',
        shield_type: str = 'none',
        logit_penalty: float = 10.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.is_multi_discrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        self.shield_type = shield_type
        if shield_type == 'none':
            self.shield = None
        elif shield_type == 'rule_based':
            self.shield = RuleBasedShield(logit_penalty=logit_penalty)
        else:
            self.shield = Shield(shield_type=shield_type, logit_penalty=logit_penalty)
        
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dims = [action_space.n]
        elif self.is_multi_discrete:
            self.action_dims = action_space.nvec.tolist()
        else:
            raise ValueError(f"Unsupported action space for categorical actor: {type(action_space)}")
        
        total_action_dim = sum(self.action_dims)
        self.logits_net = _build_mlp([obs_dim, *hidden_sizes, total_action_dim], activation)
    
    def forward(self, obs, original_obs=None):
        logits = self.logits_net(obs)
        if self.shield:
            # Always apply shield for safety rules (meal/bolus caps)
            logits = self.shield.apply(original_obs, logits, self.action_dims)
        if self.is_multi_discrete:
            splits = torch.split(logits, self.action_dims, dim=-1)
            dists = [torch.distributions.Categorical(logits=chunk) for chunk in splits]
            return MultiCategoricalDistribution(dists)
        return torch.distributions.Categorical(logits=logits)

class SimpleNormalizer:
    """Simple observation normalizer"""
    
    def __init__(self, obs_shape):
        self.obs_shape = obs_shape
        self.count = 0
        self.mean = np.zeros(obs_shape)
        self.var = np.ones(obs_shape)
        
    def normalize(self, obs):
        """Normalize observation"""
        if self.count > 0:
            return (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return obs
    
    def load_state_dict(self, state_dict):
        """Load normalizer state"""
        mean_key = state_dict.get('mean') or state_dict.get('_mean')
        var_key = state_dict.get('var') or state_dict.get('_var')
        count_key = state_dict.get('count') or state_dict.get('_count')
        print("Normalization loaded")
        if mean_key is not None:
            self.mean = mean_key.detach().cpu().numpy() if hasattr(mean_key, 'detach') else np.array(mean_key)
        if var_key is not None:
            self.var = var_key.detach().cpu().numpy() if hasattr(var_key, 'detach') else np.array(var_key)
        if count_key is not None:
            self.count = count_key.item() if hasattr(count_key, 'item') else int(count_key)
        
def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def create_diabetes_env(patient_type: str, patient_name: str, seed: int = 42) -> DiabetesEnvs:
    """Create diabetes environment
    
    Args:
        patient_type: Type of patient ('type1', 'type2', 'type2_no_pump')
        patient_name: Name of patient ('adolescent#001', 'adult#001', 'child#001')
        seed: Random seed
        
    Returns:
        DiabetesEnv instance
    """
    env = DiabetesEnvs(
        env_id=patient_type + "-v0",
        device='cpu',
        num_envs=1,
        render_mode='human',
        simulation_minutes=1440 * 7, # 7 DAYS
        sample_time=5,
        patient_name=patient_name,
        seed=seed,
    )
    return env

def load_model(model_path: str, config: Dict, env: DiabetesEnvs, shield_type: str = 'none', logit_penalty: float = 10.0) -> Tuple[nn.Module, DiabetesEnvs, SimpleNormalizer]:
    """Load the saved model and its configuration.
    
    Args:
        model_path: Path to the saved model weights
        config: The model configuration dictionary
        env: The environment instance
        use_shield: Whether to use the shield
        
    Returns:
        Tuple containing the loaded model, environment, and normalizer
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the saved model weights with CPU
    logger.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract policy state dict
    if "pi" in checkpoint:
        policy_state_dict = checkpoint["pi"]
    else:
        # If no 'pi' key, assume the whole checkpoint is the policy
        policy_state_dict = checkpoint
    
    # Get model configuration
    model_config = config["model_cfgs"]
    actor_type = model_config.get("actor_type", "gaussian")
    hidden_sizes = model_config["actor"]["hidden_sizes"]
    activation = model_config["actor"]["activation"]
    
    env_obs_dim = env.observation_space.shape[0]
    policy_obs_dim = _infer_policy_obs_dim(policy_state_dict)
    obs_dim = policy_obs_dim or env_obs_dim
    if policy_obs_dim and policy_obs_dim != env_obs_dim:
        logger.warning(
            "Checkpoint expects obs dim %s but environment provides %s. "
            "Truncating/padding observations to %s for evaluation.",
            policy_obs_dim, env_obs_dim, obs_dim,
        )
    action_space = env.action_space
    assert actor_type == "categorical" or isinstance(action_space, gym.spaces.MultiDiscrete), f"Actor type {actor_type} is not supported"
    actor = SimpleCategoricalActor(
        obs_dim=obs_dim,
        action_space=action_space,
        hidden_sizes=hidden_sizes,
        activation=activation,
        shield_type=shield_type,
        logit_penalty=logit_penalty,
    )
    
    # Load the weights into the model
    try:
        actor.load_state_dict(policy_state_dict)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        logger.info(f"Model state dict keys: {list(policy_state_dict.keys())}")
        raise
    
    # Set up observation normalization if available
    normalizer = None
    if "obs_normalizer" in checkpoint:
        logger.info("Setting up observation normalization")
        normalizer = SimpleNormalizer((obs_dim,))
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
    
    actor.eval()
    return actor, env, normalizer

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def _find_latest_epoch(torch_save_dir: str) -> int:
    """Find the highest epoch number in torch_save_dir."""
    if not os.path.isdir(torch_save_dir):
        raise FileNotFoundError(f"Missing torch_save dir: {torch_save_dir}")
    candidates = []
    for name in os.listdir(torch_save_dir):
        if not name.startswith("epoch-") or not name.endswith(".pt"):
            continue
        try:
            epoch_str = name[len("epoch-"):-len(".pt")]
            candidates.append(int(epoch_str))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No epoch-*.pt files found in {torch_save_dir}")
    return max(candidates)

def _resolve_model_paths(base_path: str, patient_type: str, algorithm: str, patient_name: str, seed: int) -> Dict[str, str]:
    """Resolve model/config paths across possible directory layouts."""
    patient_short = patient_name.split('#')[0] if patient_name else patient_name
    candidates = [
        os.path.join(base_path, patient_type, algorithm, f"seed{seed}"),
        os.path.join(base_path, patient_type, patient_short, algorithm, f"seed{seed}"),
        os.path.join(base_path, patient_type, patient_name, algorithm, f"seed{seed}"),
    ]
    for root in candidates:
        torch_save_dir = os.path.join(root, "torch_save")
        config_path = os.path.join(root, "config.json")
        if os.path.isdir(torch_save_dir) and os.path.exists(config_path):
            return {
                "torch_save_dir": torch_save_dir,
                "config_path": config_path,
                "root_dir": root,
            }
    searched = "\n  - " + "\n  - ".join(candidates)
    raise FileNotFoundError("Could not locate model directory. Searched:" + searched)

def _format_penalty_for_path(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")

def _shield_output_tag(shield_type: str, logit_penalty: float) -> str:
    if shield_type == 'none':
        return "no_shield"
    if shield_type == 'rule_based':
        return "rule_based_shield"
    return f"with_shield_{_format_penalty_for_path(logit_penalty)}"

def evaluate_model(env: DiabetesEnvs, actor: nn.Module, normalizer: SimpleNormalizer = None,
                  num_episodes: int = 10, render: bool = False, seed: int = 42, patient_type: str = "type1",
                  algorithm: str = "CUP", patient_name: str = "adolescent#001", save_seed: int = 1,
                  shield_type: str = 'child', save_plots: bool = False, logit_penalty: float = 10.0) -> Dict:
    """Evaluate the model on the diabetes environment
    
    Args:
        env: DiabetesEnvs environment
        actor: Trained actor model
        normalizer: Observation normalizer (optional)
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(seed)
    
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    episode_metrics = []
    sample_time_min = getattr(env, 'sample_time', 5)
    actor_obs_dim = getattr(actor, 'obs_dim', env.observation_space.shape[0])

    base_env = getattr(env, 'env', None)
    jax_env = base_env.envs[0] if base_env is not None and hasattr(base_env, 'envs') and base_env.envs else None
    bolus_levels = getattr(jax_env, 'bolus_levels', None) if jax_env is not None else None
    meal_levels = getattr(jax_env, 'meal_levels', None) if jax_env is not None else None
    exercise_levels = getattr(jax_env, 'exercise_levels', None) if jax_env is not None else None
    patient_params = getattr(jax_env, 'env_params', None)
    max_bolus_units = getattr(patient_params.patient_params, 'max_bolus_U', None) if patient_params is not None else None
    max_meal_grams = getattr(patient_params.patient_params, 'max_meal_g', None) if patient_params is not None else None
    max_exercise_min = getattr(patient_params.patient_params, 'max_exercise_min', None) if patient_params is not None else None
    
    shield_str = _shield_output_tag(shield_type, logit_penalty)
    action_out_dir = f"./diabetes_evaluation/{patient_type}/{algorithm}/{patient_name}/seed{save_seed}/{shield_str}"
    os.makedirs(action_out_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation for {num_episodes} episodes")
    
    def _to_numpy(x):
        """Convert tensors or lists to numpy arrays."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _convert_action_units(action_vals: np.ndarray) -> Tuple[float, float, float, int, int, int]:
        """Convert discrete action indices to physical units for plotting."""
        flat_action = np.array(action_vals, copy=False).flatten()
        bolus_idx = int(flat_action[0]) if flat_action.size > 0 else 0
        meal_idx = int(flat_action[1]) if flat_action.size > 1 else 0
        exercise_idx = int(flat_action[2]) if flat_action.size > 2 else 0

        bolus_units = float(flat_action[0]) if flat_action.size > 0 else 0.0
        meal_grams = float(flat_action[1]) if flat_action.size > 1 else 0.0
        exercise_minutes = float(flat_action[2]) if flat_action.size > 2 else 0.0

        if bolus_levels is not None and max_bolus_units is not None and bolus_idx < len(bolus_levels):
            bolus_units = float(bolus_levels[bolus_idx] * max_bolus_units)
        if meal_levels is not None and max_meal_grams is not None and meal_idx < len(meal_levels):
            meal_grams = float(meal_levels[meal_idx] * max_meal_grams)
        if exercise_levels is not None and max_exercise_min is not None and exercise_idx < len(exercise_levels):
            exercise_minutes = float(exercise_levels[exercise_idx] * max_exercise_min)

        return bolus_units, meal_grams, exercise_minutes, bolus_idx, meal_idx, exercise_idx
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        if getattr(actor, 'shield', None) is not None:
            actor.shield.reset()
        
        obs = _to_numpy(obs)
        obs_single = obs[0] if obs.ndim > 1 else obs
        episode_reward = 0.0
        episode_cost = 0.0
        episode_length = 0
        done = False
        
        # Track diabetes-specific metrics
        glucose_values = []
        hypo_events = 0
        hyper_events = 0
        
        # Track full history for animation
        history = {
            'time': [],
            'BG': [],
            'CGM': [],
            'CHO': [],
            'CHO_reccomendation': [],
            'CHO_natural': [],
            'insulin': [],
            'bolus_reccomendation': [],
            'IOB': [],
            'COB': [],
            'LBGI': [],
            'HBGI': [],
            'Risk': [],
            'bolus_units': [],
            'meal_grams': [],
            'exercise_minutes': [],
            'bolus_index': [],
            'meal_index': [],
            'exercise_index': [],
            'time_minutes': [],
            'time_hours': [],
            'action': [],
            'reward': [],
            'cost': [],
            'bolus_accepted': [],
            'meal_accepted': [],
            'bolus_limit_reached': [],
            'meal_limit_reached': [],
            'bolus_block_time_window': [],
            'bolus_block_bg_low': [],
            'meal_block_time_window': [],
            'meal_total_g': [],
            'scenario_meal_avg': [],
        }
        
        # Initialize start time for this episode
        start_time = datetime.now()
        viewer = None  # Initialize viewer variable
        
        while not done:
            obs_for_model = obs_single
            # Normalize observation if normalizer is available
            if normalizer is not None:
                obs_normalized = normalizer.normalize(obs_for_model)
            else:
                obs_normalized = obs_for_model
            
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
                original_obs_tensor = torch.FloatTensor(obs_for_model).unsqueeze(0)
                action_dist = actor(obs_tensor, original_obs_tensor)
                action = action_dist.sample()
                
                # Ensure correct type/shape for the environment
                if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                    action_env = action.detach().cpu().numpy().astype(np.int64)
                    if action_env.ndim == 1:
                        action_env = action_env[None, :]
                elif isinstance(env.action_space, gym.spaces.Discrete):
                    action_value = int(action.detach().cpu().numpy().squeeze())
                    action_env = np.array([action_value])
                else:
                    action_env = action.detach().cpu().numpy()
            
            # Step environment
            next_obs, reward, cost, terminated, truncated, info = env.step(action_env)
            
            # Update shield with the action actually taken
            if actor.shield is not None:
                # Ensure action is in correct format (batch, dim)
                action_tensor = torch.from_numpy(action_env).float().to(actor.shield.device)
                actor.shield.record_action(action_tensor)
            
            reward = float(np.asarray(reward).squeeze())
            cost = float(np.asarray(cost).squeeze())
            terminated_flag = bool(np.asarray(terminated).squeeze())
            truncated_flag = bool(np.asarray(truncated).squeeze())
            done = terminated_flag or truncated_flag
            
            next_obs = _to_numpy(next_obs)
            next_obs_single = next_obs[0] if next_obs.ndim > 1 else next_obs
            
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            
            # Track glucose metrics
            cgm_raw = info.get('cgm', obs_single[0])
            cgm = float(np.asarray(cgm_raw).squeeze())
            glucose_values.append(cgm)
            
            # Count hypoglycemic events (< 70 mg/dL)
            if cgm < 70:
                hypo_events += 1
                
            # Count hyperglycemic events (> 250 mg/dL)  
            if cgm > 250:
                hyper_events += 1
            
            # Collect history data for animation
            current_time = start_time + timedelta(minutes=episode_length) if render else datetime.now()
            history['time'].append(current_time)
            history['BG'].append(cgm)  # Use CGM as BG for simplicity
            history['CGM'].append(cgm)
            history['CHO'].append(float(np.asarray(info.get('cob', obs_single[2]))))
            action_record = np.array(action_env[0] if isinstance(action_env, np.ndarray) and action_env.ndim > 1 else action_env, copy=True)
            bolus_units, meal_grams, exercise_minutes, bolus_idx, meal_idx, exercise_idx = _convert_action_units(action_record)
            history['CHO_reccomendation'].append(meal_grams)  # Meal recommendation in grams
            history['CHO_natural'].append(0)  # No natural meals in this simulation
            history['insulin'].append(bolus_units)  # Bolus insulin in Units
            history['bolus_reccomendation'].append(history['insulin'][-1])
            history['IOB'].append(float(np.asarray(info.get('iob', obs_single[1]))))
            history['COB'].append(float(np.asarray(info.get('cob', obs_single[2]))))
            history['LBGI'].append(max(0, (39.0 - cgm)**1.084) if cgm < 39 else 0)  # Simplified LBGI
            history['HBGI'].append(max(0, (cgm - 180.0)**1.084) if cgm > 180 else 0)  # Simplified HBGI
            history['Risk'].append(history['LBGI'][-1] + history['HBGI'][-1])
            history['bolus_units'].append(bolus_units)
            history['meal_grams'].append(meal_grams)
            history['exercise_minutes'].append(exercise_minutes)
            history['bolus_index'].append(bolus_idx)
            history['meal_index'].append(meal_idx)
            history['exercise_index'].append(exercise_idx)
            history['time_minutes'].append(episode_length * sample_time_min)
            history['time_hours'].append((episode_length * sample_time_min) / 60.0)
            history['action'].append(np.array(action_record, copy=True))
            history['reward'].append(reward)
            history['cost'].append(cost)
            history['bolus_accepted'].append(bool(info.get('bolus_accepted', True)))
            history['meal_accepted'].append(bool(info.get('meal_accepted', True)))
            history['bolus_limit_reached'].append(bool(info.get('bolus_limit_reached', False)))
            history['meal_limit_reached'].append(bool(info.get('meal_limit_reached', False)))
            history['bolus_block_time_window'].append(bool(info.get('bolus_block_time_window', False)))
            history['bolus_block_bg_low'].append(bool(info.get('bolus_block_bg_low', False)))
            history['meal_block_time_window'].append(bool(info.get('meal_block_time_window', False)))
            history['meal_total_g'].append(float(np.asarray(info.get('meal_total_g', 0.0)).squeeze()))
            history['scenario_meal_avg'].append(float(np.asarray(info.get('scenario_meal_avg', 0.0)).squeeze()))
            
            obs_single = next_obs_single
        
        # Calculate episode metrics
        if glucose_values:
            glucose_trace = jnp.asarray(glucose_values)
            time_in_range_frac = float(time_in_range(glucose_trace))
            time_in_range_pct = time_in_range_frac * 100.0
            risk_score = float(risk_index(glucose_trace))
            sd_g, cv_pct, mag, mage = glucose_variability_metrics(glucose_trace, dt=sample_time_min)
            sd_g = float(sd_g)
            cv_pct = float(cv_pct)
            mag = float(mag)
            mage = float(mage)
            mean_glucose = float(np.mean(glucose_values))
        else:
            time_in_range_frac = 0.0
            time_in_range_pct = 0.0
            risk_score = 0.0
            sd_g = 0.0
            cv_pct = 0.0
            mag = 0.0
            mage = 0.0
            mean_glucose = 0.0

        episode_minutes = episode_length * sample_time_min
        episode_days = episode_minutes / 1440.0 if episode_minutes > 0 else 0.0
        meal_recommendations = sum(1 for idx in history['meal_index'] if idx > 0)
        bolus_recommendations = sum(1 for idx in history['bolus_index'] if idx > 0)
        meal_recs_per_day = meal_recommendations / episode_days if episode_days > 0 else 0.0
        bolus_recs_per_day = bolus_recommendations / episode_days if episode_days > 0 else 0.0
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
        
        episode_metrics.append({
            'episode': episode,
            'reward': episode_reward,
            'cost': episode_cost,
            'length': episode_length,
            'time_in_range_pct': time_in_range_pct,
            'time_in_range_frac': time_in_range_frac,
            'risk_index': risk_score,
            'sd_mgdl': sd_g,
            'cv_pct': cv_pct,
            'mag_mgdl_per_min': mag,
            'mage_mgdl': mage,
            'mean_glucose': mean_glucose,
            'meal_recommendations_per_day': meal_recs_per_day,
            'bolus_recommendations_per_day': bolus_recs_per_day,
            'hypo_events': hypo_events,
            'hyper_events': hyper_events
        })
        
        logger.info(f"Episode {episode + 1}/{num_episodes}: "
                   f"Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, "
                   f"Length={episode_length}, TIR={time_in_range_pct:.1f}%")
        
        # Create animation if rendering is enabled
        if render and len(history['time']) > 0:
            create_diabetes_animation(
                history, episode, patient_type, algorithm, patient_name, save_seed, 
                time_in_range_pct, episode_reward, episode_cost, shield_type=shield_type
            )
        # Always save action and glucose summaries for post-analysis
        if len(history['time']) > 0:
            # Save raw controller trace to CSV for later plotting
            # Filter out non-time-series keys for length calculation
            ts_keys = [k for k in history if k not in ['metrics', 'action'] and isinstance(history[k], list) and len(history[k]) > 0]
            if ts_keys:
                min_len = min(len(history[k]) for k in ts_keys)
                
                # Create dict for DataFrame
                data_dict = {}
                for k in ts_keys:
                    # Truncate to min_len to ensure alignment
                    data_dict[k] = history[k][:min_len]
                
                # Handle specific formatting or conversions
                if 'time' in data_dict:
                    data_dict['time'] = [t.isoformat() if isinstance(t, datetime) else t for t in data_dict['time']]
                
                df_history = pd.DataFrame(data_dict)
                
                # Add metrics as constant columns if available
                df_history["metric_tir"] = time_in_range_pct
                df_history["metric_risk_index"] = risk_score
                df_history["metric_sd"] = sd_g
                df_history["metric_cv"] = cv_pct
                df_history["metric_mag"] = mag
                df_history["metric_mage"] = mage
                
                csv_path = os.path.join(action_out_dir, f"episode_{episode}_controller.csv")
                df_history.to_csv(csv_path, index=False)
                logger.info(f"  ✅ Saved controller trace: {csv_path}")
            # Add metrics to history for optional rendering
            if save_plots:
                history['metrics'] = {
                    'tir': time_in_range_pct,
                    'sd': sd_g,
                    'cv': cv_pct,
                    'mag': mag,
                    'mage': mage
                }
                env.render(history=history, save_dir=action_out_dir, episode=episode)
            
        # Close viewer if it was created
        if viewer is not None:
            viewer.close()
    
    # Calculate summary statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_time_in_range': np.mean([m['time_in_range_pct'] for m in episode_metrics]),
        'mean_risk_index': np.mean([m['risk_index'] for m in episode_metrics]),
        'mean_sd_mgdl': np.mean([m['sd_mgdl'] for m in episode_metrics]),
        'mean_cv_pct': np.mean([m['cv_pct'] for m in episode_metrics]),
        'mean_mag_mgdl_per_min': np.mean([m['mag_mgdl_per_min'] for m in episode_metrics]),
        'mean_mage_mgdl': np.mean([m['mage_mgdl'] for m in episode_metrics]),
        'mean_glucose': np.mean([m['mean_glucose'] for m in episode_metrics]),
        'mean_meal_recommendations_per_day': np.mean([m['meal_recommendations_per_day'] for m in episode_metrics]),
        'std_meal_recommendations_per_day': np.std([m['meal_recommendations_per_day'] for m in episode_metrics]),
        'mean_bolus_recommendations_per_day': np.mean([m['bolus_recommendations_per_day'] for m in episode_metrics]),
        'std_bolus_recommendations_per_day': np.std([m['bolus_recommendations_per_day'] for m in episode_metrics]),
        'total_hypo_events': sum([m['hypo_events'] for m in episode_metrics]),
        'total_hyper_events': sum([m['hyper_events'] for m in episode_metrics]),
        'episode_details': episode_metrics
    }

    print('reuslts', results)
    
    return results

def main(patient_type: str, algorithm: str, patient_name: str, seed: int, epoch: int | None = 488,
         num_eval_episodes: int = 10, render: bool = False, shield_type: str = 'none',
         save_plots: bool = False, logit_penalty: float | None = None):
    """Main function to load and evaluate the diabetes model."""
    
    # Construct paths
    base_path = "./saved_models"
    paths = _resolve_model_paths(base_path, patient_type, algorithm, patient_name, seed)
    torch_save_dir = paths["torch_save_dir"]
    config_path = paths["config_path"]
    if epoch is None:
        epoch = _find_latest_epoch(torch_save_dir)
        logger.info("Auto-selected latest epoch: %s", epoch)
    model_path = f"{torch_save_dir}/epoch-{epoch}.pt"
    
    logger.info(f"Evaluating: {patient_type}/{algorithm}/{patient_name}/seed{seed}")
    logger.info(f"Model dir: {torch_save_dir}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Using shield: {shield_type}")
    
    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set evaluation seed
    eval_seed = 22
    set_seed(eval_seed)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    env = create_diabetes_env(patient_type=patient_type, patient_name=patient_name, seed=eval_seed)
    logger.info(f"Created {patient_type} diabetes environment")
    
    # Load model
    config_penalty = None
    if isinstance(config, dict):
        config_penalty = config.get("shield", {}).get("logit_penalty") if isinstance(config.get("shield"), dict) else None
    penalty_value = logit_penalty if logit_penalty is not None else config_penalty
    if penalty_value is None:
        penalty_value = 10.0
    actor, env, normalizer = load_model(model_path, config, env, shield_type=shield_type, logit_penalty=penalty_value)
    logger.info("Model loaded successfully")
    
    # Evaluate model
    results = evaluate_model(env, actor, normalizer, num_episodes=num_eval_episodes, render=render, seed=eval_seed,
                           patient_type=patient_type, patient_name=patient_name, algorithm=algorithm, save_seed=seed,
                           shield_type=shield_type, save_plots=save_plots, logit_penalty=penalty_value)
    
    # Create output directory
    shield_str = _shield_output_tag(shield_type, penalty_value)
    output_dir = f"./diabetes_evaluation/{patient_type}/{algorithm}/{patient_name}/seed{seed}/{shield_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results['episode_details'])
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    # Save summary results
    summary_data = {
        'Metric': [
            'Mean Episode Reward',
            'Mean Episode Cost', 
            'Mean Episode Length',
            'Mean Time in Range (%)',
            'Mean Risk Index',
            'Mean SD (mg/dL)',
            'Mean CV (%)',
            'Mean MAG (mg/dL/min)',
            'Mean MAGE (mg/dL)',
            'Mean Glucose (mg/dL)',
            'Mean Meal Recommendations per 24h',
            'Mean Bolus Recommendations per 24h',
            'Total Hypoglycemic Events',
            'Total Hyperglycemic Events'
        ],
        'Value': [
            f"{results['mean_reward']:.3f} ± {results['std_reward']:.3f}",
            f"{results['mean_cost']:.3f} ± {results['std_cost']:.3f}",
            f"{results['mean_length']:.1f} ± {results['std_length']:.1f}",
            f"{results['mean_time_in_range']:.1f}%",
            f"{results['mean_risk_index']:.3f}",
            f"{results['mean_sd_mgdl']:.3f}",
            f"{results['mean_cv_pct']:.3f}",
            f"{results['mean_mag_mgdl_per_min']:.4f}",
            f"{results['mean_mage_mgdl']:.3f}",
            f"{results['mean_glucose']:.1f} mg/dL",
            f"{results['mean_meal_recommendations_per_day']:.3f} ± {results['std_meal_recommendations_per_day']:.3f}",
            f"{results['mean_bolus_recommendations_per_day']:.3f} ± {results['std_bolus_recommendations_per_day']:.3f}",
            results['total_hypo_events'],
            results['total_hyper_events']
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    
    # Print results
    logger.info("\n=== EVALUATION RESULTS ===")
    logger.info(f"Patient Type: {patient_type}")
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Patient Name: {patient_name}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Shield: {shield_type}")
    logger.info(f"Episodes Evaluated: {num_eval_episodes}")
    logger.info(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    logger.info(f"Mean Cost: {results['mean_cost']:.3f} ± {results['std_cost']:.3f}")
    logger.info(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    logger.info(f"Time in Range: {results['mean_time_in_range']:.1f}%")
    logger.info(f"Risk Index: {results['mean_risk_index']:.3f}")
    logger.info(f"Glucose SD: {results['mean_sd_mgdl']:.3f} mg/dL")
    logger.info(f"Glucose CV: {results['mean_cv_pct']:.3f}%")
    logger.info(f"Glucose MAG: {results['mean_mag_mgdl_per_min']:.4f} mg/dL/min")
    logger.info(f"Glucose MAGE: {results['mean_mage_mgdl']:.3f} mg/dL")
    logger.info(f"Mean Glucose: {results['mean_glucose']:.1f} mg/dL")
    logger.info(f"Hypoglycemic Events: {results['total_hypo_events']}")
    logger.info(f"Hyperglycemic Events: {results['total_hyper_events']}")
    
    logger.info(f"Results saved to: {output_dir}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved diabetes RL model.")
    parser.add_argument("patient_type", type=str, help="t1d, t2d, or t2d_no_pump")
    parser.add_argument("algorithm", type=str, help="CUP, CPO, PPOLag, etc.")
    parser.add_argument("patient_name", type=str, help="adolescent#001, adult#001, child#001")
    parser.add_argument("seed", type=int, help="Model seed (e.g., 0, 1, 2)")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (default: latest)")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--shield", action="store_true", help="Enable shielding")
    parser.add_argument(
        "--shield-type",
        type=str,
        default=None,
        choices=["predictive", "rule_based", "none"],
        help="predictive (default), rule_based, or none",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save plots")
    parser.add_argument("--logit-penalty", type=float, default=-10, help="Shield logit penalty")
    args = parser.parse_args()

    # python eval_run.py t1d CPO adolescent#001 1 --shield
    # python eval_run.py t1d CPO adult#001 1 --shield --shield-type rule_based
    # python eval_run.py t1d CPO child#010 2 --epoch 488 --num-episodes 10 --render

    shield_mode = args.shield_type.lower() if args.shield_type else None
    if shield_mode == "none":
        shield_type = "none"
    elif args.shield or shield_mode is not None:
        if shield_mode in (None, "predictive"):
            shield_type = args.patient_name
        elif shield_mode == "rule_based":
            shield_type = "rule_based"
        else:
            raise ValueError(f"Unknown shield type: {shield_mode}")
        print(f"Shield type: {shield_type}")
    else:
        shield_type = "none"

    main(
        args.patient_type,
        args.algorithm,
        args.patient_name,
        args.seed,
        args.epoch,
        args.num_episodes,
        args.render,
        shield_type,
        args.save_plots,
        args.logit_penalty,
    )
