import argparse
import os

# Force CPU execution for torch/JAX before importing those libraries
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import pandas as pd
import torch
import numpy as np
import glucosim  
from glucosim import gym_env as gym
from eval_run import load_model, load_config

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# Ensure PyTorch defaults to CPU even if CUDA is present
if hasattr(torch, "set_default_device"):
    torch.set_default_device("cpu")

def _init_controller(model_path, config, env, cohort: str, use_shield: bool):
    """Create controller and grab action scaling info from the env."""
    actor, env, normalizer = load_model(model_path, config, env, shield_type=cohort if use_shield else 'none')
    if use_shield:
        print("To use shield, first you need to obtain transition datasets, and then train the dynamics models")
        print("Otherwise, you will not be able to use shield")
    return actor, normalizer


def _controller_action(actor, normalizer, obs):
    """Map controller outputs to discrete env action indices."""
    with torch.no_grad():
        # Ensure writable tensor source to avoid PyTorch warning about non-writable NumPy arrays.
        org_obs_tensor = torch.FloatTensor(np.array(obs, copy=True)).unsqueeze(0)
        if normalizer is not None:
            obs_tensor = normalizer.normalize(org_obs_tensor)
        
        action_dist = actor(obs_tensor, org_obs_tensor)
        # Deterministic: take the mode (argmax) instead of sampling
        if hasattr(action_dist, "dists"):  # MultiCategoricalDistribution
            actions = [d.probs.argmax(dim=-1) for d in action_dist.dists]
            action = torch.stack(actions, dim=-1)
        else:
            action = action_dist.probs.argmax(dim=-1)
        return action.detach().cpu().numpy().squeeze()

def collect_transitions(model_path, config, env, patient_type, cohort, use_shield, collect_obs_dim=5):
    # We need the first 5 dimensions of obs to be the blood glucose prediction
    transitions_X = []
    transitions_Y = []
    history = []  # <-- new: collect step-by-step history

    actor, normalizer = _init_controller(model_path, config, env, cohort, use_shield)
    obs, info = env.reset()
    done = False
    step_cnt = 0
    
    while not done:
        recommended_action = _controller_action(actor, normalizer, obs)    
        next_obs, reward, cost, done, truncated, info = env.step(recommended_action)

        if '_last_termination_cause' in info:
            print('what is _last_termination_cause', info['_last_termination_cause'])

        meal_accepted = info['meal_accepted']
        bolus_accepted = info['bolus_accepted']

        # Ensure action is numpy array
        action = recommended_action
        
        if not meal_accepted:
            action[1] = 0.0
        
        if not bolus_accepted:
            action[0] = 0.0

        # Use Python list concatenation instead of numpy array "+" for attending, preserving both slices as list elements
        dyna_obs = list(next_obs[:collect_obs_dim]) + list(next_obs[-3:])
        
        # X: concatenate obs and action
        # obs is (10,), action is (3,) -> (13,)
        x = np.concatenate([dyna_obs, action])
        
        # Y: Only blood glucose prediction (delta)
        # BG is obs[0]
        delta_bg = next_obs[0] - obs[0]
        y = np.array([delta_bg]) 
        
        transitions_X.append(x)
        transitions_Y.append(y)

        # ------------------ History logging ------------------
        history_dict = {
            'step': step_cnt,
            'BG': obs[0],
            'COB': obs[2],
            'IOB': obs[1],
        }

        # Announced actions
        for i in range(len(action)):
            history_dict[f'action_{i}'] = action[i]
        
        for i in range(len(recommended_action)):
            history_dict[f'recommended_action_{i}'] = recommended_action[i]

        history.append(history_dict)
        # ----------------------------------------------------
        
        obs = next_obs
        done = done or truncated
        step_cnt += 1
    
    print(f"Collected {len(transitions_X)} transitions")
    print(f"Transitions X shape: {np.array(transitions_X).shape}")
    print(f"Transitions Y shape: {np.array(transitions_Y).shape}")
    if history:
        df = pd.DataFrame(history)
        csv_file = f"episode_history_{patient_type}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved episode history ({len(history)} steps) to {csv_file}")
    return {'X': np.array(transitions_X), 'Y': np.array(transitions_Y)}

def save_data_npz(data, patient_type, patient_name, data_purpose='train'):
    # patient_name format: cohort#id (e.g., adolescent#001)
    try:
        cohort, pid = patient_name.split('#')
    except ValueError:
        # Fallback if format is unexpected
        cohort = "unknown"
        pid = patient_name

    # Remove leading zeros for folder name if preferred, or keep them. 
    # User example: "env_transitions/t1d/adolescent/1/train.npz"
    # We will strip leading zeros from pid for the folder name to match "1"
    try:
        pid_int = int(pid)
        pid_str = str(pid_int)
    except ValueError:
        pid_str = pid

    # Construct path: saved_files/env_transitions/{patient_type}/{cohort}/{pid}/
    base_dir = 'saved_files/env_transitions'
    save_dir = os.path.join(base_dir, patient_type, cohort, pid_str)
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f'{data_purpose}.npz')
    
    # Save as compressed numpyz
    # Keys will be 'data_x' and 'data_y'
    np.savez_compressed(file_path, data_x=data['X'], data_y=data['Y'])
    print(f"Saved {data_purpose} data to {file_path} (Samples: {len(data['X'])})")

DEFAULT_EPOCHS = {
    "t1d": 2400,
    "t2d": 2300,
    "t2d_no_pump": 2400,
}

def resolve_model_and_config(patient_type: str, cohort:str, epoch: int = None, model_path: str = None, config_path: str = None):
    """Resolve model/config paths with sensible defaults."""
    base_dir = os.path.join("trained_policies_for_collection", patient_type, cohort, "CPO", "seed100")
    resolved_epoch = epoch or DEFAULT_EPOCHS[patient_type]
    resolved_model = model_path or os.path.join(base_dir, "torch_save", f"epoch-{resolved_epoch}.pt")
    resolved_config = config_path or os.path.join(base_dir, "config.json")
    return resolved_model, resolved_config


def run_collection(patient_type, model_path, config, patient_name=None, cohort='none', use_shield=False, collect_obs_dim=5, simulation_days=None):
    # Define patients list
    if patient_name:
        patients = [patient_name]
    else:
        # Generate all 30 patients
        cohorts = ['adolescent', 'adult', 'child']
        patients = []
        for cohort in cohorts:
            for i in range(1, 3):
                patients.append(f"{cohort}#{i:03d}")

    # Collect for each patient
    for p_name in patients:
        print(f"Processing patient: {p_name}")
        
        # We collect transition dynamics for both training and evaluation
        for data_purpose in ['train', 'eval']:
            env_kwargs = {"patient_name": p_name}
            if simulation_days is not None:
                env_kwargs["simulation_minutes"] = simulation_days * 24 * 60
                print("Training Simulation days: ", simulation_days)
            
            if data_purpose == 'eval':
                env_kwargs["simulation_minutes"] = simulation_days * 24 * 60 // 3
                print("Evaluation Simulation days: ", min(simulation_days // 3, 1))
            env = gym.make(patient_type + '-v0', **env_kwargs)
            transitions = collect_transitions(
                model_path,
                config,
                env,
                patient_type=patient_type,
                cohort=cohort,
                collect_obs_dim=collect_obs_dim,
                use_shield=use_shield,
                )
            env.close()
            save_data_npz(transitions, patient_type, p_name, data_purpose)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect transition data for diabetes patients.')
    parser.add_argument('--patient_type', type=str, default='t1d', 
                        choices=['t1d', 't2d', 't2d_no_pump'],
                        help='Type of diabetes patient (default: t1d)')
    parser.add_argument('--patient_name', type=str, default='adolescent#001',
                        help='Specific patient name (e.g., adolescent#001). If omitted, all patients will be processed.')
    parser.add_argument('--collect_obs_dim', type=int, default=5,
                        help='Number of observations to collect (default: 5)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Model epoch to load. Defaults to 1200 for t1d and 2400 for t2d/t2d_no_pump.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Optional explicit model path. Overrides --epoch.')
    parser.add_argument('--use_shield', type=bool, default=False,
                        help='Whether to use shield. If True, use shield. If False, do not use shield.')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Optional explicit config path.')
    parser.add_argument('--simulation_days', type=int, default=3,
                        help='Override episode length (days). If omitted, uses env defaults (24h for t1d, 72h for t2d).')
    args = parser.parse_args()

    # For T2D, USE CPO epochs 2400: trained_policies_for_collection/t2d/CPO/seed1/torch_save/epoch-2400.pt
    # For T2D_no_pump, USE CPO epochs 2400: trained_policies_for_collection/t2d_no_pump/CPO/seed1/torch_save/epoch-2400.pt
    # python 1.collect_transition.py --patient_type t1d --epoch 2400 --patient_name adolescent#003 --simulation_days 3

    
    cohort = args.patient_name.split('#')[0]
    model_path, config_path = resolve_model_and_config(
        patient_type=args.patient_type,
        cohort=cohort,
        epoch=args.epoch,
        model_path=args.model_path,
        config_path=args.config_path,
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f"Using model checkpoint: {model_path}")
    print(f"Using config: {config_path}")
    config = load_config(config_path)

    run_collection(
        args.patient_type,
        model_path,
        config,
        patient_name=args.patient_name,
        cohort=cohort,
        collect_obs_dim=args.collect_obs_dim,
        simulation_days=args.simulation_days,
        use_shield=args.use_shield,
    )
