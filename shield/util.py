import pickle
import yaml
import torch.nn as nn
import torch
from FunctionEncoder import FunctionEncoder


# Returns the desired activation function by name
def get_activation(activation):
    if activation == "relu":
        return nn.relu
    if activation == "relu6":
        return nn.relu6
    elif activation == "tanh":
        return nn.tanh
    elif activation == "sigmoid":
        return nn.sigmoid
    else:
        raise ValueError(f"Unknown activation: {activation}")

def load_data(env_name, data_purpose):
    with open(f'saved_files/env_transitions/{env_name}_{data_purpose}_transitions.pkl', 'rb') as f:
        env_transitions = pickle.load(f)

    return env_transitions

def save_config(config, path):
    yaml_config = {}
    for key, value in config.items():
        if isinstance(value, tuple):
            yaml_config[key] = list(value)
        else:
            yaml_config[key] = value

    with open(path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    restored_config = {}
    tuple_keys = {'input_size', 'output_size'}
    for key, value in yaml_config.items():
        if key in tuple_keys and isinstance(value, list):
            restored_config[key] = tuple(value)
        else:
            restored_config[key] = value
    return restored_config

def load_model(folder_path, device: str | None = None):
    config = load_config(f"{folder_path}/config.yaml")
    requested_device = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested in config but not available. Falling back to CPU.")
        requested_device = "cpu"
    config["device"] = requested_device

    model = FunctionEncoder(**config)
    model.load(f"{folder_path}/best_model.pth")
    return model
