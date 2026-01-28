import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_aggregated_data(base_dir='saved_files/env_transitions'):
    train_transitions = {'X': {}, 'Y': {}}
    eval_transitions = {'X': {}, 'Y': {}}
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    for root, _, files in os.walk(base_dir):
        files.sort()
        # print(f"Processing directory: {root}")
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, base_dir)
                key = rel_path.replace(os.sep, '_')
                data = np.load(full_path)
                
                if 'data_x' in data and 'data_y' in data:
                    target = train_transitions if 'train' in file else eval_transitions
                    target['X'][key] = data['data_x']
                    target['Y'][key] = data['data_y']
    
    return train_transitions, eval_transitions
