# Installation

```bash
python -m pip install -e .
```

Important: Before running any scripts in this repository, you must install `glucosim` (environment dependency).

# Dependency to Omnisafe and FunctionEncoder
We do not have to install the other repositories, but include necessary sources code from other repositories.
This project builds upon the OmniSafe framework, and FunctionEncoder framework.

Omnisafe: https://github.com/PKU-Alignment/omnisafe

Function Encoder: https://github.com/tyler-ingebrand/FunctionEncoder

Again, don't need to install those codes, since this repository includes many of their original source codes, but modified for our implementation. 

Specifically, we have modified the original source code to support discrete action spaces by implementing a categorical_actor.py within omnisafe/models/actor folder, and modify related parts

All credit for the underlying safety-gymnasium and optimization algorithms goes to the original OmniSafe contributors.

For BA-NODE implementation, it highly depends on FunctionEncoder repository, we mainly implemented
a BA_NODE.py within FunctionEncoder/Model/Architecture folder, and modify related parts.

All credit for the underlying ba-node goes to the original FunctionEncoder contributors.

## Command
```bash
python run.py --algo CPO --env-id t1d-v0 --cohort adolescent --seed 100
```

## Key arguments
- `--algo`: algorithm name (e.g. `CPO`, `CUP`, `PPOLag`, `TRPOLag`)
- `--env-id`: `t1d-v0`, `t2d-v0`, `t2d_no_pump-v0`
- `--cohort`: `adolescent`, `adult`, or `child`
- `--total-steps`, `--vector-env-nums`, `--device`, `--seed`
- `--cost_limit`, `--lambda-lr`, `--lagrangian-multiplier-init`

## Notes
- `--cohort` maps to `patient_name=<cohort>#001` inside the environment.
- Output directories are controlled by OmniSafe; check console output for the run directory.
- To use wandb for logging, --use-wandb True, default value is False.

# Collecting Environment Transitions

This guide covers using `1.collect_transition.py` to generate transition datasets for dynamics training.

## What it does
- Runs a policy in the glucobench environment
- Logs transitions and saves `.npz` files under `saved_files/env_transitions/`
- Optionally writes a CSV episode history in the working directory

## Command
```bash
python 1.collect_transition.py \
  --patient_type t1d \
  --epoch 2400 \
  --patient_name adolescent#003 \
  --simulation_days 15
```

## Key arguments
- `--patient_type`: `t1d`, `t2d`, or `t2d_no_pump`
- `--patient_name`: patient id, e.g. `adolescent#003`
- `--epoch`: model checkpoint epoch to load (defaults are set per patient_type)
- `--simulation_days`: episode length in days
- `--model_path`, `--config_path`: optional explicit paths (override defaults)

## Output layout
Saved to:
```
saved_files/env_transitions/<patient_type>/<cohort>/<pid>/<train|eval>.npz
```
Example:
```
saved_files/env_transitions/t1d/adolescent/3/train.npz
```

## Notes
- The policy checkpoint is resolved from `trained_policies_for_collection/...` unless you pass `--model_path`.
- If you want to collect for multiple patients, you can call the script multiple times with different `--patient_name` values.


# Training Dynamics Predictors

This guide covers `2.train_dynamics_predictor.py` for training BA-NODE and baseline predictors using collected transitions.

## Prerequisite
Make sure `saved_files/env_transitions/` contains data from `1.collect_transition.py`.

## Commands
Train BA-NODE (FunctionEncoder):
```bash
python 2.train_dynamics_predictor.py --model ba_node
```

Train ITransformer baseline:
```bash
python 2.train_dynamics_predictor.py --model itf
```

Train Neural ODE baseline:
```bash
python 2.train_dynamics_predictor.py --model node
```

## Key arguments
- `--history_length`: input window length (default 24)
- `--prediction_length`: forecast horizon (default 24)
- `--n_basis`: number of bases for BA-NODE (default 5)
- `--epochs`, `--batch_size`, `--seed`
- `--load_model`: load an existing checkpoint instead of training

## Output layout
Models are saved under:
```
saved_files/dynamics_predictor/<model>_p<prediction>_h<history>/<seed>/
```
BA-NODE uses:
```
saved_files/dynamics_predictor/ba_node_b<n_basis>_p<prediction>_h<history>/<seed>/
```
Logs go to:
```
<save_folder>/logs/
```

## Notes
- The script auto-selects the correct trainer based on `--model`.
- If you change `history_length` or `prediction_length`, retrain the model.


# Evaluating Policies with Runtime Shields (eval_run.py)

This guide covers evaluating a trained policy using `eval_run.py`, with optional runtime shielding.

## Command
```bash
python eval_run.py t1d CPO adolescent#001 100 --epoch 2400 --num-episodes 10 --shield --shield-type predictive
```

## Required positional arguments
- `patient_type`: `t1d`, `t2d`, or `t2d_no_pump`
- `algorithm`: name used during training (e.g. `CPO`, `CUP`)
- `patient_name`: `adolescent#001`, `adult#001`, `child#001`
- `seed`: training seed

## Optional flags
- `--epoch`: checkpoint epoch (default: latest in `torch_save/`)
- `--num-episodes`: number of evaluation episodes
- `--render`: render episodes
- `--shield`: enable shielding
- `--shield-type`: `predictive`, `rule_based`, or `none`
- `--save-plots`: write plots/visualizations
- `--logit-penalty`: override shield logit penalty

## Expected model layout
`eval_run.py` looks for:
```
saved_models/<patient_type>/<algorithm>/seed<seed>/torch_save/epoch-*.pt
saved_models/<patient_type>/<algorithm>/seed<seed>/config.json
```
It also supports nested layouts like:
```
saved_models/<patient_type>/<cohort>/<algorithm>/seed<seed>/...
```

## Output layout
Evaluation outputs go to:
```
./diabetes_evaluation/<patient_type>/<algorithm>/<patient_name>/seed<seed>/<shield_tag>/
```

## Examples
Predictive shield (default when `--shield` is set):
```bash
python eval_run.py t1d CPO adolescent#001 100 --epoch 2400 --num-episodes 10 --shield
```

Rule-based shield:
```bash
python eval_run.py t1d CPO adolescent#001 100 --epoch 2400 --num-episodes 10 --shield --shield-type rule_based
```

No shield:
```bash
python eval_run.py t1d CPO adolescent#001 100 --epoch 2400 --num-episodes 10
```
