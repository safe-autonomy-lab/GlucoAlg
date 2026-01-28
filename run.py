import os

# Force JAX to use CPU as the default platform/backend
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict
# We need to import envs to register the environments
import glucosim
from glucosim.diabetes_cmdp import DiabetesEnvs


def dataclass_to_dict(config: object) -> dict:
    return {k.lower(): getattr(config, k) for k in dir(config) if not k.startswith('_')}

if __name__ == '__main__':
    
    # python run.py --algo ShieldedTRPOLag --env-id SafetyPointGoal1-v1 --fe-representation True
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', type=str, metavar='ALGO', default='PPOLag', help='algorithm to train', choices=omnisafe.ALGORITHMS['all'])
    parser.add_argument('--env-id', type=str, metavar='ENV', default='t1d-v0', help='the name of test environment')
    parser.add_argument('--cohort', type=str, metavar='COHORT', default='adolescent', help='patient name')
    parser.add_argument('--total-steps', type=int, default=2000000, metavar='STEPS', help='total number of steps to train for algorithm')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEVICES', help='device to use for training')
    parser.add_argument('--vector-env-nums', type=int, default=4, metavar='VECTOR-ENV', help='number of vector envs to use for training')
    parser.add_argument('--seed', type=int, default=100, metavar='SEED', help='random seed')
    parser.add_argument('--entropy-coef', type=float, default=0.01, metavar='ENTROPY-COEF', help='entropy coef')
    parser.add_argument('--safety-bonus', type=float, default=1., metavar='SAFETY-BONUS', help='safety bonus')
    parser.add_argument('--penalty-type', type=str, default='none', metavar='PENALTY-TYPE', help='penalty type')
    parser.add_argument('--use-wandb', type=bool, default=False, metavar='USE-WANDB', help='whether to use wandb')
    # TODO: back to 11 original, currently reduced!!
    parser.add_argument('--steps-per-epoch', type=int, default=2**11, metavar='STEPS-PER-EPOCH', help='steps per epoch')
    parser.add_argument('--target_kl', type=float, default=0.1, metavar='TARGET-KL', help='target kl')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BATCH-SIZE', help='batch size')
    parser.add_argument('--lagrangian-multiplier-init', type=float, default=0.001, metavar='LAMBDA-INIT', help='lambda init')
    parser.add_argument('--lambda-lr', type=float, default=0.035, metavar='LAMBDA-LR', help='lambda lr')
    parser.add_argument('--project-name', type=str, default='[saferl4diabetes] baseliness', metavar='PROJECT-NAME', help='project name')
    parser.add_argument('--actor-lr', type=float, default=float(1e-5), metavar='ACTOR-LR', help='actor lr')
    parser.add_argument('--critic-lr', type=float, default=float(5e-5), metavar='CRITIC-LR', help='critic lr')
    parser.add_argument('--parallel', type=int, default=1, metavar='PARALLEL', help='parallel')
    parser.add_argument('--cost_limit', type=float, default=100.0, metavar='COST-LIMIT', help='cost limit')

    baselines = ['PPOLag', 'TRPOLag', 'CUP', 'CPO', 'FOCOPS', 'RCPO', 'PCPO', 'OnCRPO']
    offpolicy_baselines = ['DDPGLag', 'SACLag', 'TD3Lag']
    
    args, unparsed_args = parser.parse_known_args()

    assert args.penalty_type in ['child', 'adult', 'adolescent', 'none'], "Penalty type not supported, must be one of: child, adult, adolescent, none"

    assert args.algo in baselines + offpolicy_baselines, "Algorithm not supported"
    if args.algo in offpolicy_baselines:
        assert args.parallel == 1, "Parallel must be 1 for off-policy baselines"
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    fe_representation = False
    cohort = vars(args).pop('cohort')
    patient_name = cohort + "#001"
    # Test the env if it is fixed
    """
    env = DiabetesEnvs(args.env_id, device=args.device, num_envs=args.vector_env_nums, patient_name=patient_name)
    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        action = torch.from_numpy(action[None]).to(args.device)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        print("Hidden parameters features: ", info['hidden_parameters_features'])
        if terminated or truncated:
            break
    exit()
    """

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    # We should convert the dataclass to dict for the omnisafe config
    # project name
    project_name = vars(args).pop('project_name')
    
    custom_cfgs = {
        'seed': int(vars(args).pop('seed')),
        'logger_cfgs': {
            'use_wandb': vars(args).pop('use_wandb'),
            'wandb_project': project_name,
        },
        'train_cfgs': {
            'total_steps': int(vars(args).pop('total_steps')),
            'vector_env_nums': int(vars(args).pop('vector_env_nums')),
        },
        'algo_cfgs': {
            'steps_per_epoch': int(vars(args).pop('steps_per_epoch')),
            'batch_size': int(vars(args).pop('batch_size')),
            'entropy_coef': float(vars(args).pop('entropy_coef')),
            'target_kl': float(vars(args).pop('target_kl')),
        },
        'model_cfgs': {
            'critic': {
                'lr': float(vars(args).pop('critic_lr')),
            },
            'actor': {
                'lr': float(vars(args).pop('actor_lr')),
            }
        },
        'lagrange_cfgs': {
            'lambda_lr': vars(args).pop('lambda_lr'),
            'lagrangian_multiplier_init': vars(args).pop('lagrangian_multiplier_init'),
        }
    }

    safety_bonus = float(vars(args).pop('safety_bonus'))
    penalty_type = vars(args).pop('penalty_type')
    custom_cfgs['env_cfgs'] = {
        'patient_name': patient_name
    }
    
    if args.algo in ['CPO', 'OnCRPO', 'PCPO']:
        if args.algo in ['CPO', 'OnCRPO', 'PCPO']:
            custom_cfgs['algo_cfgs']['cost_limit'] = float(vars(args).pop('cost_limit'))
        else:
            custom_cfgs['algo_cfgs']['safety_budget'] = float(vars(args).pop('cost_limit'))
        custom_cfgs.pop('lagrange_cfgs')
    
    if args.algo in ["TRPOLag", "PPOLag", "FOCOPS", "CUP", "RCPO"]:
        custom_cfgs['lagrange_cfgs']['cost_limit'] = float(vars(args).pop('cost_limit'))
    
    if args.algo in offpolicy_baselines:
        custom_cfgs['train_cfgs']['parallel'] = 1
        custom_cfgs['algo_cfgs'].pop('entropy_coef')
        custom_cfgs.pop('shield_cfgs')

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )

    agent.learn()
