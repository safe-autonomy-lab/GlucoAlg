"""Configuration regressions exercised without initializing the simulator."""

import copy
import json
import subprocess
import sys

import pytest

from glucoalg.train import ALGORITHMS, build_config, build_parser


def config(*flags):
    return build_config(build_parser().parse_args(list(flags)))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_cost_limit_reaches_correct_optimizer_and_retains_defaults(algorithm):
    result = config("--algo", algorithm, "--cost_limit", "47")
    key = "algo_cfgs" if algorithm in {"CPO", "PCPO", "OnCRPO"} else "lagrange_cfgs"
    assert result[key]["cost_limit"] == 47
    assert result["model_cfgs"]["actor"]["lr"] == 1e-5
    assert result["env_cfgs"]["patient_name"] == "adolescent#001"


def test_parser_values_not_mutated_and_explicit_overrides_applied():
    args = build_parser().parse_args(["--set", "algo_cfgs.update_iters=3", "--use-wandb", "False"])
    original = copy.deepcopy(vars(args))
    result = build_config(args)
    assert vars(args) == original
    assert result["algo_cfgs"]["update_iters"] == 3
    assert result["logger_cfgs"]["use_wandb"] is False
    assert config("--use-wandb")["logger_cfgs"]["use_wandb"] is True


@pytest.mark.parametrize("flags", [
    ["--actor-lr", "nan"], ["--critic-lr", "-1"], ["--total-steps", "10"],
    ["--steps-per-epoch", "7"], ["--seed", "-1"], ["--lambda-lr", "0"],
    ["--cost-limit", "inf"], ["--penalty-type", "adult"],
    ["--set", "algo_cfgs.nonexistent=1"], ["--set", "algo_cfgs.batch_size=true"],
    ["--set", "algo_cfgs.entropy_coef=NaN"], ["--set", "train_cfgs.total_steps=0"],
    ["--set", 'model_cfgs.actor={"lr":0.0001}'],
    ["--set", "algo_cfgs.gamma=2.0"], ["--set", "algo_cfgs.update_iters=0"],
    ["--entropy-coef", "5"], ["--set", "model_cfgs.actor.hidden_sizes=[1e999]"],
])
def test_invalid_configuration_rejected(flags):
    with pytest.raises(ValueError):
        config(*flags)


def test_unknown_cli_argument_and_continuous_algorithm_rejected():
    for flags in (["--typo-lr", "1"], ["--algo", "DDPGLag"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(flags)


def test_dry_run_is_machine_readable_and_never_initializes_simulator():
    result = subprocess.run(
        [sys.executable, "-m", "glucoalg.train", "--dry-run", "--simulator-root", "/missing/simulator"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["config"]["seed"] == 100
