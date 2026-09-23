"""Validated, reusable configuration and CLI for discrete GlucoSim baselines."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from glucoalg.runtime import initialize_simulator

ALGORITHMS = ("PPOLag", "TRPOLag", "CUP", "CPO", "FOCOPS", "RCPO", "PCPO", "OnCRPO")
ENVIRONMENTS = ("t1d-v0", "t2d-v0", "t2d_no_pump-v0")
COHORTS = ("adolescent", "adult", "child")
DIRECT_COST_ALGORITHMS = {"CPO", "PCPO", "OnCRPO"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=ALGORITHMS, default="PPOLag")
    parser.add_argument("--env-id", choices=ENVIRONMENTS, default="t1d-v0")
    parser.add_argument("--cohort", choices=COHORTS, default="adolescent")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    # Preserve legacy defaults; CPU examples select one environment explicitly.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vector-env-nums", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", "--target_kl", dest="target_kl", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument("--lambda-lr", type=float, default=0.035)
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.001)
    parser.add_argument("--cost-limit", "--cost_limit", dest="cost_limit", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-use-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--project-name", default="[saferl4diabetes] baseliness")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--simulator-root", type=Path, help="checkout containing glucosim/")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=JSON",
                        help="explicit nested override, e.g. algo_cfgs.update_iters=1")
    parser.add_argument("--dry-run", action="store_true", help="validate and print config without training")
    # Former no-op flags are accepted at their historical defaults only.
    parser.add_argument("--safety-bonus", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--penalty-type", default="none", help=argparse.SUPPRESS)
    return parser


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _set_override(config: dict, text: str) -> None:
    path, separator, encoded = text.partition("=")
    keys = path.replace(":", ".").split(".")
    if not separator or len(keys) < 2 or any(not k for k in keys):
        raise ValueError(f"Override must be PATH=JSON: {text}")
    try:
        value = json.loads(encoded, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"Override value must be finite JSON: {text}") from exc
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise ValueError(f"Unknown configuration path: {path}")
        node = node[key]
    if keys[-1] not in node:
        raise ValueError(f"Unknown configuration path: {path}")
    old = node[keys[-1]]
    if isinstance(old, dict):
        raise ValueError(f"Override a leaf field instead of replacing a configuration section: {path}")
    if old is not None:
        valid = type(value) is type(old)
        if isinstance(old, float):
            valid = type(value) in (float, int) and math.isfinite(value)
            value = float(value) if valid else value
        if not valid:
            raise ValueError(f"Incorrect type for {path}: expected {type(old).__name__}")
    node[keys[-1]] = value


def build_config(args: argparse.Namespace) -> dict:
    """Build a fully resolved config without importing the simulator or torch."""
    import yaml

    if args.penalty_type != "none" or args.safety_bonus != 1.0:
        raise ValueError("--penalty-type and --safety-bonus were no-ops; training shields are not supported by this CLI")
    path = Path(__file__).resolve().parents[1] / "omnisafe" / "configs" / "on-policy" / f"{args.algo}.yaml"
    with path.open() as file:
        defaults = yaml.safe_load(file)
    config = copy.deepcopy(defaults["defaults"])

    def merge(target: dict, update: dict) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(config, defaults.get(args.env_id, {}))
    explicit = {
        "seed": args.seed,
        "train_cfgs": {"device": args.device, "parallel": args.parallel,
                       "total_steps": args.total_steps, "vector_env_nums": args.vector_env_nums},
        "logger_cfgs": {"use_wandb": args.use_wandb, "wandb_project": args.project_name,
                        "log_dir": str(args.log_dir)},
        "algo_cfgs": {"steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
                      "entropy_coef": args.entropy_coef, "target_kl": args.target_kl},
        "model_cfgs": {"actor": {"lr": args.actor_lr}, "critic": {"lr": args.critic_lr}},
        "env_cfgs": {"patient_name": f"{args.cohort}#001"},
    }
    if args.algo in DIRECT_COST_ALGORITHMS:
        explicit["algo_cfgs"]["cost_limit"] = args.cost_limit
    else:
        explicit["lagrange_cfgs"] = {"cost_limit": args.cost_limit, "lambda_lr": args.lambda_lr,
                                    "lagrangian_multiplier_init": args.lagrangian_multiplier_init}
    merge(config, explicit)
    for override in args.overrides:
        _set_override(config, override)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    def check_finite(value, path="config"):
        if isinstance(value, dict):
            for key, child in value.items():
                check_finite(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_finite(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must be finite")

    check_finite(config)
    train, algo = config["train_cfgs"], config["algo_cfgs"]
    for name, value in {
        "total_steps": train["total_steps"], "parallel": train["parallel"],
        "vector_env_nums": train["vector_env_nums"], "steps_per_epoch": algo["steps_per_epoch"],
        "batch_size": algo["batch_size"],
        "update_iters": algo["update_iters"], "torch_threads": train["torch_threads"],
        "save_model_freq": config["logger_cfgs"]["save_model_freq"],
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if train["total_steps"] < algo["steps_per_epoch"]:
        raise ValueError("total_steps must cover at least one full epoch")
    workers = train["parallel"] * train["vector_env_nums"]
    if algo["steps_per_epoch"] % workers:
        raise ValueError("steps_per_epoch must be divisible by parallel * vector_env_nums")
    if algo["batch_size"] > algo["steps_per_epoch"] // train["parallel"]:
        raise ValueError("batch_size exceeds rollout size per worker")
    for name in ("entropy_coef", "gamma", "cost_gamma", "lam", "lam_c", "penalty_coef"):
        if not 0 <= algo[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if algo["adv_estimation_method"] not in {"gae", "gae-rtg", "vtrace", "plain"}:
        raise ValueError("adv_estimation_method must be gae, gae-rtg, vtrace or plain")
    if "clip" in algo and algo["clip"] < 0:
        raise ValueError("clip must be nonnegative")
    if config["model_cfgs"]["actor_type"] != "categorical":
        raise ValueError("GlucoSim training requires a categorical actor")
    for network in ("actor", "critic"):
        sizes = config["model_cfgs"][network]["hidden_sizes"]
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError(f"{network}.hidden_sizes must contain positive integers")
    positives = {"actor.lr": config["model_cfgs"]["actor"]["lr"],
                 "critic.lr": config["model_cfgs"]["critic"]["lr"]}
    nonnegatives = {"entropy_coef": algo["entropy_coef"], "target_kl": algo["target_kl"]}
    cost_cfg = config.get("lagrange_cfgs", algo)
    nonnegatives["cost_limit"] = cost_cfg["cost_limit"]
    if "lagrange_cfgs" in config:
        positives["lambda_lr"] = cost_cfg["lambda_lr"]
        nonnegatives["lagrangian_multiplier_init"] = cost_cfg["lagrangian_multiplier_init"]
    for name, value in {**positives, **nonnegatives}.items():
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (name in positives and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name in positives else 'nonnegative'}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"algo": args.algo, "env_id": args.env_id, "config": config}, indent=2, allow_nan=False))
        return
    provenance = initialize_simulator(args.simulator_root)
    import omnisafe

    print(json.dumps({"runtime": provenance}, sort_keys=True))
    epochs = config["train_cfgs"]["total_steps"] // config["algo_cfgs"]["steps_per_epoch"]
    print(f"Training {epochs} full epochs ({epochs * config['algo_cfgs']['steps_per_epoch']} steps).")
    agent = omnisafe.Agent(args.algo, args.env_id, custom_cfgs=config)
    from omnisafe.utils.distributed import get_rank

    if get_rank() == 0:
        output = Path(agent.agent.logger.log_dir)
        (output / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
    agent.learn()


if __name__ == "__main__":
    main()
