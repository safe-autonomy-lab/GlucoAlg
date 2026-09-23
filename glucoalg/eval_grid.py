"""Explicit checkpoint/patient evaluation with a fail-closed result manifest.

Example::

    python -m glucoalg.eval_grid --checkpoint /run/torch_save/epoch-488.pt \
        --config /run/config.json --output-dir /results/eval \
        --patient-type t1d --patients 'adolescent#002' 'adolescent#003' \
        --episodes 10 --horizon-days 7 --action-mode stochastic
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

PATIENT_PATTERN = re.compile(r"(?:adolescent|adult|child)#[0-9]{3}\Z")
REQUIRED_METRICS = (
    "episode", "eval_seed", "time_in_range_pct", "risk_index", "reward", "cost", "length",
    "horizon_steps", "coverage_fraction", "terminated", "truncated", "early_termination",
    "termination_cause", "ghost_cost", "hypo_events", "hyper_events",
    "time_below_range_pct", "time_above_range_pct", "severe_hypo_samples",
    "time_in_range_frac", "sd_mgdl", "cv_pct", "mag_mgdl_per_min", "mage_mgdl",
    "mean_glucose", "meal_recommendations_per_day", "bolus_recommendations_per_day",
)
BOOLEAN_METRICS = {"terminated", "truncated", "early_termination", "cost_limit_exceeded"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boolean(value, name):
    if value in (True, 1, "1", "True", "true"):
        return True
    if value in (False, 0, "0", "False", "false"):
        return False
    raise ValueError(f"Invalid boolean {name}: {value!r}")


def validate_episode_rows(rows, expected_episodes):
    """Reject missing, duplicated, inconsistent or non-finite episode evidence."""
    if not isinstance(expected_episodes, int) or isinstance(expected_episodes, bool) or expected_episodes < 1:
        raise ValueError("expected_episodes must be a positive integer")
    if len(rows) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {len(rows)}")
    for index, row in enumerate(rows):
        missing = set(REQUIRED_METRICS) - row.keys()
        if missing:
            raise ValueError(f"Episode {index} missing metrics: {sorted(missing)}")
        values = {}
        for key, value in row.items():
            try:
                number = float(_boolean(value, key)) if key in BOOLEAN_METRICS else float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Episode {index} invalid {key}: {value!r}") from exc
            if not math.isfinite(number):
                raise ValueError(f"Episode {index} non-finite {key}: {value!r}")
            values[key] = number
        if values["episode"] != index:
            raise ValueError("Episode indices must be unique and contiguous from zero")
        length, horizon = values["length"], values["horizon_steps"]
        if length < 1 or length > horizon or length != int(length) or horizon != int(horizon):
            raise ValueError(f"Invalid episode length/horizon: {length}/{horizon}")
        if not math.isclose(values["coverage_fraction"], length / horizon, abs_tol=1e-8):
            raise ValueError("Coverage does not match episode length / horizon")
        if not (values["terminated"] or values["truncated"]):
            raise ValueError("Episode has neither terminated nor truncated")
        if bool(values["early_termination"]) != (length < horizon):
            raise ValueError("Early termination does not match episode coverage")
        for key in ("time_in_range_pct", "time_below_range_pct", "time_above_range_pct"):
            if not 0 <= values[key] <= 100:
                raise ValueError(f"{key} must be in [0, 100]")
        if not math.isclose(sum(values[key] for key in ("time_in_range_pct", "time_below_range_pct", "time_above_range_pct")), 100, abs_tol=1e-4):
            raise ValueError("Glucose range percentages must sum to 100")
        if not math.isclose(values["time_in_range_pct"], values["time_in_range_frac"] * 100, abs_tol=1e-4):
            raise ValueError("Time in range fraction and percentage disagree")
        for key in ("risk_index", "cost", "ghost_cost", "sd_mgdl", "cv_pct", "mag_mgdl_per_min",
                    "mage_mgdl", "mean_glucose", "meal_recommendations_per_day", "bolus_recommendations_per_day"):
            if values[key] < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("hypo_events", "hyper_events", "severe_hypo_samples"):
            if not 0 <= values[key] <= length or values[key] != int(values[key]):
                raise ValueError(f"{key} must be an integer sample count between zero and episode length")
        if values["severe_hypo_samples"] > values["hypo_events"]:
            raise ValueError("Severe hypoglycemia samples exceed all hypoglycemia samples")
    return rows


def summarize_episodes(rows):
    validate_episode_rows(rows, len(rows))
    result = {
        "n_episodes": len(rows),
        "mean_tir_pct": mean(float(row["time_in_range_pct"]) for row in rows),
        "mean_risk_index": mean(float(row["risk_index"]) for row in rows),
        "mean_reward": mean(float(row["reward"]) for row in rows),
        "mean_cost": mean(float(row["cost"]) for row in rows),
        "mean_coverage_fraction": mean(float(row["coverage_fraction"]) for row in rows),
        "early_terminations": sum(_boolean(row["early_termination"], "early_termination") for row in rows),
        "terminated_episodes": sum(_boolean(row["terminated"], "terminated") for row in rows),
        "truncated_episodes": sum(_boolean(row["truncated"], "truncated") for row in rows),
        "total_hypo_samples": sum(int(row["hypo_events"]) for row in rows),
        "total_hyper_samples": sum(int(row["hyper_events"]) for row in rows),
        "total_severe_hypo_samples": sum(int(row["severe_hypo_samples"]) for row in rows),
    }
    if all("cost_limit_exceeded" in row for row in rows):
        result["cost_limit_exceeded_episodes"] = sum(_boolean(row["cost_limit_exceeded"], "cost_limit_exceeded") for row in rows)
    return result


def _write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_protocol(patient_type, patients, episodes, horizon_days, eval_seed_base, action_mode):
    if patient_type not in ("t1d", "t2d", "t2dnp", "t2d_no_pump"):
        raise ValueError(f"Unknown patient type: {patient_type}")
    if not patients or len(set(patients)) != len(patients):
        raise ValueError("Provide at least one patient with no duplicates")
    for patient in patients:
        if not PATIENT_PATTERN.fullmatch(patient) or not 1 <= int(patient[-3:]) <= 10:
            raise ValueError(f"Invalid GlucoSim patient: {patient!r}; expected cohort#001 through cohort#010")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    minutes = float(horizon_days) * 1440
    if not math.isfinite(minutes) or minutes < 1440 or not math.isclose(minutes / 5, round(minutes / 5)):
        raise ValueError("horizon_days must be at least one day and contain whole 5-minute steps")
    if not isinstance(eval_seed_base, int) or isinstance(eval_seed_base, bool) or not 0 <= eval_seed_base <= 2**32 - episodes:
        raise ValueError("Evaluation reset seeds must fit unsigned 32-bit integers")
    if action_mode not in ("deterministic", "stochastic"):
        raise ValueError("action_mode must be deterministic or stochastic")


def _validate_shield_options(shield_type, logit_penalty=None):
    if shield_type not in ("none", "rule_based"):
        raise ValueError(
            "General evaluation supports only none or rule_based shielding; "
            "use glucoalg-dynamics rollout with an explicit predictor artifact "
            "for predictive shielding."
        )
    if logit_penalty is not None and (not math.isfinite(logit_penalty) or logit_penalty < 0):
        raise ValueError("logit_penalty must be a finite nonnegative magnitude")


def _penalty_magnitude(value):
    magnitude = float(value)
    if not math.isfinite(magnitude) or magnitude < 0:
        raise argparse.ArgumentTypeError("logit penalty must be a finite nonnegative magnitude")
    return magnitude


def run_grid(*, checkpoint, config, output_dir, patient_type, patients, episodes=10,
             horizon_days=7, eval_seed_base=22, action_mode="stochastic", simulator_root=None,
             algorithm=None, train_seed=None, shield_type="none", logit_penalty=10.0,
             save_traces=False, save_plots=False, cost_limit=None, render=False,
             patient_subdirs=True, glucose_source="legacy"):
    """Evaluate one checkpoint; never overwrite a prior nonempty result directory."""
    validate_protocol(patient_type, patients, episodes, horizon_days, eval_seed_base, action_mode)
    if glucose_source not in ("legacy", "post-step"):
        raise ValueError("glucose_source must be legacy or post-step")
    _validate_shield_options(shield_type, logit_penalty)
    if cost_limit is not None and (not math.isfinite(cost_limit) or cost_limit < 0):
        raise ValueError("cost_limit must be finite and nonnegative")
    checkpoint, config, output = Path(checkpoint).resolve(strict=True), Path(config).resolve(strict=True), Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    checkpoint_hash, config_hash = sha256_file(checkpoint), sha256_file(config)
    config_dict = json.loads(config.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    # The emptiness check alone races with another process. An exclusive file
    # creation claims even a pre-existing empty directory before any results
    # are written. Keep the claim alongside success/failure evidence.
    try:
        with (output / ".evaluation.lock").open("x", encoding="utf-8") as claim:
            claim.write(datetime.now(timezone.utc).isoformat() + "\n")
    except FileExistsError as exc:
        raise FileExistsError(f"Evaluation output is already claimed: {output}") from exc
    manifest_path = output / "EVAL_MANIFEST.json"
    manifest = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash, "algorithm": algorithm, "train_seed": train_seed},
        "config": {"path": str(config), "sha256": config_hash},
        "patient_type": patient_type, "requested_patients": list(patients), "episodes_per_patient": episodes,
        "horizon_days": horizon_days, "sample_time_minutes": 5, "eval_seed_base": eval_seed_base,
        "action_mode": action_mode, "shield_type": shield_type, "logit_penalty": logit_penalty,
        "cost_limit": cost_limit, "score_scope": "observed_trace_no_absorbing_padding",
        "glucose_source": glucose_source,
        "glucose_alignment": "info.cgm_else_pre_action_observation" if glucose_source == "legacy" else "post_step_observation",
        "cost_semantics": "simulator_native_including_terminal_ghost_cost",
        "policy_rng_protocol": "seed_once_per_patient; environment_reset_seed=eval_seed_base+episode",
        "patients": {},
    }
    _write_json(manifest_path, manifest)
    try:
        from glucoalg.runtime import configure_runtime, initialize_simulator

        configure_runtime()
        manifest["simulator"] = initialize_simulator(simulator_root)
        from glucoalg.evaluation import create_diabetes_env, evaluate_model, load_model, set_seed

        all_rows = []
        for patient in patients:
            set_seed(eval_seed_base)
            env = create_diabetes_env(patient_type, patient, seed=eval_seed_base, horizon_days=horizon_days)
            patient_output = output / patient if patient_subdirs else output
            try:
                actor, _, normalizer = load_model(checkpoint, config_dict, env, shield_type, logit_penalty)
                result = evaluate_model(
                    env, actor, normalizer, num_episodes=episodes, seed=eval_seed_base,
                    patient_type=patient_type, patient_name=patient, algorithm=algorithm or "policy",
                    save_seed=train_seed, shield_type=shield_type, logit_penalty=logit_penalty,
                    output_dir=patient_output, action_mode=action_mode, save_traces=save_traces,
                    save_plots=save_plots, cost_limit=cost_limit, render=render, glucose_source=glucose_source,
                )
            finally:
                env.close()
            rows = validate_episode_rows(result["episode_details"], episodes)
            csv_path = patient_output / "detailed_results.csv"
            with csv_path.open(newline="", encoding="utf-8") as stream:
                disk_rows = list(csv.DictReader(stream))
            validate_episode_rows(disk_rows, episodes)
            for row in rows:
                if row["eval_seed"] != eval_seed_base + row["episode"]:
                    raise ValueError("Episode seed does not match the evaluation protocol")
            manifest["patients"][patient] = {
                **summarize_episodes(rows), "episodes": rows, "csv_path": str(csv_path), "csv_sha256": sha256_file(csv_path),
            }
            all_rows.extend(rows)
            _write_json(manifest_path, manifest)
        if sha256_file(checkpoint) != checkpoint_hash or sha256_file(config) != config_hash:
            raise ValueError("Checkpoint or config changed during evaluation")
        # Each patient starts episode numbering at zero; aggregate copies receive
        # contiguous indices solely for the summary validator.
        aggregate_rows = [dict(row, episode=index) for index, row in enumerate(all_rows)]
        manifest["overall"] = {**summarize_episodes(aggregate_rows), "n_patients": len(patients)}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
    return manifest


def _common_options(parser, default_episodes):
    parser.add_argument("--episodes", "--num-episodes", type=int, default=default_episodes, dest="episodes")
    parser.add_argument("--horizon-days", type=float, default=7)
    parser.add_argument("--eval-seed-base", type=int, default=22)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--glucose-source", choices=("legacy", "post-step"), default="legacy",
                        help="legacy preserves the pre-action fallback; post-step scores returned CGM")
    parser.add_argument("--simulator-root", type=Path)
    parser.add_argument("--shield-type", choices=("none", "rule_based"), default="none")
    parser.add_argument("--logit-penalty", type=_penalty_magnitude, default=10.0,
                        help="Finite nonnegative magnitude subtracted from penalized action logits")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--cost-limit", type=float, help="Report episodes exceeding this cumulative simulator cost")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact .pt checkpoint file")
    parser.add_argument("--config", type=Path, required=True, help="Training config.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New/empty directory for this evaluation attempt")
    parser.add_argument("--patient-type", choices=("t1d", "t2d", "t2dnp", "t2d_no_pump"), required=True)
    parser.add_argument("--patients", nargs="+", required=True, help="Quoted patient names, e.g. 'adult#002' 'adult#003'")
    parser.add_argument("--algorithm", help="Optional descriptive checkpoint metadata")
    parser.add_argument("--train-seed", type=int, help="Optional descriptive training seed")
    _common_options(parser, 10)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        manifest = run_grid(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps(manifest["overall"], indent=2))


def run_legacy(patient_type, algorithm, patient_name, seed, epoch=None, num_eval_episodes=1,
               render=False, shield_type="none", save_plots=False, logit_penalty=None, *,
               checkpoint=None, config=None, output_dir=None, saved_models="./saved_models",
               horizon_days=7, eval_seed_base=22, action_mode="stochastic", simulator_root=None,
               save_traces=True, cost_limit=None, glucose_source="legacy"):
    _validate_shield_options(shield_type, logit_penalty)
    from glucoalg.runtime import configure_runtime

    configure_runtime()
    from glucoalg.evaluation import _find_latest_epoch, _resolve_model_paths, _shield_output_tag

    if (checkpoint is None) != (config is None):
        raise ValueError("--checkpoint and --config must be supplied together")
    if checkpoint is None:
        paths = _resolve_model_paths(saved_models, patient_type, algorithm, patient_name, seed)
        epoch = _find_latest_epoch(paths["torch_save_dir"]) if epoch is None else epoch
        checkpoint = Path(paths["torch_save_dir"]) / f"epoch-{epoch}.pt"
        config = paths["config_path"]
    if logit_penalty is None:
        config_dict = json.loads(Path(config).read_text(encoding="utf-8"))
        logit_penalty = config_dict.get("shield", {}).get("logit_penalty", 10.0)
    _validate_shield_options(shield_type, logit_penalty)
    if output_dir is None:
        output_dir = Path("diabetes_evaluation") / patient_type / algorithm / patient_name / f"seed{seed}" / _shield_output_tag(shield_type, logit_penalty)
        if action_mode != "stochastic":
            output_dir = output_dir / action_mode
    return run_grid(
        checkpoint=checkpoint, config=config, output_dir=output_dir, patient_type=patient_type,
        patients=[patient_name], episodes=num_eval_episodes, horizon_days=horizon_days,
        eval_seed_base=eval_seed_base, action_mode=action_mode, simulator_root=simulator_root,
        algorithm=algorithm, train_seed=seed, shield_type=shield_type, logit_penalty=logit_penalty,
        save_traces=save_traces, save_plots=save_plots, cost_limit=cost_limit, render=render, patient_subdirs=False,
        glucose_source=glucose_source,
    )


def legacy_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved diabetes RL model (legacy positional interface).")
    parser.add_argument("patient_type")
    parser.add_argument("algorithm")
    parser.add_argument("patient_name")
    parser.add_argument("seed", type=int)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--saved-models", type=Path, default=Path("saved_models"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--shield", action="store_true",
                        help="Unsupported legacy predictive option; use glucoalg-dynamics rollout")
    _common_options(parser, 1)
    parser.set_defaults(save_traces=True)
    args = vars(parser.parse_args(argv))
    args["num_eval_episodes"] = args.pop("episodes")
    if args.pop("shield"):
        parser.error("--shield requested legacy predictive shielding; use glucoalg-dynamics "
                     "rollout with an explicit predictor artifact. For static rules, use "
                     "--shield-type rule_based without --shield.")
    logging.basicConfig(level=logging.INFO)
    try:
        manifest = run_legacy(**args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps(manifest["overall"], indent=2))


if __name__ == "__main__":
    main()
