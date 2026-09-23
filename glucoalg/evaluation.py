"""Checkpoint evaluation using GlucoSim's own glucose metrics.

Glucose scores describe the observed trace. Coverage, termination and simulator
terminal cost penalties are reported separately. Optional shields and plotting
are loaded only on request. Callers must initialize the selected simulator.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)
PATIENT_TYPES = {"t1d": "t1d", "t2d": "t2d", "t2dnp": "t2d_no_pump", "t2d_no_pump": "t2d_no_pump"}


def _get_activation(activation):
    if activation == "tanh":
        return nn.Tanh()
    if activation == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {activation}")


def _build_mlp(sizes, activation):
    layers = []
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(sizes) - 2:
            layers.append(_get_activation(activation))
    return nn.Sequential(*layers)


def _infer_policy_obs_dim(state_dict):
    for name in ("logits_net.0.weight", "mean.0.weight"):
        if name in state_dict:
            return state_dict[name].shape[1]
    return None


class MultiCategoricalDistribution:
    def __init__(self, dists):
        self.dists = dists

    def sample(self):
        return torch.stack([dist.sample() for dist in self.dists], dim=-1)

    @property
    def mode(self):
        return torch.stack([dist.logits.argmax(dim=-1) for dist in self.dists], dim=-1)

    def log_prob(self, actions):
        return torch.stack([dist.log_prob(actions[..., i]) for i, dist in enumerate(self.dists)], dim=-1).sum(dim=-1)


class SimpleCategoricalActor(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_sizes=(64, 64), activation="tanh", shield_type="none", logit_penalty=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.is_multi_discrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        self.shield_type = shield_type
        self.shield = None
        if shield_type == "rule_based":
            from shield.rule_based_shield import RuleBasedShield

            self.shield = RuleBasedShield(logit_penalty=logit_penalty)
        elif shield_type != "none":
            raise ValueError(
                "Predictive shielding requires the experimental glucoalg-dynamics rollout "
                "workflow with an explicit predictor artifact. General checkpoint evaluation "
                "supports unshielded and rule-based modes. "
                "See docs/predictive-shield.md. "
                "Use --shield-type none or --shield-type rule_based."
            )
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dims = [action_space.n]
        elif self.is_multi_discrete:
            self.action_dims = action_space.nvec.tolist()
        else:
            raise ValueError(f"Unsupported categorical action space: {action_space}")
        self.logits_net = _build_mlp([obs_dim, *hidden_sizes, sum(self.action_dims)], activation)

    def forward(self, obs, original_obs=None):
        logits = self.logits_net(obs)
        if self.shield is not None:
            logits = self.shield.apply(original_obs, logits, self.action_dims)
        if self.is_multi_discrete:
            return MultiCategoricalDistribution([
                torch.distributions.Categorical(logits=chunk)
                for chunk in torch.split(logits, self.action_dims, dim=-1)
            ])
        return torch.distributions.Categorical(logits=logits)


def _to_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _scalar(value, name):
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected one {name}, got shape {array.shape}; evaluation requires one environment")
    result = float(array.item())
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}: {result}")
    return result


class SimpleNormalizer:
    """Frozen OmniSafe observation normalization, including saved std and clip."""
    def __init__(self, obs_shape):
        self.obs_shape = tuple(obs_shape)
        self.count = 0.0
        self.mean = np.zeros(obs_shape)
        self.var = np.ones(obs_shape)
        self.std = np.ones(obs_shape)
        self.clip = np.full(obs_shape, 1e6)

    def normalize(self, obs):
        if self.count <= 1:
            return obs
        return np.clip((obs - self.mean) / self.std, -self.clip, self.clip)

    def load_state_dict(self, state_dict):
        def value(name):
            # Avoid tensor truthiness; a saved zero value is also valid state.
            result = state_dict.get(name)
            return state_dict.get("_" + name) if result is None else result

        for name in ("mean", "var", "std", "clip"):
            saved = value(name)
            if saved is not None:
                array = _to_numpy(saved)
                if array.shape not in (self.obs_shape, ()) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid normalizer {name}: expected {self.obs_shape}, got {array.shape}")
                setattr(self, name, array)
        saved_count = value("count")
        if saved_count is not None:
            self.count = _scalar(saved_count, "normalizer count")
        if self.count < 0 or np.any(self.var < 0) or np.any(self.clip < 0):
            raise ValueError("Normalizer count, variance and clip must be nonnegative")
        if value("std") is None:
            self.std = np.maximum(np.sqrt(self.var), 1e-2)
        if self.count > 1 and np.any(self.std <= 0):
            raise ValueError("Normalizer standard deviation must be positive")


def load_config(config_path):
    with open(config_path, encoding="utf-8") as stream:
        return json.load(stream)


def create_diabetes_env(patient_type, patient_name, seed=42, horizon_days=7):
    from glucosim.diabetes_cmdp import DiabetesEnvs

    minutes = float(horizon_days) * 1440
    if not math.isfinite(minutes) or minutes < 1440 or not math.isclose(minutes / 5, round(minutes / 5)):
        raise ValueError("horizon_days must be at least one day and contain whole 5-minute steps")
    if patient_type not in PATIENT_TYPES:
        raise ValueError(f"Unknown patient type: {patient_type}")
    return DiabetesEnvs(
        env_id=PATIENT_TYPES[patient_type] + "-v0", device="cpu", num_envs=1,
        render_mode=None, simulation_minutes=round(minutes), sample_time=5,
        patient_name=patient_name, seed=seed,
    )


def load_model(model_path, config, env, shield_type="none", logit_penalty=10.0):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    policy = checkpoint.get("pi", checkpoint)
    model_config = config["model_cfgs"]
    if model_config.get("actor_type", "gaussian") != "categorical":
        raise ValueError("Evaluation currently supports categorical policy checkpoints only")
    obs_dim = env.observation_space.shape[0]
    saved_dim = _infer_policy_obs_dim(policy)
    if saved_dim is not None and saved_dim != obs_dim:
        raise ValueError(f"Checkpoint observation width {saved_dim} does not match simulator width {obs_dim}")
    actor = SimpleCategoricalActor(
        obs_dim, env.action_space, model_config["actor"]["hidden_sizes"],
        model_config["actor"]["activation"], shield_type, logit_penalty,
    )
    actor.load_state_dict(policy)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite policy weights")
    normalizer = None
    if "obs_normalizer" in checkpoint:
        normalizer = SimpleNormalizer((obs_dim,))
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
    actor.eval()
    return actor, env, normalizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def _find_latest_epoch(torch_save_dir):
    candidates = []
    for path in Path(torch_save_dir).glob("epoch-*.pt"):
        try:
            candidates.append(int(path.stem.removeprefix("epoch-")))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No epoch-*.pt files found in {torch_save_dir}")
    return max(candidates)


def _resolve_model_paths(base_path, patient_type, algorithm, patient_name, seed):
    base = Path(base_path) / patient_type
    candidates = [base / algorithm / f"seed{seed}"]
    candidates.extend(base / patient / algorithm / f"seed{seed}" for patient in (patient_name.split("#")[0], patient_name))
    for root in candidates:
        if (root / "torch_save").is_dir() and (root / "config.json").is_file():
            return {"torch_save_dir": str(root / "torch_save"), "config_path": str(root / "config.json"), "root_dir": str(root)}
    raise FileNotFoundError("Could not locate model directory. Searched: " + ", ".join(map(str, candidates)))


def _shield_output_tag(shield_type, logit_penalty):
    if shield_type == "none":
        return "no_shield"
    if shield_type == "rule_based":
        return "rule_based_shield"
    penalty = str(int(logit_penalty)) if float(logit_penalty).is_integer() else str(logit_penalty).replace(".", "p")
    return f"with_shield_{penalty}"


def _glucose_metrics(glucose_values, sample_time):
    import jax.numpy as jnp
    from glucosim.simglucose.evaluation.metrics import glucose_variability_metrics, risk_index, time_in_range

    if not len(glucose_values):
        raise ValueError("Cannot score an empty glucose trace")
    trace = jnp.asarray(glucose_values)
    sd, cv, mag, mage = glucose_variability_metrics(trace, dt=sample_time)
    tir = float(time_in_range(trace))
    return {
        "time_in_range_pct": 100 * tir, "time_in_range_frac": tir,
        "risk_index": float(risk_index(trace)), "sd_mgdl": float(sd),
        "cv_pct": float(cv), "mag_mgdl_per_min": float(mag), "mage_mgdl": float(mage),
        "mean_glucose": float(np.mean(glucose_values)),
    }


def _action_units(env, action):
    base = getattr(env, "env", None)
    inner = base.envs[0] if base is not None and getattr(base, "envs", None) else None
    params = getattr(getattr(inner, "env_params", None), "patient_params", None)
    indices = list(np.asarray(action).reshape(-1)) + [0, 0, 0]
    values = []
    for index, name, maximum in zip(indices, ("bolus_levels", "meal_levels", "exercise_levels"), ("max_bolus_U", "max_meal_g", "max_exercise_min")):
        levels, scale = getattr(inner, name, None), getattr(params, maximum, None)
        values.append(float(levels[int(index)] * scale) if levels is not None and scale is not None else float(index))
    return values


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty results: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(env, actor, normalizer=None, num_episodes=10, render=False, seed=42,
                   patient_type="t1d", algorithm="CUP", patient_name="adolescent#001", save_seed=1,
                   shield_type="none", save_plots=False, logit_penalty=10.0, *,
                   output_dir=None, action_mode="stochastic", save_traces=True, cost_limit=None,
                   glucose_source="legacy"):
    """Run episodes using historical stochastic sampling by default.

    Reset seeds are ``seed + episode``. The policy RNG is seeded once per patient
    for compatibility. Legacy hypo_events/hyper_events columns count samples.
    """
    from glucoalg.eval_grid import validate_episode_rows

    if not isinstance(num_episodes, int) or isinstance(num_episodes, bool) or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    if action_mode not in ("stochastic", "deterministic"):
        raise ValueError("action_mode must be stochastic or deterministic")
    if glucose_source not in ("legacy", "post-step"):
        raise ValueError("glucose_source must be legacy or post-step")
    if cost_limit is not None and (not math.isfinite(cost_limit) or cost_limit < 0):
        raise ValueError("cost_limit must be finite and nonnegative")
    sample_time = _scalar(getattr(env, "sample_time", 5), "sample_time")
    if sample_time <= 0:
        raise ValueError("sample_time must be positive")
    horizon_minutes = _scalar(getattr(env, "simulation_minutes", 10080), "simulation_minutes")
    horizon_steps = round(horizon_minutes / sample_time)
    if horizon_steps < 1:
        raise ValueError("Evaluation horizon must contain at least one step")
    output = Path(output_dir) if output_dir is not None else Path("diabetes_evaluation") / patient_type / algorithm / patient_name / f"seed{save_seed}" / _shield_output_tag(shield_type, logit_penalty)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    episodes = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        shield = getattr(actor, "shield", None)
        if shield is not None:
            shield.reset()
        reward_total = cost_total = ghost_cost = 0.0
        glucose, history = [], []
        meal_count = bolus_count = 0
        terminated_flag = truncated_flag = False
        start_time = datetime.now()
        for step in range(1, horizon_steps + 1):
            raw = _to_numpy(obs).reshape(-1)
            if raw.size != actor.obs_dim or not np.isfinite(raw).all():
                raise ValueError(f"Invalid observation at episode {episode}, step {step}")
            normalized = normalizer.normalize(raw) if normalizer is not None else raw
            if not np.isfinite(normalized).all():
                raise ValueError("Non-finite normalized observation")
            with torch.inference_mode():
                distribution = actor(torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0), torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0))
                action = distribution.sample() if action_mode == "stochastic" else distribution.mode
                action_env = action.cpu().numpy().astype(np.int64)
                if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                    action_env = action_env.reshape(1, -1)
                else:
                    action_env = action_env.reshape(1)
            obs, reward, cost, terminated, truncated, info = env.step(action_env)
            if shield is not None:
                shield.record_action(torch.as_tensor(action_env, dtype=torch.float32, device=shield.device))
            reward = _scalar(reward, "reward")
            cost = _scalar(cost, "cost")
            terminated_flag = bool(_scalar(terminated, "terminated"))
            truncated_flag = bool(_scalar(truncated, "truncated"))
            next_raw = _to_numpy(obs).reshape(-1)
            if next_raw.size != actor.obs_dim or not np.isfinite(next_raw).all():
                raise ValueError(f"Invalid post-step observation at episode {episode}, step {step}")
            # Locked GlucoSim has no info['cgm']. The old evaluator therefore
            # scored the pre-action observation, including reset and excluding
            # the final reading. Preserve that as an explicit legacy protocol.
            source = info.get("cgm", raw[0]) if glucose_source == "legacy" else next_raw[0]
            cgm = _scalar(source, "cgm")
            if cgm <= 0:
                raise ValueError("Glucose must be positive for risk index scoring")
            glucose.append(cgm)
            reward_total += reward
            cost_total += cost
            ghost_cost += _scalar(info.get("ghost_cost", 0), "ghost_cost")
            indices = action_env.reshape(-1).tolist() + [0, 0, 0]
            bolus_count += int(indices[0] > 0)
            meal_count += int(indices[1] > 0)
            if save_traces or save_plots or render:
                bolus, meal, exercise = _action_units(env, action_env)
                history.append({
                    "time": (start_time + timedelta(minutes=step * sample_time)).isoformat(),
                    "BG": cgm, "CGM": cgm, "IOB": _scalar(info.get("iob", raw[1]), "iob"),
                    "LBGI": max(0, (39.0 - cgm) ** 1.084) if cgm < 39 else 0,
                    "HBGI": max(0, (cgm - 180.0) ** 1.084) if cgm > 180 else 0,
                    "COB": _scalar(info.get("cob", raw[2]), "cob"), "CHO": _scalar(info.get("cob", raw[2]), "cob"),
                    "CHO_reccomendation": meal, "CHO_natural": 0, "insulin": bolus, "bolus_reccomendation": bolus,
                    "bolus_units": bolus, "meal_grams": meal, "exercise_minutes": exercise,
                    "bolus_index": indices[0], "meal_index": indices[1], "exercise_index": indices[2],
                    "time_minutes": step * sample_time, "time_hours": step * sample_time / 60,
                    "reward": reward, "cost": cost, "action": action_env.reshape(-1).tolist(),
                    **{name: bool(_scalar(info.get(name, default), name)) for name, default in (
                        ("bolus_accepted", True), ("meal_accepted", True), ("bolus_limit_reached", False),
                        ("meal_limit_reached", False), ("bolus_block_time_window", False),
                        ("bolus_block_bg_low", False), ("meal_block_time_window", False))},
                    "meal_total_g": _scalar(info.get("meal_total_g", 0), "meal_total_g"),
                    "scenario_meal_avg": _scalar(info.get("scenario_meal_avg", 0), "scenario_meal_avg"),
                })
            if terminated_flag or truncated_flag:
                break
        else:
            raise ValueError(f"Simulator did not terminate or truncate at expected horizon ({horizon_steps} steps)")
        days = step * sample_time / 1440
        metrics = {
            "episode": episode, "eval_seed": seed + episode, "reward": reward_total, "cost": cost_total,
            "length": step, **_glucose_metrics(glucose, sample_time),
            "meal_recommendations_per_day": meal_count / days, "bolus_recommendations_per_day": bolus_count / days,
            "hypo_events": sum(value < 70 for value in glucose), "hyper_events": sum(value > 250 for value in glucose),
            "time_below_range_pct": 100 * sum(value < 70 for value in glucose) / step,
            "time_above_range_pct": 100 * sum(value > 180 for value in glucose) / step,
            "severe_hypo_samples": sum(value < 54 for value in glucose),
            "terminated": terminated_flag, "truncated": truncated_flag, "early_termination": step < horizon_steps,
            "termination_cause": int(_scalar(info.get("termination_cause", 0), "termination_cause")),
            "horizon_steps": horizon_steps, "coverage_fraction": step / horizon_steps, "ghost_cost": ghost_cost,
        }
        if cost_limit is not None:
            metrics["cost_limit_exceeded"] = cost_total > cost_limit
        episodes.append(metrics)
        validate_episode_rows(episodes, len(episodes))
        if history and save_traces:
            trace_rows = [{key: value for key, value in row.items() if key != "action"} for row in history]
            for row in trace_rows:
                row.update({"metric_tir": metrics["time_in_range_pct"], "metric_risk_index": metrics["risk_index"]})
            _write_csv(output / f"episode_{episode}_controller.csv", trace_rows)
        if history and (save_plots or render):
            plot_history = {key: [row[key] for row in history] for key in history[0]}
            plot_history["metrics"] = {"tir": metrics["time_in_range_pct"], "sd": metrics["sd_mgdl"], "cv": metrics["cv_pct"], "mag": metrics["mag_mgdl_per_min"], "mage": metrics["mage_mgdl"]}
            if save_plots:
                env.render(history=plot_history, save_dir=str(output), episode=episode)
            if render:
                from plot_utils import create_diabetes_animation

                plot_history.pop("metrics", None)
                plot_history["time"] = [datetime.fromisoformat(value) for value in plot_history["time"]]
                create_diabetes_animation(plot_history, episode, patient_type, algorithm, patient_name, save_seed,
                                          metrics["time_in_range_pct"], reward_total, cost_total, shield_type=shield_type)
        logger.info("Episode %s/%s: reward=%.2f cost=%.2f length=%s TIR=%.1f%% terminated=%s",
                    episode + 1, num_episodes, reward_total, cost_total, step, metrics["time_in_range_pct"], terminated_flag)
    results = {"episode_details": episodes}
    for name in ("reward", "cost", "length", "risk_index", "sd_mgdl", "cv_pct", "mag_mgdl_per_min", "mage_mgdl", "mean_glucose", "meal_recommendations_per_day", "bolus_recommendations_per_day"):
        label = "glucose" if name == "mean_glucose" else name
        results[f"mean_{label}"] = float(np.mean([row[name] for row in episodes]))
        results[f"std_{label}"] = float(np.std([row[name] for row in episodes]))
    results["mean_time_in_range"] = float(np.mean([row["time_in_range_pct"] for row in episodes]))
    results["total_hypo_events"] = sum(row["hypo_events"] for row in episodes)
    results["total_hyper_events"] = sum(row["hyper_events"] for row in episodes)
    results["early_terminations"] = sum(row["early_termination"] for row in episodes)
    _write_csv(output / "detailed_results.csv", episodes)
    _write_csv(output / "summary_results.csv", [{"Metric": key, "Value": value} for key, value in results.items() if key != "episode_details"])
    return results


def main(patient_type, algorithm, patient_name, seed, epoch=488, num_eval_episodes=10,
         render=False, shield_type="none", save_plots=False, logit_penalty=None, **kwargs):
    """Compatibility API for scripts that previously imported eval_run.main."""
    from glucoalg.eval_grid import run_legacy

    return run_legacy(patient_type, algorithm, patient_name, seed, epoch, num_eval_episodes,
                      render, shield_type, save_plots, logit_penalty, **kwargs)
