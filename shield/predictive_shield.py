import torch
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dynamics_trainers import load_function_encoder
from dynamics_utils import load_aggregated_data
from FunctionEncoder.Dataset.TransitionDataset import SequentialTransitionDataset


@dataclass(frozen=True)
class ShieldParams:
    """
    Safety parameters designed to prevent dangerous scenarios.
    """

    # BG Thresholds
    BG_HYPO_THRESHOLD: float = 80.0
    BG_CRITICAL_HYPO: float = 60.0
    # Minimal intervention window
    MIN_INTERVENTION_BG_LOW: float = 90.0
    MIN_INTERVENTION_BG_HIGH: float = 160.0

    # Rescue behavior
    RESCUE_MEAL_LEVEL: int = 1
    RESCUE_COOLDOWN_MIN: float = 60.0


@dataclass(frozen=True)
class PredictiveShieldConfig:
    """Controls how the predictive model is used for robust safety checks."""
    k_sigma: float = 2.0
    hypo_check_threshold: float = 80.0
    hyper_check_threshold: float = 250.0
    top_k_bolus_levels: int = 2
    logit_penalty: float = 10.0
    use_meal_hyper_check: bool = False


class Shield:
    """
    Safety shield for T1D insulin/meal recommendations with IOB-aware correction logic
    and predictive verification.
    """

    def __init__(self, params: Optional[ShieldParams] = None, shield_type: str = 'child', logit_penalty: float = 10.0):
        self.params = params or ShieldParams()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        save_folder = f"saved_files/dynamics_predictor/fe_b5_p24_h24/1"
        train_transitions, eval_transitions = load_aggregated_data()
        if shield_type != 'none':
            dataset = SequentialTransitionDataset(
                train_transitions, 
                eval_transitions, 
                history_length=24, 
                prediction_length=24, 
                batch_size=32, 
                dtype=torch.float32, 
                use_normalization=True, 
                device=device
            )
            dynamics_model = load_function_encoder(save_folder, dataset)
            self.dynamics_model = dynamics_model.to(device)
            self.dynamics_model.eval()
            self.input_mean = dataset.torch_series_mean.to(device)
            self.input_std = dataset.torch_series_std.to(device)
            self.output_std = dataset.torch_series_std_blood_glucose.to(device)
        else:
            self.dynamics_model = None
            self.input_mean = None
            self.input_std = None
            self.output_std = None

        self.shield_type = shield_type
        self.shield_index = int(shield_type.split('#')[1]) - 1
        if self.shield_type == 'adolescent':
            self.shield_index += 20
        elif self.shield_type == 'child':
            self.shield_index += 10

        # State tracking
        self.prev_cob: Optional[float] = None
        self.prev_iob: Optional[float] = None
        self.prev_bg: Optional[float] = None
        self.cob: Optional[float] = None
        self.iob: Optional[float] = None
        self.bg: Optional[float] = None
        self.bg_trend: Optional[float] = None
        self.prev_action: Optional[torch.Tensor] = None
        self.action_dim: Optional[int] = None
        self.prev_obs: Optional[torch.Tensor] = None
        
        # Hypo recovery tracking
        self.in_hypo_recovery: bool = False
        self.hypo_recovery_carbs_given: float = 0.0

        # Predictive model context
        self.pred_cfg = PredictiveShieldConfig(logit_penalty=logit_penalty)
        self.history_len = 24
        self.prediction_len = int(getattr(self.dynamics_model, "prediction_horizon", 24))
        self.obs_x_history: deque = deque(maxlen=self.history_len + self.prediction_len)
        self.obs_y_history: deque = deque(maxlen=self.history_len + self.prediction_len)
        self.example_nbr = 20
        self.context_x_history: deque = deque(maxlen=self.example_nbr)
        self.context_y_history: deque = deque(maxlen=self.example_nbr)
        
        # Recommendation cooldown
        self.step_counter: int = 0
        self.last_meal_step: Optional[int] = None
        self.last_bolus_step: Optional[int] = None
        self.last_rescue_step: Optional[int] = None

    def reset(self) -> None:
        """Reset shield state for a new episode."""
        self.prev_cob = None
        self.prev_iob = None
        self.prev_bg = None
        self.cob = None
        self.iob = None
        self.bg = None
        self.bg_trend = None
        self.in_hypo_recovery = False
        self.hypo_recovery_carbs_given = 0.0
        self.obs_x_history.clear()
        self.obs_y_history.clear()
        self.step_counter = 0
        self.last_meal_step = None
        self.last_bolus_step = None
        self.last_rescue_step = None
        self.prev_action = None
        self.action_dim = None
        self.prev_obs = None

    def _maybe_append_context_example(self) -> bool:
        if len(self.obs_x_history) < self.history_len + self.prediction_len:
            return False

        raw_x = list(self.obs_x_history)
        raw_y = list(self.obs_y_history)
        if len(raw_y) >= len(raw_x):
            raw_y = raw_y[:-1]

        y_start = self.history_len - 1
        y_end = y_start + self.prediction_len
        if y_end > len(raw_y):
            return False

        x_window = torch.stack(raw_x[:self.history_len], dim=0).transpose(0, 1)
        y_window = torch.stack(raw_y[y_start:y_end], dim=0).squeeze(1)

        self.context_x_history.append(x_window)
        self.context_y_history.append(y_window)
        return True

    def record_action(self, action: torch.Tensor) -> None:
        """Record the previous action for lagged predictive context and spacing."""
        if action.dim() == 1:
            action = action.unsqueeze(0)
        self.prev_action = action.detach().clone().to(self.device)
        if self.action_dim is None:
            self.action_dim = int(self.prev_action.shape[-1])
        
        # Update spacing trackers
        # Check if action is indices (e.g., [1, 0]) or one-hot
        is_indices = (action.shape[-1] <= 3) and (action.max() < 10)
        
        bolus_taken = False
        meal_taken = False
        
        if is_indices:
            # Assume [bolus_idx, meal_idx, ...]
            if action[0, 0] > 0:
                bolus_taken = True
            if action.shape[-1] > 1 and action[0, 1] > 0:
                meal_taken = True
        
        # Note: If one-hot, we'd need action_dims to split. 
        # For now, relying on indices which is what eval_run.py passes.
        if bolus_taken:
            self.last_bolus_step = self.step_counter - 1
        if meal_taken:
            self.last_meal_step = self.step_counter - 1

    def update_state(self, obs: torch.Tensor, prev_action: Optional[torch.Tensor] = None) -> None:
        """Update internal state from observation tensor."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if prev_action is not None:
            self.record_action(prev_action)

        obs_subset = torch.cat([obs[..., :5], obs[..., -3:]], dim=-1).detach().clone().to(self.device)
        if self.action_dim is None:
            if self.prev_action is not None:
                self.action_dim = int(self.prev_action.shape[-1])

        if self.action_dim is None:
            if self.prev_obs is not None:
                self.obs_x_history.append(self.prev_obs)
                self.obs_y_history.append(obs_subset[:, :1])
        else:
            if self.prev_action is None:
                self.prev_action = torch.zeros((obs_subset.shape[0], self.action_dim), device=self.device)
            if self.prev_obs is not None:
                # Check if prev_action is already indices (small dim) or one-hot (large dim)
                if self.prev_action.shape[-1] < 5:  # Heuristic: indices are usually 2-3 dims, one-hot is >5
                     # Take first 2 dimensions (bolus, meal) and keep batch dim
                     action_indexes = self.prev_action[:, :2]
                else:
                     action_indexes = self.prev_action.view(2, -1).argmax(dim=-1, keepdim=True).transpose(0, 1)
                
                x_pair = torch.cat([self.prev_obs, action_indexes], dim=-1)
                self.obs_x_history.append(x_pair)
                self.obs_y_history.append(obs_subset[:, :1])

        self.prev_obs = obs_subset

        self.prev_bg = self.bg
        self.prev_iob = self.iob
        self.prev_cob = self.cob

        self.bg = float(obs[0, 0].item())
        self.iob = float(obs[0, 1].item())
        self.cob = float(obs[0, 2].item())
        self.bg_trend = float(obs[0, 3].item()) if obs.shape[-1] > 3 else 0.0

        if self.bg < self.params.BG_HYPO_THRESHOLD:
            if not self.in_hypo_recovery:
                self.in_hypo_recovery = True
                self.hypo_recovery_carbs_given = 0.0
        
        if self.in_hypo_recovery and self.prev_cob is not None:
            cob_increase = max(0, self.cob - self.prev_cob)
            if cob_increase > 0:
                self.hypo_recovery_carbs_given += cob_increase
        
        if self.in_hypo_recovery and self.bg > self.params.BG_HYPO_THRESHOLD + 10:
            self.in_hypo_recovery = False
            self.hypo_recovery_carbs_given = 0.0
        
        self.step_counter += 1

    def _build_model_inputs(
        self,
        bolus_level: int,
        meal_level: int,
        action_dims: List[int],
    ) -> Dict[str, torch.Tensor]:
        if not self.obs_x_history:
            raise ValueError("No predictive context available; obs_history is empty.")

        xs = list(self.context_x_history)
        ys = list(self.context_y_history)
        
        context_x = torch.stack(xs, dim=0).transpose(0, 1)
        context_y = torch.stack(ys, dim=0)

        if self.prev_obs is None:
            raise ValueError("Missing current observation; prev_obs is None.")

        past_len = self.history_len - 1
        if len(self.obs_x_history) < past_len:
            raise ValueError("Insufficient history for query; obs_x_history too short.")

        past_x = torch.stack(list(self.obs_x_history)[-past_len:], dim=0)
        action_idx = torch.tensor(
            [[bolus_level, meal_level]],
            device=self.device,
            dtype=self.prev_obs.dtype,
        )
        current_x = torch.cat([self.prev_obs.to(self.device), action_idx], dim=-1)
        query_seq = torch.cat([past_x, current_x.unsqueeze(0)], dim=0).transpose(0, 1)
        query_x = query_seq.unsqueeze(1)

        input_mean = self.input_mean[:1]
        input_std = self.input_std[:1].clamp_min(1e-6)
        output_std = self.output_std[:1].squeeze().clamp_min(1e-6)
        context_x = (context_x - input_mean) / input_std
        query_x = (query_x - input_mean) / input_std
        context_y = context_y / output_std

        bolus_dim, meal_dim = action_dims[0], action_dims[1]
        bolus_oh = torch.zeros((1, bolus_dim), device=self.device)
        meal_oh = torch.zeros((1, meal_dim), device=self.device)
        bolus_oh[0, bolus_level] = 1.0
        meal_oh[0, meal_level] = 1.0
        action_cond = torch.cat([bolus_oh, meal_oh], dim=-1)

        return {
            "context_x": context_x,
            "context_y": context_y,
            "query_x": query_x,
            "action_cond": action_cond,
        }

    def _parse_model_output(self, model_out) -> Tuple[torch.Tensor, torch.Tensor]:
        if model_out.dim() == 1:
            mean_bg = model_out
            std_bg = torch.zeros_like(mean_bg)
            return mean_bg, std_bg

        mean_bg = torch.mean(model_out, dim=0)
        std_bg = torch.std(model_out, dim=0, unbiased=False)
        return mean_bg, std_bg

    @torch.no_grad()
    def _predict_bg_bounds(
        self,
        bolus_level: int,
        meal_level: int,
        action_dims: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.obs_x_history) == 0:
            H = self.pred_cfg.horizon_steps
            z = torch.zeros((H,), device=self.device)
            return z, z, z

        inputs = self._build_model_inputs(bolus_level, meal_level, action_dims)
        context_x = inputs["context_x"]
        context_y = inputs["context_y"].squeeze(-1).unsqueeze(0)
    
        representation, _ = self.dynamics_model.compute_representation(context_x, context_y)
        query_x = inputs["query_x"]

        
        out, _ = self.dynamics_model.predict(query_x, representation, prediction_horizon=24, denormalize=True, index=self.shield_index)

        mean_bg, std_bg = self._parse_model_output(out.squeeze())
        
        k = self.pred_cfg.k_sigma
        lower = mean_bg - k * std_bg
        upper = mean_bg + k * std_bg
        return mean_bg, lower, upper

    @torch.no_grad()
    def _predict_lower_bounds_batch(
        self,
        action_pairs: List[Tuple[int, int]],
    ) -> Optional[torch.Tensor]:
        if len(self.obs_x_history) < self.history_len + self.prediction_len:
            return None
        
        if len(self.context_x_history) < self.example_nbr:
            if not self._maybe_append_context_example():
                return None
            return None

        # Batched counterfactuals assume single-env evaluation.
        if self.prev_obs is None or self.prev_obs.shape[0] != 1:
            return None

        if not action_pairs:
            return None
        
        xs = list(self.context_x_history)
        ys = list(self.context_y_history)
        context_x = torch.stack(xs, dim=0).transpose(0, 1)
        context_y = torch.stack(ys, dim=0)

        past_len = self.history_len - 1
        if len(self.obs_x_history) < past_len:
            return None

        past_x = torch.stack(list(self.obs_x_history)[-past_len:], dim=0)

        input_mean = self.input_mean[:1]
        input_std = self.input_std[:1].clamp_min(1e-6)
        output_std = self.output_std[:1].squeeze().clamp_min(1e-6)
        context_x = (context_x - input_mean) / input_std
        context_y = context_y / output_std

        query_list = []
        for bolus_level, meal_level in action_pairs:
            action_idx = torch.tensor(
                [[bolus_level, meal_level]],
                device=self.device,
                dtype=self.prev_obs.dtype,
            )
            current_x = torch.cat([self.prev_obs.to(self.device), action_idx], dim=-1)
            query_seq = torch.cat([past_x, current_x.unsqueeze(0)], dim=0).transpose(0, 1)
            query_list.append(query_seq[0].unsqueeze(0).unsqueeze(0))

        query_x = torch.cat(query_list, dim=0)
        query_x = (query_x - input_mean) / input_std

        context_y = context_y.squeeze(-1).unsqueeze(0)
        representation, _ = self.dynamics_model.compute_representation(context_x, context_y)
        if representation.shape[0] == 1 and query_x.shape[0] > 1:
            repeat_shape = [query_x.shape[0]] + [1] * (representation.dim() - 1)
            representation = representation.repeat(*repeat_shape)

        if self.shield_type == 'adolescent':
            index = 20
        elif self.shield_type == 'child':
            index = 10
        else:
            index = 0

        out, _ = self.dynamics_model.predict(
            query_x,
            representation,
            prediction_horizon=24,
            denormalize=True,
            index=index,
        )

        out = out.squeeze(-1)
        if out.dim() == 3:
            out = out[:, 0, :]
        return out

    def apply(
        self,
        obs: torch.Tensor,
        logits: torch.Tensor,
        action_dims: List[int],
        prev_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_dim is None:
            self.action_dim = int(sum(action_dims))
        self.update_state(obs, prev_action=prev_action)
        return self.apply_strict_shield_with_correction(obs, logits, action_dims)

    def apply_strict_shield_with_correction(
        self,
        obs: torch.Tensor,
        logits: torch.Tensor,
        action_dims: List[int],
    ) -> torch.Tensor:
        squeezed = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            obs = obs.unsqueeze(0)
            squeezed = True

        shield_mask = torch.zeros_like(logits)

        bolus_dim = action_dims[0]
        meal_dim = action_dims[1] if len(action_dims) > 1 else 0
        meal_start = bolus_dim
        bg_level = obs[..., 0]

        # ==================== CONSTANTS ====================
        NEG_INF = -1.0e9
        penalty = float(self.pred_cfg.logit_penalty)

        # ==================== CRITICAL RESCUE (FORCE ACTION) ====================
        if float(bg_level[0].item()) < self.params.BG_CRITICAL_HYPO:
            if meal_dim > 1:
                rescue_cooldown_steps = max(1, int(round(self.params.RESCUE_COOLDOWN_MIN / 5.0)))
                allow_rescue = True
                if self.last_rescue_step is not None:
                    if (self.step_counter - self.last_rescue_step) < rescue_cooldown_steps:
                        allow_rescue = False
                if allow_rescue:
                    rescue_mask = torch.full_like(
                        shield_mask[..., meal_start : meal_start + meal_dim],
                        -penalty,
                    )
                    rescue_mask[..., self.params.RESCUE_MEAL_LEVEL] = penalty
                    shield_mask[..., meal_start : meal_start + meal_dim] = rescue_mask

                    if bolus_dim > 0:
                        bolus_mask = torch.full_like(shield_mask[..., :bolus_dim], -penalty)
                        bolus_mask[..., 0] = penalty
                        shield_mask[..., :bolus_dim] = bolus_mask

                    self.last_rescue_step = self.step_counter
                    result = logits + shield_mask
                    return result.squeeze(0) if squeezed else result

        # Soft cap: discourage larger bolus levels in the safe BG range.
        bg_value = float(bg_level[0].item())
        if 70.0 <= bg_value <= 180.0 and bolus_dim > 2:
            shield_mask[..., 2:bolus_dim] = -penalty

        # ==================== APPLY SHIELD RULES ====================
        # Dynamic safety cap based on future predictions.
        unsafe_bolus = set()
        unsafe_meal = set()
        if self.dynamics_model is not None:
            # CRITICAL: Worst-case over meals for hypo safety.
            bolus_logits = logits[..., :bolus_dim]
            k = min(self.pred_cfg.top_k_bolus_levels, bolus_dim)
            topk = torch.topk(bolus_logits, k, dim=-1).indices
            candidates = topk[0].tolist() if topk.dim() == 2 else topk.tolist()
            if len(action_dims) > 1 and action_dims[1] > 1:
                meal_candidates = list(range(action_dims[1]))
            else:
                meal_candidates = [0]
            action_pairs = [
                (level, meal_level)
                for level in candidates
                for meal_level in meal_candidates
            ]
            preds = self._predict_lower_bounds_batch(action_pairs)
            if preds is not None:
                min_preds = preds.min(dim=1).values
                max_preds = preds.max(dim=1).values
                for idx, (level, meal_level) in enumerate(action_pairs):
                    if min_preds[idx] < self.pred_cfg.hypo_check_threshold:
                        unsafe_bolus.add(level)
                    if (
                        self.pred_cfg.use_meal_hyper_check
                        and max_preds[idx] > self.pred_cfg.hyper_check_threshold
                    ):
                        unsafe_meal.add(meal_level)
        
        # Minimal intervention: only block if the model flags a risk.
        if self.params.MIN_INTERVENTION_BG_LOW < float(bg_level[0].item()) < self.params.MIN_INTERVENTION_BG_HIGH:
            if not unsafe_bolus and not unsafe_meal:
                return logits.squeeze(0) if squeezed else logits

        if bolus_dim > 1 and unsafe_bolus:
            for lvl in unsafe_bolus:
                if lvl >= 1:
                    shield_mask[..., lvl] = -penalty

        if meal_dim > 1 and unsafe_meal:
            for lvl in unsafe_meal:
                if lvl >= 1:
                    shield_mask[..., meal_start + lvl] = -penalty

        result = logits + shield_mask
        return result.squeeze(0) if squeezed else result
