from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class RuleBasedShieldConfig:
    """Configuration for the static rule-based shield."""

    bg_block_threshold: float = 90.0
    bg_trend_threshold: float = 100.0
    bg_rescue_threshold: float = 70.0
    bg_high_threshold: float = 250.0
    iob_safe_threshold: float = 2.0
    rescue_meal_level: int = 1
    rescue_bolus_level: int = 1
    logit_penalty: float = 10.0
    rescue_logit_penalty: float = 1.0e9


class RuleBasedShield:
    """Static safety shield that blocks boluses under low glucose or downward trend."""

    def __init__(self, config: Optional[RuleBasedShieldConfig] = None, logit_penalty: float = 10.0):
        if config is None:
            config = RuleBasedShieldConfig(logit_penalty=logit_penalty)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self) -> None:
        """Reset any internal state (no-op for static rules)."""
        return None

    def record_action(self, action: torch.Tensor) -> None:
        """Record the previous action (no-op for static rules)."""
        return None

    def apply(
        self,
        obs: torch.Tensor,
        logits: torch.Tensor,
        action_dims: List[int],
        prev_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rule-based bolus blocking and rescue overrides to logits."""
        del prev_action
        squeezed = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            obs = obs.unsqueeze(0)
            squeezed = True

        bolus_dim = action_dims[0] if action_dims else 0
        meal_dim = action_dims[1] if len(action_dims) > 1 else 0
        meal_start = bolus_dim

        if bolus_dim <= 1 and meal_dim <= 1:
            return logits.squeeze(0) if squeezed else logits

        bg_level = float(obs[0, 0].item())
        iob_level = float(obs[0, 1].item()) if obs.shape[-1] > 1 else 0.0
        bg_trend = float(obs[0, 3].item()) if obs.shape[-1] > 3 else 0.0

        if bg_level < self.config.bg_rescue_threshold and meal_dim > 1:
            rescue_penalty = float(self.config.rescue_logit_penalty)
            shield_mask = torch.zeros_like(logits)
            meal_mask = torch.full_like(
                shield_mask[..., meal_start:meal_start + meal_dim],
                -rescue_penalty,
            )
            rescue_meal_level = min(self.config.rescue_meal_level, meal_dim - 1)
            meal_mask[..., rescue_meal_level] = rescue_penalty
            shield_mask[..., meal_start:meal_start + meal_dim] = meal_mask

            if bolus_dim > 0:
                bolus_mask = torch.full_like(shield_mask[..., :bolus_dim], -rescue_penalty)
                bolus_mask[..., 0] = rescue_penalty
                shield_mask[..., :bolus_dim] = bolus_mask

            result = logits + shield_mask
            return result.squeeze(0) if squeezed else result

        should_correct = (
            bg_level > self.config.bg_high_threshold
            and iob_level < self.config.iob_safe_threshold
        )
        if should_correct and bolus_dim > 1:
            penalty = float(self.config.logit_penalty)
            shield_mask = torch.zeros_like(logits)
            bolus_mask = torch.zeros_like(shield_mask[..., :bolus_dim])
            bolus_mask[..., 0] = -penalty
            correction_idx = min(self.config.rescue_bolus_level, bolus_dim - 1)
            bolus_mask[..., correction_idx] = penalty
            shield_mask[..., :bolus_dim] = bolus_mask
            result = logits + shield_mask
            return result.squeeze(0) if squeezed else result

        block_bolus = (
            bg_level < self.config.bg_block_threshold
            or (bg_level < self.config.bg_trend_threshold and bg_trend < 0.0)
        )
        if not block_bolus:
            return logits.squeeze(0) if squeezed else logits

        shield_mask = torch.zeros_like(logits)
        shield_mask[..., 1:bolus_dim] = -float(self.config.logit_penalty)
        result = logits + shield_mask
        return result.squeeze(0) if squeezed else result
