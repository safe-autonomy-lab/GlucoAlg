"""Experimental finite-penalty intervention using an explicitly supplied predictor.

This interface does not load a trained model or establish predictive safety.
"""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Sequence, Union

import torch

from shield.predictor import (
    ActionIndices,
    ForecastRequest,
    ForecastUnavailable,
    GLUCOSIM_OBSERVATION_FEATURES,
    ObservedTransition,
    PatientIdentity,
    PointForecast,
    Predictor,
    PredictorMetadata,
    RawObservation,
)


@dataclass(frozen=True)
class ShieldParams:
    """Experimental intervention thresholds, preserved from the legacy rule."""

    BG_CRITICAL_HYPO: float = 60.0
    MIN_INTERVENTION_BG_LOW: float = 90.0
    MIN_INTERVENTION_BG_HIGH: float = 160.0
    RESCUE_MEAL_LEVEL: int = 1
    RESCUE_COOLDOWN_MIN: float = 60.0


@dataclass(frozen=True)
class PredictiveShieldConfig:
    """Point-forecast checks; finite penalties do not exclude actions.

    forecast_penalty_scale multiplies only the legacy forecast contribution
    beyond the static mask. Zero retains forecasts but applies static rules;
    one preserves the legacy rule. Static caps and rescue are not scaled.
    """

    hypo_check_threshold: float = 80.0
    hyper_check_threshold: float = 250.0
    top_k_bolus_levels: int = 2
    logit_penalty: float = 10.0
    use_meal_hyper_check: bool = False
    use_forecast: bool = True
    forecast_penalty_scale: float = 1.0


@dataclass(frozen=True)
class ShieldDecision:
    """Immutable explanation of one successful legacy-soft decision.

    Masks contain ten component-logit offsets in bolus/meal order. The static
    mask is the same rule with empty forecast-risk sets; prediction_mask is
    final_mask minus static_mask, including a forecast-triggered cap. Candidate
    extrema/flags are empty unless an actual point forecast was available.
    Changed flags compare logits, not sampled or simulator-accepted actions.
    """

    step_index: int
    current_cgm: float
    use_forecast: bool
    forecast_status: str
    forecast_reason: Optional[str]
    static_mask: tuple[float, ...]
    prediction_mask: tuple[float, ...]
    final_mask: tuple[float, ...]
    candidate_actions: tuple[ActionIndices, ...]
    forecast_min_cgm: tuple[float, ...]
    forecast_max_cgm: tuple[float, ...]
    candidate_below_threshold: tuple[bool, ...]
    candidate_above_threshold: tuple[bool, ...]
    flagged_bolus_levels: tuple[int, ...]
    flagged_meal_levels: tuple[int, ...]
    static_reasons: tuple[str, ...]
    prediction_reasons: tuple[str, ...]
    static_changed: bool
    prediction_changed: bool
    final_changed: bool


def _observation(value: torch.Tensor) -> RawObservation:
    value = torch.as_tensor(value).detach()
    size = len(GLUCOSIM_OBSERVATION_FEATURES)
    if value.shape not in ((size,), (1, size)) or value.is_complex():
        raise ValueError(f'observation must have shape ({size},) or (1, {size})')
    if not torch.isfinite(value).all().item():
        raise ValueError('observation must contain only finite raw values')
    return tuple(float(x) for x in value.cpu().reshape(-1).tolist())


def _action(value: Sequence[int], name: str) -> ActionIndices:
    value = torch.as_tensor(value).detach()
    if value.shape not in ((2,), (1, 2)) or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f'{name} must contain two integer indices in (bolus, meal) order')
    numbers = value.cpu().reshape(-1).tolist()
    if any(not math.isfinite(x) or int(x) != x or not 0 <= x < 5 for x in numbers):
        raise ValueError(f'{name} indices must be integers in [0, 4]')
    return (int(numbers[0]), int(numbers[1]))


class Shield:
    """Own causal raw history and apply the legacy experimental soft rule.

    Use apply(pre_observation), step the environment, then record_action with
    actual executed indices and next_observation. Recommendations alone do
    not establish a completed transition. Only one environment is supported.
    """

    def __init__(
        self,
        *,
        predictor: Optional[Predictor] = None,
        patient: Optional[PatientIdentity] = None,
        device: Union[str, torch.device] = 'cpu',
        controller_interval_minutes: float = 5.0,
        config: Optional[PredictiveShieldConfig] = None,
        params: Optional[ShieldParams] = None,
        shield_type: Optional[str] = None,
    ) -> None:
        if shield_type is not None or predictor is None:
            raise ValueError('Experimental predictive shielding requires an explicitly injected predictor '
                             'and PatientIdentity; legacy shield_type model loading is unsupported')
        if not isinstance(patient, PatientIdentity):
            raise ValueError('patient must be a full PatientIdentity, including diabetes type and cohort#id')
        if not isinstance(getattr(predictor, 'metadata', None), PredictorMetadata):
            raise ValueError('predictor.metadata must be PredictorMetadata')
        if not callable(getattr(predictor, 'forecast', None)) or not callable(getattr(predictor, 'reset', None)):
            raise TypeError('predictor must implement forecast(request) and reset()')
        if isinstance(controller_interval_minutes, bool) or not math.isfinite(
            controller_interval_minutes
        ) or controller_interval_minutes <= 0:
            raise ValueError('controller_interval_minutes must be finite and positive')
        if controller_interval_minutes != predictor.metadata.controller_interval_minutes:
            raise ValueError('shield and predictor controller intervals must match')
        if config is not None and not isinstance(config, PredictiveShieldConfig):
            raise TypeError('config must be PredictiveShieldConfig')
        if params is not None and not isinstance(params, ShieldParams):
            raise TypeError('params must be ShieldParams')
        self.predictor = predictor
        self.metadata = predictor.metadata
        self.patient = patient
        # Resolve aliases such as cpu:0 or cuda to the actual selected device.
        # The default allocates only an empty CPU tensor, with no GPU probing.
        self.device = torch.empty(0, device=torch.device(device)).device
        self.controller_interval_minutes = controller_interval_minutes
        self.params = params if params is not None else ShieldParams()
        self.pred_cfg = config if config is not None else PredictiveShieldConfig()
        self._validate_rule_config()
        self._history = deque(maxlen=self.metadata.history_length)
        self.reset()

    def _validate_rule_config(self) -> None:
        for name in ('hypo_check_threshold', 'hyper_check_threshold', 'logit_penalty'):
            value = getattr(self.pred_cfg, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        scale = self.pred_cfg.forecast_penalty_scale
        if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale < 0:
            raise ValueError('forecast_penalty_scale must be finite and nonnegative')
        if type(self.pred_cfg.top_k_bolus_levels) is not int or self.pred_cfg.top_k_bolus_levels < 1:
            raise ValueError('top_k_bolus_levels must be a positive integer')
        for name in ('use_meal_hyper_check', 'use_forecast'):
            if type(getattr(self.pred_cfg, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name in ('BG_CRITICAL_HYPO', 'MIN_INTERVENTION_BG_LOW',
                     'MIN_INTERVENTION_BG_HIGH', 'RESCUE_COOLDOWN_MIN'):
            value = getattr(self.params, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.params.MIN_INTERVENTION_BG_LOW >= self.params.MIN_INTERVENTION_BG_HIGH:
            raise ValueError('minimal intervention lower threshold must be below upper threshold')
        if type(self.params.RESCUE_MEAL_LEVEL) is not int or not 1 <= self.params.RESCUE_MEAL_LEVEL < 5:
            raise ValueError('RESCUE_MEAL_LEVEL must be an integer in [1, 4]')

    def reset(self) -> None:
        """Clear history, pending transition, cooldown and predictor episode state."""
        self._history.clear()
        self._current_observation = None
        self._awaiting_transition = False
        self.step_counter = 0
        self.last_rescue_step = None
        self.last_forecast = ForecastUnavailable('episode has no forecast yet')
        self.last_decision: Optional[ShieldDecision] = None
        self.predictor.reset()

    def record_action(
        self,
        executed_action: Sequence[int],
        *,
        next_observation: torch.Tensor,
        recommended_action: Optional[Sequence[int]] = None,
        accepted: Optional[Sequence[bool]] = None,
    ) -> None:
        """Append a completed transition with explicit actual execution.

        Action indices and optional acceptance flags both use (bolus, meal)
        order. Flags are diagnostics: this method never infers execution from
        a recommendation or acceptance flag. Outcome data are copied.
        """
        if not self._awaiting_transition:
            raise ValueError('record_action requires a preceding apply and may complete it only once')
        action = _action(executed_action, 'executed_action')
        outcome = _observation(next_observation)
        recommendation = None if recommended_action is None else _action(
            recommended_action, 'recommended_action'
        )
        flags = None
        if accepted is not None:
            flags = tuple(accepted)
            if len(flags) != 2 or any(type(x) is not bool for x in flags):
                raise ValueError('accepted must contain two booleans in (bolus, meal) order')
        self._history.append(ObservedTransition(
            pre_observation=self._current_observation,
            executed_action=action,
            next_observation=outcome,
            recommended_action=recommendation,
            accepted=flags,
        ))
        self._current_observation = outcome
        self._awaiting_transition = False
        self.step_counter += 1

    def _forecast(self, candidates: tuple) -> Union[PointForecast, ForecastUnavailable]:
        if not self.pred_cfg.use_forecast:
            result = ForecastUnavailable('disabled ablation')
        elif len(self._history) < self.metadata.history_length:
            result = ForecastUnavailable(
                f'warmup: {len(self._history)}/{self.metadata.history_length} completed transitions'
            )
        else:
            request = ForecastRequest(
                patient=self.patient,
                past_transitions=tuple(self._history),
                current_observation=self._current_observation,
                candidate_actions=candidates,
                device=self.device,
            )
            with torch.no_grad():
                result = self.predictor.forecast(request)
            if isinstance(result, PointForecast):
                values = result.glucose_mg_dl
                expected = (len(candidates), self.metadata.horizon_steps)
                if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected:
                    raise ValueError(f'point forecast must be a tensor with shape {expected}')
                if not values.is_floating_point() or not torch.isfinite(values).all().item():
                    raise ValueError('point forecast must contain finite floating-point glucose in mg/dL')
                if values.device != self.device:
                    raise ValueError('point forecast device must match the explicit shield device')
                result = PointForecast(values.detach().clone())
            elif not isinstance(result, ForecastUnavailable):
                raise TypeError('predictor must return PointForecast or ForecastUnavailable')
        self.last_forecast = result
        return result

    def apply(self, obs: torch.Tensor, logits: torch.Tensor, action_dims: Sequence[int]) -> torch.Tensor:
        """Adjust single-environment logits; proposed candidates are not executed history."""
        observation = _observation(obs)
        if tuple(action_dims) != self.metadata.action_dims or any(type(x) is not int for x in action_dims):
            raise ValueError('action_dims must match predictor metadata: (5, 5)')
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) not in ((10,), (1, 10)):
            raise ValueError('logits must have shape (10,) or (1, 10)')
        if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
            raise ValueError('logits must contain finite floating-point values')
        if logits.device != self.device:
            raise ValueError('logits device must match the explicit shield device')
        if self._current_observation is not None and observation != self._current_observation:
            raise ValueError('observation changed without the matching record_action outcome; '
                             'complete the transition or reset for a new episode')
        if self._awaiting_transition:
            raise ValueError('previous apply still requires record_action before another apply')
        if self.pred_cfg.logit_penalty > torch.finfo(logits.dtype).max:
            raise ValueError('logit_penalty must be representable in the logits dtype')
        self._current_observation = observation
        result = self._apply_rules(logits)
        self._awaiting_transition = True
        return result

    def _apply_rules(self, logits: torch.Tensor) -> torch.Tensor:
        squeezed = logits.ndim == 1
        rows = logits.unsqueeze(0) if squeezed else logits
        mask = torch.zeros_like(rows)
        penalty = float(self.pred_cfg.logit_penalty)
        # Feature zero is measured CGM, not a latent simulator BG observation.
        bg = self._current_observation[0]
        if bg < self.params.BG_CRITICAL_HYPO:
            cooldown = max(1, int(round(
                self.params.RESCUE_COOLDOWN_MIN / self.controller_interval_minutes
            )))
            if self.last_rescue_step is None or self.step_counter - self.last_rescue_step >= cooldown:
                mask.fill_(-penalty)
                mask[..., 0] = penalty
                mask[..., 5 + self.params.RESCUE_MEAL_LEVEL] = penalty
                result = self._apply_mask(rows, mask, squeezed)
                decision = self._decision(
                    logits, result, result, mask, mask, 'critical_rescue',
                    'critical rescue rule bypassed forecast', static_reasons=('critical_rescue',),
                    prediction_reasons=('critical_rescue_bypass',),
                )
                self.last_rescue_step = self.step_counter
                self.last_forecast = ForecastUnavailable('critical rescue rule bypassed forecast')
                self.last_decision = decision
                return result

        if 70.0 <= bg <= 180.0:
            mask[..., 2:5] = -penalty
        # Evaluate the same legacy static rules with empty forecast-risk sets.
        # Inside the minimal-intervention window these discard the earlier cap.
        minimal_window = self.params.MIN_INTERVENTION_BG_LOW < bg < self.params.MIN_INTERVENTION_BG_HIGH
        static_mask = torch.zeros_like(mask) if minimal_window else mask.clone()
        static_result = logits if minimal_window else self._apply_mask(rows, static_mask, squeezed)
        static_reasons = ('minimal_intervention_bypass',) if minimal_window else (
            ('bolus_cap',) if 70.0 <= bg <= 180.0 else ('outside_static_window',)
        )
        if bg < self.params.BG_CRITICAL_HYPO:
            static_reasons += ('rescue_cooldown',)
        k = min(self.pred_cfg.top_k_bolus_levels, 5)
        levels = torch.topk(rows[..., :5], k, dim=-1).indices[0].tolist()
        candidates = tuple((level, meal) for level in levels for meal in range(5))
        forecast = self._forecast(candidates)
        unsafe_bolus, unsafe_meal = set(), set()
        minima, maxima, below, above = (), (), (), ()
        if isinstance(forecast, PointForecast):
            minima = tuple(forecast.glucose_mg_dl.min(dim=1).values.tolist())
            maxima = tuple(forecast.glucose_mg_dl.max(dim=1).values.tolist())
            below = tuple(value < self.pred_cfg.hypo_check_threshold for value in minima)
            above = tuple(value > self.pred_cfg.hyper_check_threshold for value in maxima)
            for i, (bolus, meal) in enumerate(candidates):
                if below[i]:
                    unsafe_bolus.add(bolus)
                if self.pred_cfg.use_meal_hyper_check and above[i]:
                    unsafe_meal.add(meal)
        # Preserve the legacy rule, including its removal of the earlier cap
        # within this window when no candidate prediction flags a risk.
        if minimal_window and not unsafe_bolus and not unsafe_meal:
            mask = torch.zeros_like(mask)
        else:
            for level in unsafe_bolus:
                if level >= 1:
                    mask[..., level] = -penalty
            for level in unsafe_meal:
                if level >= 1:
                    mask[..., 5 + level] = -penalty
        # Scale the mask contribution, never differences of rounded logits.
        # Delay the legacy addition: it might overflow even when a zero or
        # reduced contribution is representable. Exact zero/one branches keep
        # the static control and the legacy default unchanged.
        scale = float(self.pred_cfg.forecast_penalty_scale)
        if scale == 0:
            mask, result = static_mask, static_result
        elif scale == 1:
            result = logits if minimal_window and not unsafe_bolus and not unsafe_meal else (
                self._apply_mask(rows, mask, squeezed)
            )
        elif torch.equal(mask, static_mask):
            result = static_result
        else:
            mask = static_mask + (mask - static_mask) * scale
            result = self._apply_mask(rows, mask, squeezed)
        if isinstance(forecast, PointForecast):
            status, reason = 'available', None
            prediction_reasons = ()
            if unsafe_bolus:
                prediction_reasons += ('hypo_risk',)
            if unsafe_meal:
                prediction_reasons += ('hyper_risk',)
            if minimal_window and (unsafe_bolus or unsafe_meal) and 70.0 <= bg <= 180.0:
                prediction_reasons += ('forecast_reenabled_bolus_cap',)
            if not prediction_reasons:
                prediction_reasons = ('no_candidate_risk',)
        else:
            status = 'disabled' if not self.pred_cfg.use_forecast else (
                'warmup' if len(self._history) < self.metadata.history_length else 'unavailable'
            )
            reason, prediction_reasons = forecast.reason, (status,)
        self.last_decision = self._decision(
            logits, static_result, result, static_mask, mask, status, reason,
            candidates=candidates, minima=minima, maxima=maxima, below=below, above=above,
            unsafe_bolus=unsafe_bolus, unsafe_meal=unsafe_meal,
            static_reasons=static_reasons, prediction_reasons=prediction_reasons,
        )
        return result

    def _decision(self, logits, static_result, result, static_mask, final_mask, status, reason,
                  *, candidates=(), minima=(), maxima=(), below=(), above=(),
                  unsafe_bolus=(), unsafe_meal=(), static_reasons=(), prediction_reasons=()):
        def offsets(mask):
            return tuple(float(value) for value in mask.detach().cpu().reshape(-1).tolist())

        return ShieldDecision(
            step_index=self.step_counter, current_cgm=self._current_observation[0],
            use_forecast=self.pred_cfg.use_forecast, forecast_status=status, forecast_reason=reason,
            static_mask=offsets(static_mask), prediction_mask=offsets(final_mask - static_mask),
            final_mask=offsets(final_mask), candidate_actions=tuple(candidates),
            forecast_min_cgm=minima, forecast_max_cgm=maxima,
            candidate_below_threshold=below, candidate_above_threshold=above,
            flagged_bolus_levels=tuple(sorted(unsafe_bolus)), flagged_meal_levels=tuple(sorted(unsafe_meal)),
            static_reasons=static_reasons, prediction_reasons=prediction_reasons,
            static_changed=not torch.equal(logits, static_result),
            prediction_changed=not torch.equal(static_result, result),
            final_changed=not torch.equal(logits, result),
        )

    @staticmethod
    def _apply_mask(rows: torch.Tensor, mask: torch.Tensor, squeezed: bool) -> torch.Tensor:
        result = rows + mask
        if not torch.isfinite(result).all().item():
            raise ValueError('adjusted logits must remain finite; penalty overflowed the logits dtype')
        return result.squeeze(0) if squeezed else result
