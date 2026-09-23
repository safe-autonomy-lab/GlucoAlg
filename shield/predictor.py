"""Explicit point-forecast boundary for the experimental predictive shield.

Adapters own normalization, target encoding, patient support and model state.
This module contains no model loader or dataset dependency.
"""

from dataclasses import dataclass
import math
import re
from typing import Protocol, Tuple, Union

import torch


GLUCOSIM_OBSERVATION_FEATURES = (
    'cgm', 'iob', 'cob', 'cgm_trend', 'time_sin', 'time_cos',
    'time_since_meal', 'time_since_bolus', 'planned_meal_left',
    'meal_count_norm', 'bolus_count_norm', 'time_until_meal_norm',
    'next_meal_size_norm', 'is_pre_bolus_window',
)
ActionIndices = Tuple[int, int]  # (bolus, meal)
RawObservation = Tuple[float, ...]


@dataclass(frozen=True)
class PatientIdentity:
    """Full simulator identity; an adapter must separately check patient support."""

    diabetes_type: str
    patient_name: str

    def __post_init__(self) -> None:
        if self.diabetes_type not in ('t1d', 't2d', 't2d_no_pump'):
            raise ValueError('diabetes_type must be t1d, t2d or t2d_no_pump')
        if not isinstance(self.patient_name, str) or not re.fullmatch(
            r'(child|adolescent|adult)#(00[1-9]|010)', self.patient_name
        ):
            raise ValueError('patient_name must identify a cohort and patient #001-#010, e.g. adolescent#003')


@dataclass(frozen=True)
class PredictorMetadata:
    """Declared interface dimensions, not evidence of forecast validity.

    continuation_policy describes actions after the proposed current action.
    The adapter is responsible for implementing and validating that meaning.
    """

    history_length: int
    horizon_steps: int
    continuation_policy: str
    controller_interval_minutes: float = 5.0
    observation_features: Tuple[str, ...] = GLUCOSIM_OBSERVATION_FEATURES
    action_dims: Tuple[int, int] = (5, 5)

    def __post_init__(self) -> None:
        for name in ('history_length', 'horizon_steps'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.continuation_policy, str) or not self.continuation_policy.strip():
            raise ValueError('continuation_policy must explicitly describe future actions')
        if isinstance(self.controller_interval_minutes, bool) or not math.isfinite(
            self.controller_interval_minutes
        ) or self.controller_interval_minutes <= 0:
            raise ValueError('controller_interval_minutes must be finite and positive')
        if tuple(self.observation_features) != GLUCOSIM_OBSERVATION_FEATURES:
            raise ValueError('observation_features must match the raw GlucoSim 14-feature order')
        if tuple(self.action_dims) != (5, 5) or any(type(x) is not int for x in self.action_dims):
            raise ValueError('action_dims must be (5, 5): bolus indices then meal indices')
        object.__setattr__(self, 'observation_features', tuple(self.observation_features))
        object.__setattr__(self, 'action_dims', tuple(self.action_dims))


@dataclass(frozen=True)
class ObservedTransition:
    """A completed causal transition, snapshotted as immutable raw values.

    Both action fields and accepted flags use (bolus, meal) order. Execution
    must come from the caller's authoritative simulator outcome; acceptance
    is optional diagnostic information and never used to infer execution.
    """

    pre_observation: RawObservation
    executed_action: ActionIndices
    next_observation: RawObservation
    recommended_action: Union[ActionIndices, None] = None
    accepted: Union[Tuple[bool, bool], None] = None


@dataclass(frozen=True)
class ForecastRequest:
    patient: PatientIdentity
    past_transitions: Tuple[ObservedTransition, ...]
    current_observation: RawObservation
    candidate_actions: Tuple[ActionIndices, ...]
    device: torch.device


@dataclass(frozen=True)
class PointForecast:
    """Absolute point glucose in mg/dL, shape (candidates, horizon_steps).

    Row i corresponds to request.candidate_actions[i]; column zero is the
    next controller observation, one declared controller interval ahead.
    No confidence-bound or calibrated uncertainty interpretation is implied.
    """

    glucose_mg_dl: torch.Tensor


@dataclass(frozen=True)
class ForecastUnavailable:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError('unavailable forecast requires a nonempty reason')


class Predictor(Protocol):
    metadata: PredictorMetadata

    def reset(self) -> None:
        """Clear all predictor episode state; no implicit cross-episode adaptation."""
        ...

    def forecast(self, request: ForecastRequest) -> Union[PointForecast, ForecastUnavailable]:
        """Return point forecasts on request.device or an explicit unavailable result."""
        ...
