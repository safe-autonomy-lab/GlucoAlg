"""Small BA-NODE point predictor with causal context and portable artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from shield.predictor import (
    ForecastRequest, ForecastUnavailable, GLUCOSIM_OBSERVATION_FEATURES,
    PatientIdentity, PointForecast, PredictorMetadata,
)


ARTIFACT_SCHEMA = 'glucoalg.ba_node_predictor'
TRANSFER_POLICIES = ('seen-patients-only', 'same-cohort')
PREDICTION_MODES = ('context_only', 'direct', 'residual')
INPUT_ENCODING = 'raw pre-observation14 + recommended bolus onehot5 + recommended meal onehot5'
TARGET_ENCODING = 'CGM[t+h]-CGM[t], h=1..H; divided by pooled training CGM std'


@dataclass(frozen=True)
class ModelConfig:
    history_length: int = 12
    horizon_steps: int = 12
    context_size: int = 5
    n_basis: int = 3
    hidden_size: int = 32
    ridge_lambda: float = 1e-3
    basis_regularization: float = 1.0
    prediction_mode: str = 'context_only'
    direct_hidden_size: int = 64
    residual_direct_weight: float = 1.0

    def __post_init__(self):
        for name in ('history_length', 'horizon_steps', 'context_size', 'n_basis', 'hidden_size', 'direct_hidden_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden_size % 4:
            raise ValueError('hidden_size must be divisible by four attention heads')
        if isinstance(self.ridge_lambda, bool) or not math.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0:
            raise ValueError('ridge_lambda must be finite and positive')
        if isinstance(self.basis_regularization, bool) or not math.isfinite(self.basis_regularization) or self.basis_regularization < 0:
            raise ValueError('basis_regularization must be finite and nonnegative')
        if self.prediction_mode not in PREDICTION_MODES:
            raise ValueError('unsupported prediction_mode')
        if isinstance(self.residual_direct_weight, bool) or not math.isfinite(self.residual_direct_weight) or self.residual_direct_weight < 0:
            raise ValueError('residual_direct_weight must be finite and nonnegative')

    @property
    def required_transitions(self):
        return self.history_length + self.horizon_steps + self.context_size - 2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def architecture_spec(config):
    legacy = {'n_basis': config.n_basis, 'hidden_size': config.hidden_size,
            'ode_state_size': config.hidden_size, 'dynamics_layers': 2,
            'encoder_layers': 2, 'decoder_layers': 2, 'activation': 'silu',
            'latent_ode_dt': 0.1, 'history_encoder': 'itransformer',
            'history_encoder_hidden': config.hidden_size, 'attention_heads': 4,
            'transformer_layers': 1, 'execution': 'eager CPU RK4', 'dtype': 'float32'}
    if config.prediction_mode == 'context_only':
        return legacy  # Exact pre-mode schema and state keys remain loadable.
    result = {'prediction_mode': config.prediction_mode, 'dtype': 'float32',
              'direct_head': {'input': 'flatten normalized causal history [L,24]',
                              'hidden_size': config.direct_hidden_size, 'hidden_layers': 2,
                              'activation': 'silu', 'output': 'normalized cumulative CGM delta [H]'},
              'warmup': 'same complete causal context as context_only for comparable query sets'}
    if config.prediction_mode == 'residual':
        result.update(context_residual=legacy,
                      residual_gradient='differentiate through context baseline subtraction and query addition')
    return result


def artifact_model_type(config):
    return {'context_only': 'BA_NODE/FunctionEncoder', 'direct': 'CausalMLP',
            'residual': 'CausalMLP+BA_NODE/FunctionEncoder'}[config.prediction_mode]


def atomic_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_normalization(episodes):
    """Pool each TRAIN pre-observation once; never fit statistics on windows/holdout."""
    if not episodes or any(ep.metadata['split'] != 'train' for ep in episodes):
        raise ValueError('normalization requires nonempty training episodes only')
    values = torch.cat([torch.tensor(ep.observations[:-1], dtype=torch.float64) for ep in episodes])
    if values.ndim != 2 or values.shape[1] != 14 or not torch.isfinite(values).all():
        raise ValueError('training observations must be finite [N,14]')
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), std.float()


class DynamicsModel(torch.nn.Module):
    """Normalized cumulative deltas, with an optional global path independent of context labels."""

    def __init__(self, config: ModelConfig, observation_mean, observation_std):
        super().__init__()
        self.config = config
        mean = torch.as_tensor(observation_mean, dtype=torch.float32).clone()
        std = torch.as_tensor(observation_std, dtype=torch.float32).clone()
        if mean.shape != (14,) or std.shape != (14,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('normalization must be finite14-feature means and positive scales')
        self.register_buffer('observation_mean', mean)
        self.register_buffer('observation_std', std)
        if config.prediction_mode != 'context_only':
            self.direct_head = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=-2),
                torch.nn.Linear(config.history_length * 24, config.direct_hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(config.direct_hidden_size, config.direct_hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(config.direct_hidden_size, config.horizon_steps),
            )
        if config.prediction_mode != 'direct':
            self.function_encoder = self._make_function_encoder(config)

    @staticmethod
    def _make_function_encoder(config):
        return FunctionEncoder(
            input_size=(config.history_length, 24), output_size=(config.horizon_steps,),
            data_type='deterministic', n_basis=config.n_basis, model_type='BA_NODE',
            method='least_squares', use_residuals_method=False,
            regularization_parameter=config.basis_regularization,
            model_kwargs={
                'hidden_size': config.hidden_size, 'ode_state_size': config.hidden_size,
                'n_layers': 2, 'activation': 'silu', 'prediction_length': config.horizon_steps,
                'encoder_type': 'itransformer',
                'encoder_kwargs': {'history_length': config.history_length,
                                   'hidden_size': config.hidden_size, 'num_heads': 4, 'num_layers': 1},
            }, device='cpu',
        ).float()

    @property
    def target_scale(self):
        return self.observation_std[0]

    def _normalize(self, value):
        return torch.cat(((value[..., :14] - self.observation_mean) / self.observation_std,
                          value[..., 14:]), dim=-1)

    def _predict(self, context_x, context_y, query_x):
        config = self.config
        if context_x.ndim != 4 or tuple(context_x.shape[1:]) != (config.context_size, config.history_length, 24):
            raise ValueError('context_x must be [batch,context_size,history_length,24]')
        if tuple(context_y.shape) != (context_x.shape[0], config.context_size, config.horizon_steps):
            raise ValueError('context_y must be [batch,context_size,horizon_steps] cumulativeCGM deltas')
        if query_x.ndim != 4 or query_x.shape[0] != context_x.shape[0] or tuple(query_x.shape[2:]) != (config.history_length, 24):
            raise ValueError('query_x must be [batch,queries,history_length,24]')
        if any(not torch.isfinite(value).all() for value in (context_x, context_y, query_x)):
            raise ValueError('model inputs must be finite')
        normalized_query = self._normalize(query_x)
        direct = self.direct_head(normalized_query) if config.prediction_mode != 'context_only' else None
        gram = None
        if config.prediction_mode == 'direct':
            prediction = direct
        else:
            normalized_context = self._normalize(context_x)
            residual_targets = context_y / self.target_scale
            if config.prediction_mode == 'residual':
                # End-to-end derivative is intentional. A separately supervised
                # direct loss keeps this global predictor identified even when
                # context fitting can cancel its contribution at the query.
                residual_targets = residual_targets - self.direct_head(normalized_context)
            coefficients, gram = self.function_encoder.compute_representation(
                normalized_context, residual_targets,
                prediction_horizon=config.horizon_steps, lambd=config.ridge_lambda,
            )
            prediction = self.function_encoder.predict(
                normalized_query, coefficients, prediction_horizon=config.horizon_steps,
            )
            if direct is not None:
                prediction = prediction + direct
        if not torch.isfinite(prediction).all() or (gram is not None and not torch.isfinite(gram).all()):
            raise ValueError('model produced nonfinite predictions or context Gram matrix')
        return prediction, gram, direct

    def forward(self, context_x, context_y, query_x):
        prediction, gram, _ = self._predict(context_x, context_y, query_x)
        return prediction, gram

    def loss(self, context_x, context_y, query_x, query_y, *, return_components=False):
        prediction, gram, direct = self._predict(context_x, context_y, query_x)
        if prediction.shape != query_y.shape:
            raise ValueError('query target shape must exactly match [batch,queries,horizon]')
        if not torch.isfinite(query_y).all():
            raise ValueError('query targets must be finite cumulativeCGM deltas')
        prediction_loss = torch.nn.functional.mse_loss(prediction, query_y / self.target_scale)
        zero = prediction_loss.new_zeros(())
        regularization = (torch.diagonal(gram, dim1=-2, dim2=-1) - 1).square().mean() if gram is not None else zero
        direct_loss = torch.nn.functional.mse_loss(direct, query_y / self.target_scale) if self.config.prediction_mode == 'residual' else zero
        total = (prediction_loss + self.config.basis_regularization * regularization
                 + self.config.residual_direct_weight * direct_loss)
        components = {'total': total, 'prediction': prediction_loss,
                      'basis_regularization': regularization, 'direct_auxiliary': direct_loss}
        return components if return_components else total


def patient_scope(training_episodes, transfer_policy):
    if transfer_policy not in TRANSFER_POLICIES:
        raise ValueError('unsupported patient transfer policy')
    trained = sorted({(ep.metadata['patient_type'], ep.metadata['patient_name']) for ep in training_episodes})
    supported = trained
    if transfer_policy == 'same-cohort':
        cohorts = sorted({(kind, name.split('#')[0]) for kind, name in trained})
        supported = [(kind, f'{cohort}#{number:03d}') for kind, cohort in cohorts for number in range(1, 11)]
    return {'transfer_policy': transfer_policy,
            'training_patients': [asdict(PatientIdentity(*item)) for item in trained],
            'supported_patients': [asdict(PatientIdentity(*item)) for item in supported],
            'statistics_policy': 'pooled training observations only; no patient-specific future statistics',
            'adaptation_policy': 'least-squares representation from this episode fully observed causal context only'}


def save_artifact(directory, model, *, scope, continuation_policy, provenance, training):
    """Write once inside an already exclusively claimed training directory.

    artifact.json is published last; absence means no completed loadable model.
    """
    directory = Path(directory)
    if any((directory / name).exists() for name in ('weights.pt', 'artifact.json', 'artifact.sha256')):
        raise FileExistsError('predictor artifact already exists; use a fresh output directory')
    fd, temporary = tempfile.mkstemp(prefix='.weights.', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(model.state_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / 'weights.pt')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    config_values = asdict(model.config)
    if model.config.prediction_mode == 'context_only':
        for key in ('prediction_mode', 'direct_hidden_size', 'residual_direct_weight'):
            config_values.pop(key)
    scope = dict(scope)
    if model.config.prediction_mode == 'direct':
        scope['adaptation_policy'] = 'fixed global model; causal context labels are not used'
    elif model.config.prediction_mode == 'residual':
        scope['adaptation_policy'] = 'global prediction plus least-squares residual from fully observed causal context'
    artifact = {
        'schema': ARTIFACT_SCHEMA, 'schema_version': 1, 'model_type': artifact_model_type(model.config),
        'config': config_values, 'architecture': architecture_spec(model.config),
        'required_transitions': model.config.required_transitions,
        'observation_features': list(GLUCOSIM_OBSERVATION_FEATURES), 'action_dims': [5, 5],
        'input_encoding': INPUT_ENCODING, 'target_encoding': TARGET_ENCODING,
        'forecast_units': 'mg/dL', 'controller_interval_minutes': 5.0,
        'continuation_policy': continuation_policy, 'patient_scope': scope,
        'normalization': {'mean': model.observation_mean.tolist(), 'std': model.observation_std.tolist(),
                          'target_scale': model.target_scale.item(), 'constant_feature_scale': 1.0},
        'weights_sha256': sha256_file(directory / 'weights.pt'),
        'provenance': provenance, 'training': training,
    }
    if model.config.prediction_mode != 'context_only':
        artifact['prediction_mode'] = model.config.prediction_mode
    # Hash canonical manifest bytes separately. Both are required by the loader.
    atomic_json(directory / 'artifact.json', artifact)
    atomic_json(directory / 'artifact.sha256', {'sha256': sha256_file(directory / 'artifact.json')})
    return artifact


class BANODEPredictor:
    """No inference dataset loading or cross-episode context cache."""

    def __init__(self, model, artifact):
        self.model = model.eval()
        self.config = model.config
        self.artifact = artifact
        self.metadata = PredictorMetadata(
            history_length=self.config.required_transitions,
            horizon_steps=self.config.horizon_steps,
            continuation_policy=artifact['continuation_policy'],
        )
        self._patients = {PatientIdentity(**value) for value in artifact['patient_scope']['supported_patients']}

    def reset(self):
        # Representations are recomputed from each request's bounded raw past.
        self.model.eval()

    def forecast(self, request: ForecastRequest):
        from .data import online_context_query
        if request.patient not in self._patients:
            raise ValueError('patient is outside the artifact explicit transfer scope')
        if request.device != torch.device('cpu'):
            raise ValueError('this adapter supports explicit CPU inference only')
        if len(request.past_transitions) < self.config.required_transitions:
            return ForecastUnavailable('insufficient fully observed causal context')
        if any(item.recommended_action is None for item in request.past_transitions):
            return ForecastUnavailable('recommendation-conditioned predictor requires past recommendations')
        arrays = online_context_query(
            request.past_transitions, request.current_observation, request.candidate_actions,
            history_length=self.config.history_length, horizon_steps=self.config.horizon_steps,
            context_size=self.config.context_size,
        )
        tensors = [torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
                   for value in (arrays.context_x, arrays.context_y, arrays.query_x)]
        with torch.no_grad():
            prediction, _ = self.model(*tensors)
            absolute = float(request.current_observation[0]) + prediction[0] * self.model.target_scale
        if tuple(absolute.shape) != (len(request.candidate_actions), self.config.horizon_steps) or not torch.isfinite(absolute).all():
            raise ValueError('absolute CGM forecast must be finite with shape [candidates,horizon]')
        return PointForecast(absolute)


def load_predictor(directory, *, device='cpu'):
    """Load checked weights/config/statistics without opening any episode data."""
    if torch.device(device).type != 'cpu':
        raise ValueError('this predictor currently supports CPU only')
    directory = Path(directory)
    artifact_path = directory / 'artifact.json'
    digest = json.loads((directory / 'artifact.sha256').read_text())
    if digest != {'sha256': sha256_file(artifact_path)}:
        raise ValueError('artifact manifest hash mismatch')
    artifact = json.loads(artifact_path.read_text())
    if artifact.get('schema') != ARTIFACT_SCHEMA or artifact.get('schema_version') != 1:
        raise ValueError('unsupported predictor artifact schema')
    config = ModelConfig(**artifact['config'])
    if artifact.get('model_type') != artifact_model_type(config) or artifact.get('architecture') != architecture_spec(config):
        raise ValueError('artifact architecture does not match this adapter')
    if artifact.get('prediction_mode', 'context_only') != config.prediction_mode:
        raise ValueError('artifact prediction_mode disagrees with its configuration')
    if artifact.get('input_encoding') != INPUT_ENCODING or artifact.get('target_encoding') != TARGET_ENCODING:
        raise ValueError('artifact input/target encoding mismatch')
    if artifact.get('required_transitions') != config.required_transitions:
        raise ValueError('artifact raw context length disagrees with model dimensions')
    if artifact.get('observation_features') != list(GLUCOSIM_OBSERVATION_FEATURES) or artifact.get('action_dims') != [5, 5]:
        raise ValueError('artifact feature/action schema mismatch')
    if artifact.get('forecast_units') != 'mg/dL' or artifact.get('controller_interval_minutes') != 5.0:
        raise ValueError('artifact units/controller interval mismatch')
    if sha256_file(directory / 'weights.pt') != artifact.get('weights_sha256'):
        raise ValueError('predictor weights hash mismatch')
    norm = artifact['normalization']
    # Initializing a disposable model must not advance the caller's policy RNG.
    with torch.random.fork_rng(devices=[]):
        model = DynamicsModel(config, norm['mean'], norm['std'])
        state = torch.load(directory / 'weights.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('artifact contains nonfinite weights/statistics')
    if model.observation_mean.tolist() != norm['mean'] or model.observation_std.tolist() != norm['std'] or model.target_scale.item() != norm['target_scale']:
        raise ValueError('artifact statistics disagree with saved model buffers')
    return BANODEPredictor(model, artifact)
