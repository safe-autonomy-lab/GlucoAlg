"""Synthetic architectural checks: these do not establish simulator effectiveness."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from glucoalg.dynamics.model import (
    DynamicsModel, ModelConfig, architecture_spec, atomic_json, load_predictor,
    save_artifact, sha256_file,
)
from glucoalg.dynamics.train import build_parser
from shield.predictor import PatientIdentity


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(mode='direct'):
    torch.manual_seed(41)
    return DynamicsModel(ModelConfig(history_length=2, horizon_steps=2, context_size=2,
                                     n_basis=2, hidden_size=8, direct_hidden_size=8,
                                     prediction_mode=mode), np.zeros(14), np.ones(14)).eval()


def inputs():
    context = torch.zeros(1, 2, 2, 24)
    context[..., 14] = context[..., 19] = 1
    query = context.clone()
    query[:, 1, -1, 19] = 0
    query[:, 1, -1, 23] = 1  # Current recommendation meal4 instead of meal0.
    target = torch.tensor([[[0., 0.], [2., 4.]]])
    return context, torch.zeros(1, 2, 2), query, target


def test_legacy_zero_context_structurally_annihilates_data_gradients():
    network = model('context_only')
    cx, cy, qx, target = inputs()
    prediction, _ = network(cx, cy, qx)
    assert torch.count_nonzero(prediction) == 0
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in network.parameters())


@pytest.mark.parametrize('mode', ['direct', 'residual'])
def test_global_path_has_action_sensitivity_and_learns_with_zero_context_labels(mode):
    network = model(mode)
    cx, cy, qx, target = inputs()
    qx.requires_grad_()
    parts = network.loss(cx, cy, qx, target, return_components=True)
    parts['total'].backward()
    assert torch.count_nonzero(qx.grad[:, :, -1, [19, 23]]) > 0
    action_columns = network.direct_head[1].weight.grad[:, [-5, -1]]
    assert torch.isfinite(action_columns).all() and torch.count_nonzero(action_columns) > 0
    optimizer = torch.optim.Adam(network.parameters(), lr=.02)
    initial = network.loss(cx, cy, qx.detach(), target).item()
    for _ in range(20):
        optimizer.zero_grad()
        network.loss(cx, cy, qx.detach(), target).backward()
        optimizer.step()
    final = network.loss(cx, cy, qx.detach(), target).item()
    assert final < initial * .8
    forecast, _ = network(cx, cy, qx.detach())
    assert not torch.allclose(forecast[:, 0], forecast[:, 1])


def test_direct_has_no_fe_and_ignores_context_values():
    network = model()
    cx, cy, qx, target = inputs()
    assert not hasattr(network, 'function_encoder')
    prediction, gram = network(cx, cy, qx)
    changed, _ = network(cx + 100, cy - 500, qx)
    assert gram is None
    torch.testing.assert_close(prediction, changed, rtol=0, atol=0)
    parts = network.loss(cx, cy, qx, target, return_components=True)
    assert parts['basis_regularization'] == parts['direct_auxiliary'] == 0
    torch.testing.assert_close(parts['total'], parts['prediction'])


def test_residual_subtraction_and_query_addition_are_both_differentiable(monkeypatch):
    network = model('residual')
    cx, cy, qx, _ = inputs()
    outputs = []
    handle = network.direct_head.register_forward_hook(lambda module, args, result: outputs.append(result))
    captured = {}
    original = network.function_encoder.compute_representation
    def observe(x, y, **kwargs):
        captured['targets'] = y
        return original(x, y, **kwargs)
    monkeypatch.setattr(network.function_encoder, 'compute_representation', observe)
    prediction, _ = network(cx, cy, qx)
    handle.remove()
    direct_query, direct_context = outputs
    torch.testing.assert_close(captured['targets'], cy - direct_context)
    context_gradient, query_gradient = torch.autograd.grad(prediction.sum(), (direct_context, direct_query))
    assert torch.count_nonzero(context_gradient) > 0
    torch.testing.assert_close(query_gradient, torch.ones_like(direct_query))


def test_direct_auxiliary_identifies_baseline_even_if_context_cancels_it(monkeypatch):
    network = model('residual')
    cx, cy, qx, _ = inputs()
    with torch.no_grad():
        for parameter in network.direct_head.parameters():
            parameter.zero_()
        network.direct_head[-1].bias.fill_(2)
    def fit(x, y, **kwargs):
        return y.mean(dim=1), torch.eye(2).expand(x.shape[0], 2, 2)
    def predict(x, coefficients, **kwargs):
        return coefficients[:, None].expand(x.shape[0], x.shape[1], 2)
    monkeypatch.setattr(network.function_encoder, 'compute_representation', fit)
    monkeypatch.setattr(network.function_encoder, 'predict', predict)
    target = torch.ones(1, 2, 2)
    parts = network.loss(cx, cy, qx, target, return_components=True)
    assert parts['prediction'] == parts['direct_auxiliary'] == 1
    assert parts['total'] == 2
    parts['total'].backward()
    torch.testing.assert_close(network.direct_head[-1].bias.grad, torch.ones(2))


@pytest.mark.parametrize('mode', ['context_only', 'direct', 'residual'])
def test_modes_roundtrip_and_legacy_artifact_remains_exact(tmp_path, mode):
    network = model(mode)
    scope = {'supported_patients': [asdict(PatientIdentity('t1d', 'adolescent#001'))]}
    artifact = save_artifact(tmp_path, network, scope=scope, continuation_policy='unit fixture', provenance={}, training={})
    if mode == 'context_only':
        assert 'prediction_mode' not in artifact
        assert set(artifact['config']) == {'history_length', 'horizon_steps', 'context_size', 'n_basis', 'hidden_size', 'ridge_lambda', 'basis_regularization'}
        assert not any('direct_head' in key for key in network.state_dict())
    else:
        assert artifact['prediction_mode'] == artifact['config']['prediction_mode'] == mode
    state = torch.random.get_rng_state().clone()
    loaded = load_predictor(tmp_path).model
    assert torch.equal(state, torch.random.get_rng_state())
    cx, cy, qx, _ = inputs()
    cy += 1  # Check nonzero legacy predictions too.
    torch.testing.assert_close(loaded(cx, cy, qx)[0], network(cx, cy, qx)[0], rtol=0, atol=0)
    assert loaded.config.required_transitions == 4
    if mode == 'context_only':
        coefficients, _ = network.function_encoder.compute_representation(
            network._normalize(cx), cy / network.target_scale, prediction_horizon=2, lambd=.001)
        reference = network.function_encoder.predict(network._normalize(qx), coefficients, prediction_horizon=2)
        torch.testing.assert_close(loaded(cx, cy, qx)[0], reference, rtol=0, atol=0)
    else:
        artifact.pop('prediction_mode')
        atomic_json(tmp_path / 'artifact.json', artifact)
        atomic_json(tmp_path / 'artifact.sha256', {'sha256': sha256_file(tmp_path / 'artifact.json')})
        with pytest.raises(ValueError, match='prediction_mode'):
            load_predictor(tmp_path)


@pytest.mark.parametrize('changes', [
    {'prediction_mode': 'magic'}, {'direct_hidden_size': 0}, {'direct_hidden_size': True},
    {'residual_direct_weight': -1}, {'residual_direct_weight': float('nan')},
])
def test_mode_configuration_validation(changes):
    with pytest.raises(ValueError):
        ModelConfig(**changes)


def test_new_cli_fields_map_to_complete_model_config():
    args = build_parser().parse_args(['--prediction-mode', 'residual', '--direct-hidden-size', '64', '--residual-direct-weight', '1'])
    config = ModelConfig(**{key: getattr(args, key) for key in ModelConfig.__dataclass_fields__})
    assert config.prediction_mode == 'residual'
    assert architecture_spec(config)['direct_head']['hidden_size'] == 64
    json.dumps(asdict(config), allow_nan=False)
