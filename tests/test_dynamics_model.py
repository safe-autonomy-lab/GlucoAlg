"""Real BA-NODE gradient/artifact tests and deterministic causal failure fixtures."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glucoalg.dynamics.data import build_context_query
from glucoalg.dynamics.model import (
    DynamicsModel, ModelConfig, atomic_json, fit_normalization, load_predictor,
    patient_scope, save_artifact, sha256_file,
)
from glucoalg.dynamics.train import check_splits, prepare_windows, selftest
from shield.predictor import ForecastRequest, ForecastUnavailable, ObservedTransition, PatientIdentity


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trajectory(n=10):
    observations = np.zeros((n + 1, 14), dtype=np.float64)
    observations[:, 0] = 100 + np.arange(n + 1) ** 2
    observations[:, 1:] = np.arange(13) + np.arange(n + 1)[:, None] / 10
    actions = np.array([(i % 5, (i + 1) % 5) for i in range(n)], dtype=np.int64)
    return observations, actions


def small_model(horizon=2):
    torch.manual_seed(5)
    config = ModelConfig(history_length=2, horizon_steps=horizon, context_size=2,
                         n_basis=2, hidden_size=8)
    return DynamicsModel(config, np.arange(14) / 10, np.arange(14) + 2)


def sample(model):
    obs, actions = trajectory()
    arrays = build_context_query(obs, actions, query_index=model.config.required_transitions,
                                history_length=model.config.history_length,
                                horizon_steps=model.config.horizon_steps,
                                context_size=model.config.context_size)
    tensors = [torch.tensor(value, dtype=torch.float32).unsqueeze(0) for value in
               (arrays.context_x, arrays.context_y, arrays.query_x)]
    target = torch.tensor(arrays.query_y, dtype=torch.float32)[None, None]
    return tensors, target


def identity(split='train', seed=1, **changes):
    result = dict(split=split, seed=seed, episode_id=f'episode-{seed}',
                  patient_type='t1d', patient_name='adolescent#001', data_sha256=str(seed) * 64)
    result.update(changes)
    return SimpleNamespace(metadata=result)


def scope():
    return patient_scope([identity()], 'same-cohort')


def save(tmp_path, model=None):
    model = model or small_model()
    artifact = save_artifact(tmp_path, model, scope=scope(), continuation_policy='fixed policy fixture',
                             provenance={'purpose': 'unit fixture; not research evidence'}, training={'seed': 5})
    return model, artifact


def request(model, *, patient=None, missing_recommendation=False):
    obs, actions = trajectory()
    count = model.config.required_transitions
    past = tuple(ObservedTransition(tuple(obs[i]), (0, 0), tuple(obs[i + 1]),
                                    None if missing_recommendation else tuple(actions[i]), (False, False))
                 for i in range(count))
    return ForecastRequest(patient or PatientIdentity('t1d', 'adolescent#002'), past,
                           tuple(obs[count]), ((0, 0), (4, 3)), torch.device('cpu'))


def rewrite_manifest(directory, artifact):
    atomic_json(directory / 'artifact.json', artifact)
    atomic_json(directory / 'artifact.sha256', {'sha256': sha256_file(directory / 'artifact.json')})


def test_registered_ode_parameters_receive_gradients_and_optimizer_updates():
    model = small_model()
    before = [[parameter.detach().clone() for parameter in member.parameters()]
              for member in model.function_encoder.model.dynamics_models]
    inputs, target = sample(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.loss(*inputs, target).backward()
    for member in model.function_encoder.model.dynamics_models:
        assert all(parameter.grad is not None for parameter in member.parameters())
        assert any(torch.count_nonzero(parameter.grad) for parameter in member.parameters())
    optimizer.step()
    for previous, member in zip(before, model.function_encoder.model.dynamics_models):
        assert any(not torch.equal(old, parameter) for old, parameter in zip(previous, member.parameters()))


@pytest.mark.parametrize('horizon', [1, 2, 4])
def test_training_preserves_horizon_axis_and_finite_gradients(horizon):
    model = small_model(horizon)
    inputs, target = sample(model)
    predicted, gram = model(*inputs)
    assert predicted.shape == (1, 1, horizon)
    assert gram.shape == (1, 2, 2)
    loss = model.loss(*inputs, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_normalization_uses_training_pre_observations_once_and_rejects_validation():
    one = SimpleNamespace(metadata={'split': 'train'}, observations=np.zeros((3, 14)))
    one.observations[:, 0] = [10, 20, 9999]  # final outcome is not a training pre-observation
    two = SimpleNamespace(metadata={'split': 'train'}, observations=np.zeros((2, 14)))
    two.observations[:, 0] = [30, -9999]
    mean, std = fit_normalization([one, two])
    assert mean[0] == 20
    torch.testing.assert_close(std[0], torch.tensor(np.std([10, 20, 30]), dtype=torch.float32))
    assert torch.all(std[1:] == 1)
    one.metadata['split'] = 'validation'
    with pytest.raises(ValueError, match='training episodes only'):
        fit_normalization([one, two])


def test_offline_and_online_share_cumulative_target_and_absolute_baseline(tmp_path):
    model, _ = save(tmp_path)
    predictor = load_predictor(tmp_path)
    query = request(model)
    online = predictor.forecast(query).glucose_mg_dl
    obs, actions = trajectory()
    arrays = build_context_query(obs, actions, query_index=model.config.required_transitions,
                                history_length=2, horizon_steps=2, context_size=2,
                                candidate_actions=query.candidate_actions)
    model.eval()
    inputs = [torch.tensor(value, dtype=torch.float32).unsqueeze(0) for value in
              (arrays.context_x, arrays.context_y, arrays.query_x)]
    with torch.no_grad():
        normalized, _ = model(*inputs)
    expected = query.current_observation[0] + normalized[0] * model.target_scale
    torch.testing.assert_close(online, expected)
    assert online.shape == (2, 2)
    # The context target is endpoint minus anchor, not one-step delta or a second cumsum.
    np.testing.assert_array_equal(arrays.context_y, [[3, 8], [5, 12]])


def test_adapter_reset_and_load_do_not_change_policy_rng_or_need_training_data(tmp_path, monkeypatch):
    model, _ = save(tmp_path)
    state = torch.random.get_rng_state().clone()
    predictor = load_predictor(tmp_path)
    assert torch.equal(torch.random.get_rng_state(), state)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('inference loaded episode data'))
    forecast = predictor.forecast(request(model)).glucose_mg_dl
    predictor.reset()
    torch.testing.assert_close(predictor.forecast(request(model)).glucose_mg_dl, forecast)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_adapter_rejects_unsupported_patient_and_missing_recommendations(tmp_path):
    model, _ = save(tmp_path)
    predictor = load_predictor(tmp_path)
    with pytest.raises(ValueError, match='transfer scope'):
        predictor.forecast(request(model, patient=PatientIdentity('t2d', 'adolescent#002')))
    missing = predictor.forecast(request(model, missing_recommendation=True))
    assert isinstance(missing, ForecastUnavailable)
    assert 'recommendations' in missing.reason
    short = request(model)
    short = ForecastRequest(short.patient, short.past_transitions[1:], short.current_observation,
                            short.candidate_actions, short.device)
    assert isinstance(predictor.forecast(short), ForecastUnavailable)


def test_artifact_is_written_once_and_checks_both_manifest_and_weights(tmp_path):
    model, _ = save(tmp_path)
    with pytest.raises(FileExistsError):
        save(tmp_path, model)
    with (tmp_path / 'weights.pt').open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='weights hash mismatch'):
        load_predictor(tmp_path)
    (tmp_path / 'artifact.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        load_predictor(tmp_path)


@pytest.mark.parametrize('key, value', [
    ('target_encoding', 'absolute unscaled glucose'),
    ('input_encoding', 'post-action observation'),
    ('forecast_units', 'mmol/L'), ('controller_interval_minutes', 10),
    ('required_transitions', 1), ('observation_features', ['cgm']),
    ('action_dims', [2, 3]), ('schema_version', 999),
])
def test_rehashed_incompatible_artifact_is_rejected(tmp_path, key, value):
    _, artifact = save(tmp_path)
    artifact[key] = value
    rewrite_manifest(tmp_path, artifact)
    with pytest.raises(ValueError):
        load_predictor(tmp_path)


def test_rehashed_nonfinite_weights_and_disagreeing_statistics_are_rejected(tmp_path):
    model, artifact = save(tmp_path)
    state = model.state_dict()
    state['observation_std'][0] = 999
    torch.save(state, tmp_path / 'weights.pt')
    artifact['weights_sha256'] = sha256_file(tmp_path / 'weights.pt')
    rewrite_manifest(tmp_path, artifact)
    with pytest.raises(ValueError, match='statistics disagree'):
        load_predictor(tmp_path)
    state['observation_mean'][0] = float('nan')
    torch.save(state, tmp_path / 'weights.pt')
    artifact['weights_sha256'] = sha256_file(tmp_path / 'weights.pt')
    rewrite_manifest(tmp_path, artifact)
    with pytest.raises(ValueError, match='nonfinite'):
        load_predictor(tmp_path)


@pytest.mark.parametrize('overrides', [
    {'episode_id': 'episode-1'}, {'seed': 1}, {'data_sha256': '1' * 64},
])
def test_episode_split_relabeling_is_rejected(overrides):
    values = {'seed': 2, **overrides}
    with pytest.raises(ValueError, match='overlap'):
        check_splits([identity()], [identity('validation', **values)])


def test_same_patient_with_fresh_seed_is_allowed_but_duplicate_inputs_are_rejected():
    assert check_splits([identity()], [identity('validation', seed=2)])['overlap_count'] == 0
    with pytest.raises(ValueError, match='duplicate'):
        check_splits([identity(), identity()], [identity('validation', seed=2)])


def test_patient_scope_is_explicit_and_contains_no_positional_statistics():
    seen = patient_scope([identity()], 'seen-patients-only')
    assert seen['supported_patients'] == [asdict(PatientIdentity('t1d', 'adolescent#001'))]
    transferred = patient_scope([identity()], 'same-cohort')
    assert len(transferred['supported_patients']) == 10
    assert asdict(PatientIdentity('t1d', 'adolescent#010')) in transferred['supported_patients']


@pytest.mark.parametrize('options', [
    {'history_length': 0}, {'horizon_steps': True}, {'context_size': 0},
    {'hidden_size': 6}, {'ridge_lambda': 0}, {'ridge_lambda': float('nan')},
    {'basis_regularization': -1},
])
def test_bad_model_dimensions_and_regularization_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(**options)


def test_window_preparation_does_not_cross_episode_boundaries():
    model = small_model()
    obs, actions = trajectory(8)
    episode = SimpleNamespace(observations=obs, recommended_actions=actions)
    windows = prepare_windows([episode], model.config)
    assert len(windows) == 8 - model.config.required_transitions - model.config.horizon_steps + 1
    short = SimpleNamespace(observations=obs[:5], recommended_actions=actions[:4])
    with pytest.raises(ValueError, match='no complete causal'):
        prepare_windows([short, short], model.config)


def test_packaged_selftest_checks_failure_modes():
    assert selftest() == {'selftest_checks_passed': 8, 'dynamics_models_updated': 2}


def test_compact_trainer_completes_and_seals_reports_without_overwriting(tmp_path, monkeypatch):
    from glucoalg.dynamics import train as training
    from glucoalg.tuning import plan
    obs, actions = trajectory(8)
    episodes = []
    for split, seed in [('train', 1), ('validation', 2)]:
        item = identity(split, seed)
        item.observations = obs.copy()
        item.recommended_actions = actions.copy()
        item.metadata.update({
            'policy': {'checkpoint_sha256': 'a' * 64, 'config_sha256': 'b' * 64},
            'runtime': {'glucosim_commit': 'c' * 40, 'glucosim_content_sha256': 'd' * 64},
            'controller_interval_minutes': 5.0, 'continuation_policy': 'synthetic fixed policy fixture',
            'behavior': {'action_mode': 'stochastic', 'exploration_probability': 0.1,
                         'exploration_rng': 'numpy.PCG64'},
        })
        directory = tmp_path / split
        directory.mkdir()
        for name in ('episode.npz', 'metadata.json', 'manifest.json'):
            (directory / name).write_text(f'{name}:{seed}')
        item.metadata['data_sha256'] = sha256_file(directory / 'episode.npz')
        episodes.append(item)
    monkeypatch.setattr(training, 'load_episodes', lambda paths, expected_split:
                        [ep for ep in episodes if ep.metadata['split'] == expected_split])
    monkeypatch.setattr(plan, 'probe_sources', lambda root: {
        'repo': {'content_sha256': 'e' * 64},
        'simulator': {'head': 'c' * 40, 'content_sha256': 'd' * 64}})
    output = tmp_path / 'artifact'
    arguments = dict(train_paths=[tmp_path / 'train'], validation_paths=[tmp_path / 'validation'],
                     output_dir=output, simulator_root=tmp_path, epochs=1, batch_size=2,
                     config=small_model().config, max_train_windows=2)
    metrics = training.train_model(**arguments)
    assert metrics['epochs'] == metrics['best_epoch'] == 1
    assert metrics['train_windows'] == 2
    assert metrics['validation_windows'] == 3
    predictor = load_predictor(output)
    artifact = predictor.artifact
    assert artifact['provenance']['simulator_content_sha256'] == 'd' * 64
    for name, digest in artifact['provenance']['report_hashes'].items():
        assert sha256_file(output / name) == digest
    assert all(np.isfinite(value) for value in metrics.values())
    with pytest.raises(FileExistsError):
        training.train_model(**arguments)
    def failed_loss(*args, **kwargs):
        raise ValueError('intentional optimizer fixture failure')
    monkeypatch.setattr(DynamicsModel, 'loss', failed_loss)
    arguments['output_dir'] = tmp_path / 'failed-artifact'
    with pytest.raises(ValueError, match='intentional optimizer fixture failure'):
        training.train_model(**arguments)
    failed = json.loads((arguments['output_dir'] / 'run.json').read_text())
    assert failed['status'] == 'failed'
    assert failed['error']['type'] == 'ValueError'
    assert not (arguments['output_dir'] / 'artifact.json').exists()
    # Publication failures must also leave a failed primary status record.
    def fail_initial_publication(path, value):
        if path.name == 'run.json' and value.get('status') == 'started':
            raise TypeError('intentional initial publication failure')
        atomic_json(path, value)
    monkeypatch.setattr(training, 'atomic_json', fail_initial_publication)
    arguments['output_dir'] = tmp_path / 'publication-failure'
    with pytest.raises(TypeError, match='intentional initial publication failure'):
        training.train_model(**arguments)
    publication = json.loads((arguments['output_dir'] / 'run.json').read_text())
    assert publication['status'] == 'failed'
    assert publication['error']['type'] == 'TypeError'


def test_adapter_rejects_overflow_after_denormalization(tmp_path, monkeypatch):
    model, _ = save(tmp_path)
    predictor = load_predictor(tmp_path)
    # Normalized output and training scale2 are finite; their product is not.
    def large_prediction(*args):
        return torch.full((1, 2, 2), 3e38), torch.eye(2)[None]
    monkeypatch.setattr(predictor.model, 'forward', large_prediction)
    with pytest.raises(ValueError, match='absolute CGM forecast must be finite'):
        predictor.forecast(request(model))
