"""Independent arithmetic for grouped, ragged paired-action supervision."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glucoalg.dynamics.branch_training import (
    branch_errors, branch_loss, check_branch_splits, evaluate_branches,
    prepare_branch_windows, validation_score,
)
from glucoalg.dynamics.model import DynamicsModel, ModelConfig, load_predictor, sha256_file


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return ModelConfig(history_length=2, horizon_steps=2, context_size=2,
                       n_basis=2, hidden_size=8, direct_hidden_size=8, prediction_mode='direct')


def group(split='train', seed=1, partial=False):
    observations = np.zeros((5, 14))
    observations[:, 0] = 100 + np.arange(5)
    candidates = np.array([(b, m) for b in range(5) for m in range(5)])
    branches = []
    for i, (bolus, meal) in enumerate(candidates):
        count = 1 if partial and i == 24 else 2
        delta = np.arange(1, count + 1) * (1. + meal - bolus)
        future = np.repeat(observations[-1:], count + 1, axis=0)
        future[1:, 0] += delta
        branches.append({'glucose_delta': delta, 'observations': future})
    metadata = {'anchor': 4, 'horizon_steps': 2, 'history_length': 2, 'context_size': 2, 'split': split, 'patient_type': 't1d',
                'patient_name': 'adolescent#001', 'seed': seed, 'group_id': f'{seed}:4',
                'data_sha256': str(seed) * 64, 'prefix_data_sha256': f'{seed+1}' * 64}
    return SimpleNamespace(prefix={'observations': observations, 'recommended_actions': np.zeros((4, 2), dtype=int)},
                           candidates=candidates, branches=tuple(branches), metadata=metadata)


def windows(partial=False):
    return prepare_branch_windows([group(partial=partial)], config())


def test_branch_inputs_do_not_depend_on_future_outcomes_or_acceptance():
    original = group()
    changed = copy.deepcopy(original)
    for branch in changed.branches:
        branch['glucose_delta'] += 1000
        branch['observations'][1:, 0] += 1000
        branch['accepted'] = np.ones((2, 2), dtype=bool)
    before, after = [prepare_branch_windows([value], config())[0] for value in (original, changed)]
    for key in ('context_x', 'context_y', 'query_x'):
        torch.testing.assert_close(getattr(before, key), getattr(after, key), rtol=0, atol=0)
    assert not torch.equal(before.targets[0], after.targets[0])


def test_ragged_response_uses_only_mutually_observed_horizons():
    window = windows(partial=True)[0]
    prediction = torch.zeros(25, 2)
    # Exact outcome prediction where observed; absurd missing value must not
    # influence loss, coverage or gradients.
    for i, target in enumerate(window.targets):
        prediction[i, :len(target)] = target
    prediction[-1, -1] = 1e20
    absolute, paired, zero = branch_errors(prediction, window)
    assert sum(v.numel() for v in absolute) == 49
    assert sum(v.numel() for v in paired) == sum(v.numel() for v in zero) == 47
    assert torch.count_nonzero(torch.cat(absolute)) == torch.count_nonzero(torch.cat(paired)) == 0
    class Oracle:
        target_scale = 1.
        def eval(self):
            return self
        def __call__(self, *args):
            return prediction[None], None
    metrics = evaluate_branches(Oracle(), [window])
    assert metrics['observed_group_coverage_fraction'] == 49 / 50
    assert metrics['observed_group_paired_coverage_fraction'] == 47 / 48
    assert metrics['early_branches'] == 1
    assert metrics['absolute_mae'] == metrics['response_mae'] == 0


def test_candidate_permutation_preserves_group_losses():
    original = group()
    changed = copy.deepcopy(original)
    order = np.arange(24, -1, -1)
    changed.candidates = changed.candidates[order]
    changed.branches = tuple(changed.branches[i] for i in order)
    left, right = [prepare_branch_windows([value], config())[0] for value in (original, changed)]
    assert right.control_index == 24
    prediction = torch.arange(50, dtype=torch.float32).reshape(25, 2)
    for a, b in zip(branch_errors(prediction, left), branch_errors(prediction[order.copy()], right)):
        torch.testing.assert_close(torch.stack([v.abs().mean() for v in a]).mean(),
                                   torch.stack([v.abs().mean() for v in b]).mean())


def test_branch_loss_has_direct_action_gradients_with_zero_context_outcomes():
    torch.manual_seed(9)
    network = DynamicsModel(config(), np.zeros(14), np.ones(14))
    window = windows()[0]
    window.context_y.zero_()
    losses = branch_loss(network, window)
    sum(losses.values()).backward()
    gradient = network.direct_head[1].weight.grad[:, -10:]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


def test_validation_score_keeps_legacy_mae_and_registered_denominator_floors():
    factual = {'mae': 2., 'persistence_mae': .5}
    assert validation_score(factual) == (2., {})
    score, denominators = validation_score(factual, {'response_mae': 6., 'zero_response_mae': 3.})
    assert score == 4.
    assert denominators == {'factual_denominator_mg_dl': 1., 'response_denominator_mg_dl': 3.}
    assert validation_score(factual, {'response_mae': .5, 'zero_response_mae': 0.})[0] == 2.5


def test_validation_averages_prefix_groups_not_observed_scalar_count():
    full, partial = windows()[0], windows(partial=True)[0]
    class Zero:
        target_scale = 1.
        def eval(self):
            return self
        def __call__(self, *args):
            return torch.zeros(1, 25, 2), None
    metrics = evaluate_branches(Zero(), [full, partial])
    a, b = [evaluate_branches(Zero(), [window])['response_mae'] for window in (full, partial)]
    assert metrics['response_mae'] == (a + b) / 2


def test_partial_candidate_has_equal_weight_within_anchor():
    window = windows(partial=True)[0]
    prediction = torch.zeros(25, 2)
    for i, target in enumerate(window.targets):
        prediction[i, :len(target)] = target
    prediction[-1, 0] += 24  # Only this one-horizon candidate has error.
    class Oracle:
        target_scale = 1.
        def eval(self):
            return self
        def __call__(self, *args):
            return prediction[None], None
    metrics = evaluate_branches(Oracle(), [window])
    assert metrics['response_mae'] == 1.  # 24/24 candidates, not 24/47 horizons.
    assert metrics['absolute_mae'] == pytest.approx(24 / 25)


@pytest.mark.parametrize('mutate', [
    lambda g: g.candidates.__setitem__(0, (4, 4)),
    lambda g: g.metadata.__setitem__('anchor', 5),
    lambda g: g.metadata.__setitem__('horizon_steps', 3),
    lambda g: g.metadata.__setitem__('history_length', 3),
    lambda g: g.metadata.__setitem__('context_size', 3),
    lambda g: g.branches[0]['glucose_delta'].__setitem__(0, np.nan),
    lambda g: g.branches[0]['observations'].__setitem__((0, 0), 999),
    lambda g: g.branches[0]['glucose_delta'].__setitem__(0, 999),
])
def test_branch_tensor_boundary_rejects_inconsistent_data(mutate):
    value = group()
    mutate(value)
    with pytest.raises(ValueError):
        prepare_branch_windows([value], config())


def test_branch_siblings_stay_together_and_factual_family_relabeling_is_rejected():
    train, validation = group(), group('validation', 2)
    assert check_branch_splits([train], [validation], [], [])['branch_family_overlap_count'] == 0
    sibling = copy.deepcopy(train)
    sibling.metadata.update(group_id='1:5', anchor=5, data_sha256='different')
    assert check_branch_splits([train, sibling], [validation], [], [])['train_branch_families'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        check_branch_splits([train, train], [validation], [], [])
    validation.metadata['seed'] = 1
    with pytest.raises(ValueError, match='family overlap'):
        check_branch_splits([train], [validation], [], [])
    validation.metadata['seed'] = 2
    with pytest.raises(ValueError, match='family overlap'):
        check_branch_splits([train], [validation], [], [SimpleNamespace(metadata=train.metadata)])
    validation.metadata['prefix_data_sha256'] = train.metadata['prefix_data_sha256']
    with pytest.raises(ValueError, match='prefix data overlap'):
        check_branch_splits([train], [validation], [], [])


@pytest.mark.parametrize('mode', ['direct', 'residual'])
def test_trainer_uses_expanded_group_paths_train_only_stats_and_registered_selection(tmp_path, monkeypatch, mode):
    from dataclasses import replace
    from glucoalg.dynamics import branches, train
    from glucoalg.tuning import plan
    protocol = {'policy': {'checkpoint_sha256': 'a' * 64, 'config_sha256': 'b' * 64},
                'runtime': {'glucosim_commit': 'c' * 40, 'glucosim_content_sha256': 'd' * 64},
                'controller_interval_minutes': 5.0, 'continuation_policy': 'fixed synthetic continuation',
                'behavior': {'action_mode': 'stochastic', 'exploration_probability': .1, 'exploration_rng': 'numpy.PCG64'}}
    factual = []
    observations = np.zeros((9, 14))
    observations[:, 0] = 100 + np.arange(9)
    for split, seed in [('train', 1), ('validation', 2)]:
        directory = tmp_path / f'factual-{split}'
        directory.mkdir()
        for name in ('episode.npz', 'metadata.json', 'manifest.json'):
            (directory / name).write_text(f'{name}:{seed}')
        factual.append(SimpleNamespace(observations=observations.copy(), recommended_actions=np.zeros((8, 2), dtype=int),
                        metadata={**protocol, 'split': split, 'seed': seed, 'episode_id': f'fact-{seed}',
                                  'patient_type': 't1d', 'patient_name': 'adolescent#001',
                                  'data_sha256': sha256_file(directory / 'episode.npz')}))
    groups = [group(seed=10), group(seed=11), group('validation', 20)]
    for value in groups:
        value.metadata.update(protocol)
        # Far-out branch states must not enter observation-statistics fitting.
        value.prefix['observations'][:, 0] += 1000
        for future in value.branches:
            future['observations'][:, 0] += 1000
        value.path = tmp_path / f'group-{value.metadata["seed"]}'
        value.path.mkdir()
        for name in ('data.npz', 'metadata.json', 'audit.json', 'manifest.json'):
            (value.path / name).write_text(f'{name}:{value.metadata["seed"]}')
        value.metadata.update({key: sha256_file(value.path / name) for key, name in
                               [('data_sha256', 'data.npz'), ('metadata_sha256', 'metadata.json'), ('manifest_sha256', 'manifest.json')]})
    monkeypatch.setattr(train, 'load_episodes', lambda paths, expected_split: [ep for ep in factual if ep.metadata['split'] == expected_split])
    monkeypatch.setattr(branches, 'load_branch_groups', lambda paths, required_split: [g for g in groups if g.metadata['split'] == required_split])
    monkeypatch.setattr(plan, 'probe_sources', lambda root: {'repo': {'content_sha256': 'e' * 64},
                        'simulator': {'head': 'c' * 40, 'content_sha256': 'd' * 64}})
    registered = tmp_path / 'PROTOCOL.md'
    registered.write_text('Synthetic protocol fixture, not a research run.\n')
    clarification = tmp_path / 'CLARIFICATIONS.md'
    clarification.write_text('Equal horizons then candidates then anchors.\n')
    output = tmp_path / 'model'
    arguments = dict(train_paths=[tmp_path / 'factual-train'], validation_paths=[tmp_path / 'factual-validation'],
                train_branch_paths=[tmp_path / 'expanded-train-root'], validation_branch_paths=[tmp_path / 'expanded-val-root'],
                protocol_file=registered, protocol_clarifications=clarification, output_dir=output, simulator_root=tmp_path,
                config=replace(config(), prediction_mode=mode), epochs=2, batch_size=2, max_train_windows=2)
    metrics = train.train_model(**arguments)
    loaded = load_predictor(output)
    torch.testing.assert_close(loaded.model.observation_mean[0], torch.tensor(observations[:-1, 0].mean(), dtype=torch.float32))
    assert loaded.config.prediction_mode == mode
    provenance = loaded.artifact['provenance']
    assert len(provenance['train_branches']) == 2  # One input collection expanded to two groups.
    assert {r['path'] for r in provenance['train_branches']} == {str(g.path.resolve()) for g in groups[:2]}
    assert provenance['experiment_protocol']['sha256'] == sha256_file(registered)
    assert provenance['experiment_protocol']['clarifications']['sha256'] == sha256_file(clarification)
    epochs = json.loads((output / 'epoch_metrics.json').read_text())
    assert metrics['best_epoch'] == min(epochs, key=lambda row: row['validation_selection_score'])['epoch']
    assert metrics['validation_selection_score'] == min(row['validation_selection_score'] for row in epochs)
    assert metrics['train_branch_groups'] == 2 and metrics['validation_branch_groups'] == 1
    assert metrics['validation_branch_candidates'] == 25
    for record in provenance['train_branches'] + provenance['validation_branches']:
        assert set(record['file_hashes']) == {'data.npz', 'metadata.json', 'audit.json', 'manifest.json'}
    if mode == 'direct':
        original_evaluate = train.evaluate_windows
        for index, changed_path in enumerate((tmp_path / 'factual-validation' / 'metadata.json',
                                               groups[0].path / 'audit.json', clarification)):
            original_bytes = changed_path.read_bytes()
            def mutate_input(*args, **kwargs):
                result = original_evaluate(*args, **kwargs)
                changed_path.write_bytes(original_bytes + b'changed after loading')
                return result
            monkeypatch.setattr(train, 'evaluate_windows', mutate_input)
            failed_output = tmp_path / f'drift-failure-{index}'
            with pytest.raises(ValueError, match='changed during training'):
                train.train_model(**dict(arguments, output_dir=failed_output, epochs=1))
            assert json.loads((failed_output / 'run.json').read_text())['status'] == 'failed'
            assert not (failed_output / 'artifact.json').exists()
            changed_path.write_bytes(original_bytes)


def test_actual_collector_serialization_is_compatible_with_training_tensors(tmp_path):
    from test_dynamics_branches import collect_fixture
    from glucoalg.dynamics.branches import load_branch_groups, save_branch_group
    from glucoalg.dynamics.train import branch_identity, episode_protocol
    _, (groups, _, _) = collect_fixture(terminal=19)
    paths = [save_branch_group(tmp_path / f'group-{index}', value) for index, value in enumerate(groups)]
    loaded = load_branch_groups(paths, required_split='train')
    dimensions = ModelConfig(history_length=3, horizon_steps=4, context_size=2,
                             hidden_size=8, n_basis=2, direct_hidden_size=8, prediction_mode='direct')
    prepared = prepare_branch_windows(loaded, dimensions)
    network = DynamicsModel(dimensions, np.zeros(14), np.ones(14))
    loss = sum(branch_loss(network, prepared[1]).values())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(len(target) == 2 for target in prepared[1].targets)
    assert evaluate_branches(network, prepared)['observed_group_coverage_fraction'] == .75
    for value in loaded:
        record = branch_identity(value, value.path)
        assert record['data_sha256'] == record['file_hashes']['data.npz']
        assert episode_protocol(value)['behavior']['exploration_probability'] == .1
