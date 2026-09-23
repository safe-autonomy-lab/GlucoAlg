"""CPU training on explicit, disjoint causal episode artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from .data import build_context_query, load_episodes
from .model import (
    DynamicsModel, ModelConfig, PREDICTION_MODES, TRANSFER_POLICIES, atomic_json, fit_normalization,
    load_predictor, patient_scope, save_artifact, sha256_file,
)


def episode_identity(episode, path):
    result = {key: episode.metadata[key] for key in
              ('episode_id', 'patient_type', 'patient_name', 'seed', 'data_sha256')}
    result['path'] = str(Path(path).resolve())
    result['file_hashes'] = {name: sha256_file(Path(path) / name)
                             for name in ('episode.npz', 'metadata.json', 'manifest.json')}
    if result['file_hashes']['episode.npz'] != result['data_sha256']:
        raise ValueError('factual episode changed after loading')
    return result


def branch_identity(group, path):
    result = {key: group.metadata[key] for key in
              ('group_id', 'patient_type', 'patient_name', 'seed', 'anchor',
               'prefix_data_sha256', 'data_sha256', 'metadata_sha256', 'manifest_sha256')}
    result['path'] = str(Path(path).resolve())
    result['file_hashes'] = {name: sha256_file(Path(path) / name)
                             for name in ('data.npz', 'metadata.json', 'audit.json', 'manifest.json')}
    for name, key in (('data.npz', 'data_sha256'), ('metadata.json', 'metadata_sha256'), ('manifest.json', 'manifest_sha256')):
        if result['file_hashes'][name] != result[key]:
            raise ValueError('branch files changed after loading')
    return result


def check_splits(train, validation):
    """Reject relabeling the same episode, same patient/reset seed, or same data."""
    if not train or not validation:
        raise ValueError('training and validation episodes must both be nonempty')
    for expected, episodes in (('train', train), ('validation', validation)):
        if any(ep.metadata['split'] != expected for ep in episodes):
            raise ValueError(f'expected only {expected} episodes')
    checks = {
        'episode_id': lambda ep: ep.metadata['episode_id'],
        'patient_seed': lambda ep: tuple(ep.metadata[key] for key in ('patient_type', 'patient_name', 'seed')),
        'data_sha256': lambda ep: ep.metadata['data_sha256'],
    }
    for name, identify in checks.items():
        train_ids = [identify(ep) for ep in train]
        val_ids = [identify(ep) for ep in validation]
        if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids):
            raise ValueError(f'train/validation split overlap or duplicate: {name}')
    return {'overlap_count': 0, 'episode_overlap_count': 0, 'patient_seed_overlap_count': 0,
            'data_hash_overlap_count': 0, 'train_episode_count': len(train),
            'validation_episode_count': len(validation), 'unit': 'whole episodes before windows'}


def episode_protocol(episode):
    meta = episode.metadata
    runtime = meta['runtime']
    return {
        'policy_checkpoint_sha256': meta['policy']['checkpoint_sha256'],
        'policy_config_sha256': meta['policy']['config_sha256'],
        'continuation_policy': meta['continuation_policy'],
        'simulator_commit': runtime.get('glucosim_commit'),
        'simulator_content_sha256': runtime.get('glucosim_content_sha256'),
        'controller_interval_minutes': meta['controller_interval_minutes'],
        'behavior': {key: meta['behavior'][key] for key in
                     ('action_mode', 'exploration_probability', 'exploration_rng')},
    }


def prepare_windows(episodes, config, *, max_windows=None):
    windows = []
    for episode in episodes:
        n = len(episode.recommended_actions)
        for anchor in range(config.required_transitions, n - config.horizon_steps + 1):
            arrays = build_context_query(
                episode.observations, episode.recommended_actions, query_index=anchor,
                history_length=config.history_length, horizon_steps=config.horizon_steps,
                context_size=config.context_size,
            )
            windows.append(tuple(np.asarray(value, dtype=np.float32) for value in
                                 (arrays.context_x, arrays.context_y, arrays.query_x, arrays.query_y)))
    if not windows:
        raise ValueError('no complete causal context/query windows in the supplied episodes')
    if max_windows is not None:
        if type(max_windows) is not int or max_windows < 1:
            raise ValueError('max_train_windows must be positive')
        indices = np.linspace(0, len(windows) - 1, min(max_windows, len(windows)), dtype=int)
        windows = [windows[index] for index in indices]
    return windows


def _batch(windows, indices):
    selected = [windows[index] for index in indices]
    values = [torch.from_numpy(np.stack([sample[i] for sample in selected])) for i in range(4)]
    # helper query_y is [H], and query_x has a single observed recommendation.
    values[3] = values[3].unsqueeze(1)
    return values


def evaluate_windows(model, windows, batch_size=16):
    model.eval()
    absolute_errors, squared_errors, baseline_errors = [], [], []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            context_x, context_y, query_x, target = _batch(windows, range(start, min(start + batch_size, len(windows))))
            prediction, _ = model(context_x, context_y, query_x)
            difference = prediction * model.target_scale - target
            absolute_errors.append(difference.abs().flatten())
            squared_errors.append(difference.square().flatten())
            baseline_errors.append(target.abs().flatten())
    return {'mae': torch.cat(absolute_errors).mean().item(),
            'rmse': torch.cat(squared_errors).mean().sqrt().item(),
            'persistence_mae': torch.cat(baseline_errors).mean().item()}


def train_model(*, train_paths, validation_paths, output_dir, simulator_root, seed=1101,
                epochs=20, batch_size=16, learning_rate=1e-3, config=None,
                transfer_policy='same-cohort', max_train_windows=None, torch_threads=1,
                train_branch_paths=None, validation_branch_paths=None, protocol_file=None,
                protocol_clarifications=None):
    from .branch_training import (
        branch_loss, check_branch_splits, evaluate_branches, prepare_branch_windows, validation_score,
    )
    from glucoalg.tuning.plan import probe_sources
    config = config if config is not None else ModelConfig()
    for name, value in (('epochs', epochs), ('batch_size', batch_size), ('torch_threads', torch_threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned32-bit integer')
    if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    train_paths, validation_paths = list(train_paths), list(validation_paths)
    train = load_episodes(train_paths, expected_split='train')
    validation = load_episodes(validation_paths, expected_split='validation')
    split_check = check_splits(train, validation)
    protocol = episode_protocol(train[0])
    if any(episode_protocol(ep) != protocol for ep in train + validation):
        raise ValueError('episodes disagree on policy, continuation, behavior or simulator provenance')
    if not protocol['simulator_commit'] or not protocol['simulator_content_sha256']:
        raise ValueError('episodes must record simulator commit and content hash')
    sources_before = probe_sources(simulator_root)
    if sources_before['simulator']['head'] != protocol['simulator_commit'] or sources_before['simulator']['content_sha256'] != protocol['simulator_content_sha256']:
        raise ValueError('current simulator source differs from collected episodes')
    scope = patient_scope(train, transfer_policy)
    supported = {(value['diabetes_type'], value['patient_name']) for value in scope['supported_patients']}
    if any((ep.metadata['patient_type'], ep.metadata['patient_name']) not in supported for ep in validation):
        raise ValueError('validation patient is outside the declared training transfer scope')
    mean, std = fit_normalization(train)
    train_windows = prepare_windows(train, config, max_windows=max_train_windows)
    validation_windows = prepare_windows(validation, config)
    train_branch_paths = list(train_branch_paths or [])
    validation_branch_paths = list(validation_branch_paths or [])
    if bool(train_branch_paths) != bool(validation_branch_paths):
        raise ValueError('training and validation branch groups must be provided together')
    train_branch_windows = validation_branch_windows = None
    train_branch_records, validation_branch_records = [], []
    experiment_protocol = None
    if train_branch_paths:
        from .branches import load_branch_groups
        if protocol_file is None:
            raise ValueError('branch training requires the sealed --protocol-file')
        experiment_protocol = {'path': str(Path(protocol_file).resolve()),
                               'sha256': sha256_file(protocol_file)}
        if protocol_clarifications is not None:
            experiment_protocol['clarifications'] = {'path': str(Path(protocol_clarifications).resolve()),
                                                     'sha256': sha256_file(protocol_clarifications)}
        train_branches = list(load_branch_groups(train_branch_paths, required_split='train'))
        validation_branches = list(load_branch_groups(validation_branch_paths, required_split='validation'))
        split_check.update(check_branch_splits(train_branches, validation_branches, train, validation))
        if any(episode_protocol(group) != protocol for group in train_branches + validation_branches):
            raise ValueError('branch continuation, policy or simulator differs from factual episodes')
        if any((group.metadata['patient_type'], group.metadata['patient_name']) not in supported
               for group in train_branches + validation_branches):
            raise ValueError('branch patient is outside the declared training transfer scope')
        train_branch_windows = prepare_branch_windows(train_branches, config)
        validation_branch_windows = prepare_branch_windows(validation_branches, config)
        train_branch_records = [branch_identity(group, group.path) for group in train_branches]
        validation_branch_records = [branch_identity(group, group.path) for group in validation_branches]
        split_check.update(train_branches=train_branch_records, validation_branches=validation_branch_records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # one claim; no resume/overwrite
    train_records = [episode_identity(ep, path) for ep, path in zip(train, train_paths)]
    validation_records = [episode_identity(ep, path) for ep, path in zip(validation, validation_paths)]
    split_check.update(train_episodes=train_records, validation_episodes=validation_records)
    settings = {'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
                'learning_rate': learning_rate, 'torch_threads': torch_threads,
                'max_train_windows': max_train_windows, 'config': asdict(config)}
    if train_branch_windows:
        settings.update(branch_sampling='one uniformly sampled prefix group per factual minibatch; all25 candidates',
                        branch_absolute_weight=1.0, branch_response_weight=1.0,
                        branch_coverage_denominator='observed groups only; full planned-anchor coverage is in collection reports',
                        validation_selection='factualMAE/max(1,persistenceMAE) + responseMAE/max(1,zeroResponseMAE)',
                        experiment_protocol=experiment_protocol)
    try:
        atomic_json(output / 'split_check.json', split_check)
        atomic_json(output / 'run.json', {'status': 'started', 'settings': settings, 'sources': sources_before})
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(torch_threads)
        rng = np.random.default_rng(seed)
        model = DynamicsModel(config, mean, std)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_score, best_epoch, best_state = float('inf'), 0, None
        epoch_metrics = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(train_windows))
            total_loss = 0.0
            component_totals = {name: 0.0 for name in ('prediction', 'basis_regularization', 'direct_auxiliary')}
            gradient_norm_total, clipped_batches, batch_count = 0.0, 0, 0
            branch_totals = {'absolute': 0.0, 'response': 0.0}
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                components = model.loss(*_batch(train_windows, indices), return_components=True)
                loss = components['total']
                if train_branch_windows:
                    branch_index = int(rng.integers(len(train_branch_windows)))
                    branch_components = branch_loss(model, train_branch_windows[branch_index])
                    loss = loss + branch_components['absolute'] + branch_components['response']
                    for name in branch_totals:
                        branch_totals[name] += branch_components[name].item()
                if not torch.isfinite(loss):
                    raise ValueError('training loss is nonfinite')
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(gradient_norm):
                    raise ValueError('training gradients are nonfinite')
                optimizer.step()
                total_loss += loss.item() * len(indices)
                for name in component_totals:
                    component_totals[name] += components[name].item() * len(indices)
                gradient_norm_total += gradient_norm.item()
                clipped_batches += int(gradient_norm.item() > 1.0)
                batch_count += 1
            validation_metrics = evaluate_windows(model, validation_windows, batch_size)
            validation_branch_metrics = evaluate_branches(model, validation_branch_windows) if validation_branch_windows else None
            selection_score, denominators = validation_score(validation_metrics, validation_branch_metrics)
            record = {'epoch': epoch, 'train_loss': total_loss / len(train_windows),
                      **{f'train_{key}_loss': value / len(train_windows) for key, value in component_totals.items()},
                      'gradient_norm_mean': gradient_norm_total / batch_count,
                      'gradient_clipped_fraction': clipped_batches / batch_count,
                      **{f'validation_{key}': value for key, value in validation_metrics.items()}}
            if validation_branch_metrics:
                record.update(validation_selection_score=selection_score,
                              **{f'validation_selection_{key}': value for key, value in denominators.items()},
                              **{f'validation_branch_{key}': value for key, value in validation_branch_metrics.items()},
                              **{f'train_branch_{key}_loss': value / batch_count for key, value in branch_totals.items()})
            epoch_metrics.append(record)
            atomic_json(output / 'epoch_metrics.json', epoch_metrics)
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
            if selection_score < best_score:
                best_score, best_epoch = selection_score, epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        training_metrics = evaluate_windows(model, train_windows, batch_size)
        validation_metrics = evaluate_windows(model, validation_windows, batch_size)
        metrics = {**{f'train_{key}': value for key, value in training_metrics.items()},
                   **{f'validation_{key}': value for key, value in validation_metrics.items()},
                   'best_epoch': best_epoch, 'epochs': epochs, 'seed': seed,
                   'train_windows': len(train_windows), 'validation_windows': len(validation_windows),
                   'overlap_count': 0}
        if validation_branch_windows:
            training_branch_metrics = evaluate_branches(model, train_branch_windows)
            validation_branch_metrics = evaluate_branches(model, validation_branch_windows)
            score, denominators = validation_score(validation_metrics, validation_branch_metrics)
            metrics.update(validation_selection_score=score,
                           **{f'validation_selection_{key}': value for key, value in denominators.items()},
                           **{f'train_branch_{key}': value for key, value in training_branch_metrics.items()},
                           **{f'validation_branch_{key}': value for key, value in validation_branch_metrics.items()})
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError('final training metrics must all be finite numbers')
        atomic_json(output / 'metrics.json', metrics)
        sources_after = probe_sources(simulator_root)
        if any(sources_before[name]['content_sha256'] != sources_after[name]['content_sha256'] for name in ('repo', 'simulator')):
            raise ValueError('runtime source content changed during training')
        for record in train_records + validation_records:
            if any(sha256_file(Path(record['path']) / name) != digest for name, digest in record['file_hashes'].items()):
                raise ValueError('factual episode inputs changed during training')
        if experiment_protocol is not None:
            if sha256_file(experiment_protocol['path']) != experiment_protocol['sha256']:
                raise ValueError('experiment protocol changed during training')
            clarification = experiment_protocol.get('clarifications')
            if clarification is not None and sha256_file(clarification['path']) != clarification['sha256']:
                raise ValueError('experiment protocol clarifications changed during training')
            for record in train_branch_records + validation_branch_records:
                if any(sha256_file(Path(record['path']) / name) != digest for name, digest in record['file_hashes'].items()):
                    raise ValueError('branch inputs changed during training')
        provenance = {**protocol, 'train_episodes': train_records, 'validation_episodes': validation_records,
                      'sources': {'before': sources_before, 'after': sources_after},
                      'report_hashes': {name: sha256_file(output / name) for name in
                                        ('metrics.json', 'split_check.json', 'epoch_metrics.json')}}
        if experiment_protocol is not None:
            provenance.update(train_branches=train_branch_records, validation_branches=validation_branch_records,
                              experiment_protocol=experiment_protocol)
        save_artifact(output, model, scope=scope, continuation_policy=protocol['continuation_policy'],
                      provenance=provenance, training={**settings, 'best_epoch': best_epoch})
        atomic_json(output / 'run.json', {'status': 'complete', 'settings': settings,
                                         'artifact_sha256': sha256_file(output / 'artifact.json')})
        return metrics
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        atomic_json(output / 'failure.json', failure)
        atomic_json(output / 'run.json', {'status': 'failed', 'settings': settings, 'error': failure})
        raise


def selftest():
    """Executable failure fixtures; this is a tooling check, not forecast validation."""
    from shield.predictor import ForecastRequest, ObservedTransition, PatientIdentity
    torch.manual_seed(9)
    torch.set_num_threads(1)
    config = ModelConfig(history_length=2, horizon_steps=2, context_size=2,
                         n_basis=2, hidden_size=8)
    observations = np.zeros((9, 14))
    observations[:, 0] = 100 + 2 * np.arange(9)
    actions = np.zeros((8, 2), dtype=np.int64)
    arrays = build_context_query(observations, actions, query_index=4,
                                history_length=2, horizon_steps=2, context_size=2)
    np.testing.assert_array_equal(arrays.context_y, [[2, 4], [2, 4]])
    np.testing.assert_array_equal(arrays.query_y, [2, 4])
    changed = observations.copy()
    changed[5:, 0] += 1000
    changed_arrays = build_context_query(changed, actions, query_index=4,
                                        history_length=2, horizon_steps=2, context_size=2)
    for key in ('context_x', 'context_y', 'query_x'):
        np.testing.assert_array_equal(getattr(arrays, key), getattr(changed_arrays, key))
    model = DynamicsModel(config, np.zeros(14), np.ones(14))
    tensors = [torch.as_tensor(value, dtype=torch.float32).unsqueeze(0) for value in
               (arrays.context_x, arrays.context_y, arrays.query_x)]
    targets = torch.as_tensor(arrays.query_y, dtype=torch.float32)[None, None]
    before = [[p.detach().clone() for p in module.parameters()]
              for module in model.function_encoder.model.dynamics_models]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.loss(*tensors, targets).backward()
    for module in model.function_encoder.model.dynamics_models:
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()), 'ODE parameter gradients disconnected'
    optimizer.step()
    for old, module in zip(before, model.function_encoder.model.dynamics_models):
        assert any(not torch.equal(a, b) for a, b in zip(old, module.parameters())), 'ODE parameters did not update'
    scope = {'supported_patients': [asdict(PatientIdentity('t1d', 'adolescent#001'))]}
    with tempfile.TemporaryDirectory(prefix='glucoalg-predictor-selftest-') as temporary:
        artifact = save_artifact(temporary, model, scope=scope, continuation_policy='selftest fixture',
                                 provenance={'purpose': 'synthetic selftest only'}, training={})
        predictor = load_predictor(temporary)
        past = tuple(ObservedTransition(tuple(observations[i]), (0, 0), tuple(observations[i + 1]), (0, 0), (False, False)) for i in range(4))
        request = ForecastRequest(PatientIdentity('t1d', 'adolescent#001'), past,
                                  tuple(observations[4]), ((0, 0), (1, 0)), torch.device('cpu'))
        prediction = predictor.forecast(request).glucose_mg_dl
        assert prediction.shape == (2, 2) and torch.isfinite(prediction).all()
        predictor.reset()
        torch.testing.assert_close(predictor.forecast(request).glucose_mg_dl, prediction)
        try:
            save_artifact(temporary, model, scope=scope, continuation_policy='fixture', provenance={}, training={})
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing artifact was overwritten')
        with (Path(temporary) / 'weights.pt').open('ab') as handle:
            handle.write(b'tampered')
        try:
            load_predictor(temporary)
        except ValueError as error:
            assert 'hash mismatch' in str(error)
        else:
            raise AssertionError('tampered weights accepted')
        assert artifact['required_transitions'] == 4
    try:
        fit_normalization([SimpleNamespace(metadata={'split': 'validation'}, observations=observations)])
    except ValueError:
        pass
    else:
        raise AssertionError('validation data fitted normalization')
    result = {'selftest_checks_passed': 8, 'dynamics_models_updated': config.n_basis}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--train', nargs='+', type=Path)
    parser.add_argument('--validation', nargs='+', type=Path)
    parser.add_argument('--train-branches', nargs='+', type=Path)
    parser.add_argument('--validation-branches', nargs='+', type=Path)
    parser.add_argument('--protocol-file', type=Path)
    parser.add_argument('--protocol-clarifications', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--simulator-root', type=Path)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--history-length', type=int, default=12)
    parser.add_argument('--horizon-steps', type=int, default=12)
    parser.add_argument('--context-size', type=int, default=5)
    parser.add_argument('--n-basis', type=int, default=3)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--ridge-lambda', type=float, default=1e-3)
    parser.add_argument('--basis-regularization', type=float, default=1.0)
    parser.add_argument('--prediction-mode', choices=PREDICTION_MODES, default='context_only')
    parser.add_argument('--direct-hidden-size', type=int, default=64)
    parser.add_argument('--residual-direct-weight', type=float, default=1.0)
    parser.add_argument('--transfer-policy', choices=TRANSFER_POLICIES, default='same-cohort')
    parser.add_argument('--max-train-windows', type=int)
    parser.add_argument('--torch-threads', type=int, default=1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if any(getattr(args, key) is None for key in ('train', 'validation', 'output_dir', 'simulator_root')):
        parser.error('--train, --validation, --output-dir and --simulator-root are required')
    config = ModelConfig(**{key: getattr(args, key) for key in ModelConfig.__dataclass_fields__})
    train_model(train_paths=args.train, validation_paths=args.validation, output_dir=args.output_dir,
                simulator_root=args.simulator_root, seed=args.seed, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate, config=config,
                transfer_policy=args.transfer_policy, max_train_windows=args.max_train_windows,
                torch_threads=args.torch_threads, train_branch_paths=args.train_branches,
                validation_branch_paths=args.validation_branches, protocol_file=args.protocol_file,
                protocol_clarifications=args.protocol_clarifications)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
