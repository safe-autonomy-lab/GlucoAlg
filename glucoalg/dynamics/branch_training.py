"""Prefix-group training helpers; future outcomes enter targets only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import build_context_query


@dataclass(frozen=True)
class BranchWindow:
    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    targets: tuple[torch.Tensor, ...]  # Ragged observed horizons, never padded.
    control_index: int
    horizon_steps: int


def prepare_branch_windows(groups, config):
    windows = []
    for group in groups:
        candidates = np.asarray(group.candidates)
        if candidates.shape != (25, 2) or set(map(tuple, candidates.tolist())) != {(b, m) for b in range(5) for m in range(5)}:
            raise ValueError('branch group must contain every joint candidate exactly once')
        if len(group.branches) != len(candidates):
            raise ValueError('branch outcomes must match candidate count')
        prefix = group.prefix
        anchor = len(prefix['recommended_actions'])
        if anchor != group.metadata['anchor'] or any(group.metadata[key] != getattr(config, key)
                for key in ('history_length', 'horizon_steps', 'context_size')):
            raise ValueError('branch anchor/history/horizon/context differs from recorded model inputs')
        arrays = build_context_query(
            prefix['observations'], prefix['recommended_actions'], query_index=anchor,
            history_length=config.history_length, horizon_steps=config.horizon_steps,
            context_size=config.context_size, candidate_actions=candidates,
        )
        inputs = tuple(torch.tensor(value, dtype=torch.float32).unsqueeze(0) for value in
                       (arrays.context_x, arrays.context_y, arrays.query_x))
        targets = []
        for branch in group.branches:
            delta = np.asarray(branch['glucose_delta'])
            if delta.ndim != 1 or not 1 <= len(delta) <= config.horizon_steps or not np.isfinite(delta).all():
                raise ValueError('branch target must contain finite observed horizons only')
            observed = np.asarray(branch['observations'])
            if observed.shape != (len(delta) + 1, 14) or not np.array_equal(observed[0], prefix['observations'][-1]):
                raise ValueError('branch must begin at the exact common prefix observation')
            if not np.array_equal(delta, observed[1:, 0] - observed[0, 0]):
                raise ValueError('branch cumulative target disagrees with observed CGM')
            targets.append(torch.tensor(delta, dtype=torch.float32))
        control = next(i for i, pair in enumerate(candidates) if not np.any(pair))
        windows.append(BranchWindow(*inputs, tuple(targets), control, config.horizon_steps))
    if not windows:
        raise ValueError('branch split contains no observed groups')
    return windows


def branch_errors(prediction, window):
    """Ragged per-candidate errors within an anchor; paired errors exclude [0,0]."""
    if prediction.shape != (25, window.horizon_steps) or not torch.isfinite(prediction).all():
        raise ValueError('branch prediction must be finite [25,horizon]')
    absolute, response, zero_response = [], [], []
    control = window.control_index
    for index, target in enumerate(window.targets):
        absolute.append(prediction[index, :len(target)] - target)
        if index != control:
            length = min(len(target), len(window.targets[control]))
            realized = target[:length] - window.targets[control][:length]
            response.append(prediction[index, :length] - prediction[control, :length] - realized)
            zero_response.append(realized)
    if not response:
        raise ValueError('branch validation has no jointly observed response targets')
    return tuple(tuple(values) for values in (absolute, response, zero_response))


def _candidate_mean(values, *, squared=False):
    # Horizons within candidate, candidates within anchor, anchors across data.
    return torch.stack([value.square().mean() if squared else value.abs().mean()
                        for value in values]).mean()


def branch_loss(model, window):
    prediction, _ = model(window.context_x, window.context_y, window.query_x)
    absolute, response, _ = branch_errors(prediction[0] * model.target_scale, window)
    # Exactly one equal-weight prefix group per factual minibatch. Both terms
    # use the same TRAIN-only target scale as the factual cumulative targets.
    return {'absolute': _candidate_mean(absolute, squared=True) / model.target_scale.square(),
            'response': _candidate_mean(response, squared=True) / model.target_scale.square()}


def evaluate_branches(model, windows):
    model.eval()
    totals = {'absolute_mae': 0., 'response_mae': 0., 'zero_response_mae': 0.}
    observed = paired = early = 0
    with torch.no_grad():
        for window in windows:
            prediction, _ = model(window.context_x, window.context_y, window.query_x)
            errors = branch_errors(prediction[0] * model.target_scale, window)
            for key, values in zip(totals, errors):
                totals[key] += _candidate_mean(values).item()
            observed += sum(value.numel() for value in errors[0])
            paired += sum(value.numel() for value in errors[1])
            early += sum(len(target) < window.horizon_steps for target in window.targets)
    if not windows or not paired:
        raise ValueError('branch validation has no jointly observed response targets')
    requested = sum(25 * window.horizon_steps for window in windows)
    paired_requested = sum(24 * window.horizon_steps for window in windows)
    return {**{key: value / len(windows) for key, value in totals.items()},
            'groups': len(windows), 'candidates': len(windows) * 25,
            'observed_horizons': observed, 'requested_horizons': requested,
            'observed_group_coverage_fraction': observed / requested,
            'paired_observed_horizons': paired, 'paired_requested_horizons': paired_requested,
            'observed_group_paired_coverage_fraction': paired / paired_requested, 'early_branches': early}


def validation_score(factual, branches=None):
    if branches is None:
        return factual['mae'], {}
    factual_denominator = max(1., factual['persistence_mae'])
    response_denominator = max(1., branches['zero_response_mae'])
    return (factual['mae'] / factual_denominator + branches['response_mae'] / response_denominator,
            {'factual_denominator_mg_dl': factual_denominator,
             'response_denominator_mg_dl': response_denominator})


def check_branch_splits(train, validation, train_episodes, validation_episodes):
    """Group siblings may share reset identity only within the same partition."""
    def family(item):
        return tuple(item.metadata[key] for key in ('patient_type', 'patient_name', 'seed'))
    if not train or not validation:
        raise ValueError('training and validation branch groups must both be nonempty')
    for expected, groups in (('train', train), ('validation', validation)):
        if any(group.metadata['split'] != expected for group in groups):
            raise ValueError('branch split labels disagree with training roles')
    for key in ('group_id', 'data_sha256'):
        values = [group.metadata[key] for group in train + validation]
        if len(values) != len(set(values)):
            raise ValueError(f'branch group overlap or duplicate: {key}')
    training_families = {family(item) for item in train + train_episodes}
    validation_families = {family(item) for item in validation + validation_episodes}
    if training_families & validation_families:
        raise ValueError('branch/factual train-validation patient/reset family overlap')
    train_prefixes = {group.metadata['prefix_data_sha256'] for group in train}
    validation_prefixes = {group.metadata['prefix_data_sha256'] for group in validation}
    if train_prefixes & validation_prefixes:
        raise ValueError('branch train-validation prefix data overlap')
    return {'branch_family_overlap_count': 0, 'branch_data_overlap_count': 0,
            'train_branch_groups': len(train), 'validation_branch_groups': len(validation),
            'train_branch_families': len({family(item) for item in train}),
            'validation_branch_families': len({family(item) for item in validation})}
