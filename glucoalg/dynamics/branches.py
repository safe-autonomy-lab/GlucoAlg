#!/usr/bin/env python3
"""Causal branch corpora from exact restored GlucoSim states.

Prefixes recommend no controller action. Each fixed candidate changes only the
first future recommendation; later recommendations follow the declared fixed
policy. Acceptance and delivered doses are labels, never forecast inputs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
from types import FunctionType

import numpy as np
import torch

from glucoalg.dynamics.data import ARRAY_NAMES, EXECUTION_SEMANTICS, FEATURES, SPLITS, sha256_file


def _canonical(value, seen=None):
    """Value fingerprint, rejecting unknown state types instead of ignoring them."""
    seen = {} if seen is None else seen
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        return {'float': float(value).hex()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (torch.device, np.dtype)):
        return {'type': type(value).__name__, 'value': str(value)}
    if isinstance(value, Enum):
        return {'enum': f'{type(value).__module__}.{type(value).__qualname__}',
                'value': _canonical(value.value, seen)}
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {'torch': str(value.dtype), 'array': _canonical(array, seen)}
    if isinstance(value, np.ndarray) or hasattr(value, '__array__'):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError('object arrays are unsupported in exact state fingerprints')
        return {'array': str(array.dtype), 'shape': list(array.shape),
                'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    if isinstance(value, np.random.Generator):
        return {'numpy_generator': _canonical(value.bit_generator.state, seen)}
    if isinstance(value, torch.Generator):
        return {'torch_generator': _canonical(value.get_state(), seen)}
    if isinstance(value, FunctionType):
        return {'function': f'{value.__module__}.{value.__qualname__}',
                'code': hashlib.sha256(value.__code__.co_code).hexdigest(),
                'defaults': _canonical(value.__defaults__, seen),
                'closure': [_canonical(cell.cell_contents, seen) for cell in (value.__closure__ or ())]}
    if id(value) in seen:
        return {'reference': seen[id(value)]}
    seen[id(value)] = len(seen)
    if isinstance(value, dict):
        return {'dict': [[_canonical(key, seen), _canonical(item, seen)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return {'sequence': f'{type(value).__module__}.{type(value).__qualname__}',
                'items': [_canonical(item, seen) for item in value]}
    if hasattr(value, '__dict__'):
        return {'object': f'{type(value).__module__}.{type(value).__qualname__}',
                'state': _canonical(vars(value), seen)}
    raise TypeError(f'unsupported simulator state type: {type(value)}')


def digest(value):
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def trace_digest(trace):
    """JSON traces are independent of object-member ordering after storage."""
    return hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def live_state(env, policy_rng, exploration):
    return {'environment': vars(env), 'policy_rng': policy_rng.get_state(),
            'exploration_rng': exploration.bit_generator.state}


def state_digest(env, policy_rng, exploration):
    return digest(live_state(env, policy_rng, exploration))


def core_environment(env):
    vector = env.env
    if len(vector.envs) != 1:
        raise ValueError('branch probe requires exactly one synchronous simulator')
    core = vector.envs[0]
    if core._jax_state is None:
        raise ValueError('branch probe requires a reset and warmed simulator')
    if np.any(vector._autoreset_envs):
        raise ValueError('cannot branch a terminated environment awaiting autoreset')
    if getattr(core, 'log_file', None) is not None:
        raise ValueError('branch probe requires disabled simulator file logging')
    return core


@dataclass
class Snapshot:
    state: dict
    sha256: str
    inventory: dict


def buffer_aliases(vector):
    arrays = [(f'_env_obs[{index}]', value) for index, value in enumerate(vector._env_obs)]
    arrays += [(name, vars(vector)[name]) for name in
               ('_observations', '_rewards', '_costs', '_terminations', '_truncations', '_autoreset_envs')]
    return [(left_name, right_name) for index, (left_name, left) in enumerate(arrays)
            for right_name, right in arrays[index + 1:]
            if np.shares_memory(np.asarray(left), np.asarray(right))]


def capture(env, policy_rng, exploration):
    core = core_environment(env)
    state = copy.deepcopy(live_state(env, policy_rng, exploration))
    aliases = buffer_aliases(env.env)
    if buffer_aliases(state['environment']['env']) != aliases:
        raise ValueError('deepcopy changed simulator buffer alias relationships')
    snapshot = Snapshot(state, digest(state), {
        'root_keys': sorted(vars(env)), 'vector_keys': sorted(vars(env.env)),
        'core_keys': sorted(vars(core)),
        'jax_state_sha256': digest(core._jax_state), 'jax_key_sha256': digest(core.key),
        'vector_buffers_sha256': digest({name: vars(env.env)[name] for name in
                                        ('_env_obs', '_observations', '_rewards', '_costs',
                                         '_terminations', '_truncations', '_autoreset_envs')}),
        'policy_rng_sha256': digest(policy_rng.get_state()),
        'exploration_rng_sha256': digest(exploration.bit_generator.state),
        'class_warmup_cache_sha256': digest(getattr(type(core), '_warmup_cache', None)),
        'buffer_aliases': aliases,
    })
    if state_digest(env, policy_rng, exploration) != snapshot.sha256:
        raise ValueError('deepcopy did not preserve the complete simulator/RNG state')
    return snapshot


def restore(env, policy_rng, exploration, snapshot):
    core = core_environment(env) if not np.any(env.env._autoreset_envs) else env.env.envs[0]
    if digest(getattr(type(core), '_warmup_cache', None)) != snapshot.inventory['class_warmup_cache_sha256']:
        raise ValueError('class warmup cache changed during branching')
    state = copy.deepcopy(snapshot.state)
    vars(env).clear()
    vars(env).update(state['environment'])
    policy_rng.set_state(state['policy_rng'])
    exploration.bit_generator.state = state['exploration_rng']
    if state_digest(env, policy_rng, exploration) != snapshot.sha256:
        raise ValueError('state restoration fingerprint mismatch')
    if buffer_aliases(env.env) != snapshot.inventory['buffer_aliases']:
        raise ValueError('restoration changed simulator buffer alias relationships')
    core_environment(env)


def recommend(actor, normalizer, observation, mode, probability, policy_rng, exploration):
    """Same categorical sampler as collection, with an owned/restorable CPU RNG."""
    from glucoalg.dynamics.collect import _recommend
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(policy_rng.get_state())
        action = _recommend(actor, normalizer, observation, mode)
        policy_rng.set_state(torch.random.get_rng_state())
    exploring = exploration.random() < probability
    if exploring:
        action = exploration.integers(0, 5, size=2, dtype=np.int64)
    return action, bool(exploring)


def step(env, recommendation):
    from glucoalg.dynamics.collect import _flag, _observation, _scalar, accepted_execution
    observation, reward, cost, terminated, truncated, info = env.step(np.asarray(recommendation, dtype=np.int64)[None].copy())
    executed, accepted = accepted_execution(recommendation, info)
    row = {'observation': _observation(observation).tolist(),
           'recommended_action': np.asarray(recommendation).tolist(),
           'executed_action': executed.tolist(), 'accepted': accepted.tolist(),
           'reward': _scalar(reward, 'reward'), 'cost': _scalar(cost, 'cost'),
           'terminated': _flag(terminated, 'terminated'), 'truncated': _flag(truncated, 'truncated'),
           'termination_cause': int(_scalar(info['termination_cause'], 'termination_cause'))}
    for key in ('meal_total_g', 'insulin_total_U', 'scenario_meal_avg'):
        if key in info:
            row[key] = _scalar(info[key], key)
    if row['cost'] < 0:
        raise ValueError('negative simulator cost')
    return row


def run_branch(env, actor, normalizer, observation, candidate, horizon, mode, probability,
               policy_rng, exploration, *, draw=recommend, step_function=step):
    trace = []
    raw = np.array(observation, copy=True)
    for index in range(horizon):
        proposed, explored = draw(actor, normalizer, raw, mode, probability, policy_rng, exploration)
        # Consume exactly the ordinary current-step RNG draws before replacing
        # the first action. Future streams start equally for every candidate.
        recommendation = np.array(candidate if index == 0 else proposed, dtype=np.int64, copy=True)
        row = step_function(env, recommendation)
        row['unforced_proposal'] = np.asarray(proposed).tolist()
        row['explored'] = explored
        trace.append(row)
        # JSON lists lose dtype. Retain the wrapper's original float32 policy
        # observation arithmetic, matching collection before normalization.
        raw = np.asarray(row['observation'], dtype=raw.dtype)
        if row['terminated'] or row['truncated']:
            break  # never step into SafetySyncVectorEnv's automatic reset
    return {'trace': trace, 'trace_sha256': trace_digest(trace),
            'end_state_sha256': state_digest(env, policy_rng, exploration),
            'length': len(trace), 'horizon': horizon, 'coverage_fraction': len(trace) / horizon}


def verify_control(first, second):
    if first['trace_sha256'] != second['trace_sha256'] or first['end_state_sha256'] != second['end_state_sha256']:
        raise ValueError('duplicate control branch failed exact trace/end-state reproduction')


def verify_episode_arrays(first, second):
    """Require identical dtype, shape and bytes, including terminal observations."""
    from glucoalg.dynamics.data import ARRAY_NAMES
    checks = {}
    for name in ARRAY_NAMES:
        left, right = np.asarray(first[name]), np.asarray(second[name])
        equal = (left.dtype == right.dtype and left.shape == right.shape
                 and left.tobytes() == right.tobytes())
        checks[name] = {'identical': equal, 'shape': list(left.shape),
                        'dtype': str(left.dtype), 'sha256': digest(left)}
        if not equal:
            raise ValueError(f'collection/unshielded-rollout bitwise mismatch: {name}')
    return checks


def collection_rollout_parity(env_factory, actor, normalizer, metadata):
    """Fresh environments, exact shared seeds; return engineering-only traces.

    The transient Episode uses the data validator's test split. It is never
    saved as a collection episode or included in fitting/held-out metrics.
    """
    from glucoalg.dynamics.collect import collect_episode
    from glucoalg.dynamics.data import ARRAY_NAMES
    from glucoalg.dynamics.rollout import rollout_episode
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first_env = second_env = None
    try:
        with torch.random.fork_rng(devices=[]):
            first_env = env_factory()
            episode = collect_episode(first_env, actor, normalizer, metadata=metadata)
            first_env.close()
            first_env = None
            second_env = env_factory()
            arrays, _, _ = rollout_episode(
                second_env, actor, normalizer, shield=None, seed=metadata['seed'],
                action_seed=metadata['action_seed'], exploration_seed=metadata['exploration_seed'],
                horizon_steps=metadata['horizon_steps'], action_mode=metadata['behavior']['action_mode'],
                exploration_probability=metadata['behavior']['exploration_probability'])
        first = {name: getattr(episode, name) for name in ARRAY_NAMES}
        second = {name: arrays[name] for name in ARRAY_NAMES}
        checks = verify_episode_arrays(first, second)
        report = {'purpose': 'engineering regression only; excluded from model datasets/results',
                  'env_seed': metadata['seed'], 'action_seed': metadata['action_seed'],
                  'exploration_seed': metadata['exploration_seed'],
                  'length': len(episode.rewards), 'all_arrays_bitwise_identical': True,
                  'arrays': checks}
        return report, first, second
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for env in (first_env, second_env):
            if env is not None:
                env.close()


def _seed(value):
    value = int(value)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError('seed must fit unsigned32 bits')
    return value


BRANCH_SCHEMA = 'glucoalg.branch_group/1'
COLLECTION_SCHEMA = 'glucoalg.branch_collection/1'
DEFAULT_ANCHORS = tuple(range(27, 268, 24))
CANDIDATES = tuple((bolus, meal) for bolus in range(5) for meal in range(5))
PREFIX_BEHAVIOR = 'no_controller_action'


@dataclass(frozen=True)
class BranchGroup:
    """One reset/anchor family; all sibling candidates share its declared split.

    ``prefix`` has the eight causal Episode array names but is an unfinished
    past prefix, not a standalone Episode. Each ``branches`` entry has the same
    names plus ``glucose_delta``, ``unforced_proposals`` and ``explored``.
    Branch observations include the common current observation at row zero.
    Short terminal branches are ragged; targets are never padded.
    """
    prefix: dict
    candidates: np.ndarray
    branches: tuple
    metadata: dict
    audit: dict
    path: Path | None = None

    def __post_init__(self):
        def arrays(values):
            result = {}
            for name, value in values.items():
                result[name] = np.array(value, copy=True)
                result[name].setflags(write=False)
            return result
        object.__setattr__(self, 'prefix', arrays(self.prefix))
        object.__setattr__(self, 'branches', tuple(arrays(branch) for branch in self.branches))
        candidates = np.array(self.candidates, copy=True)
        candidates.setflags(write=False)
        object.__setattr__(self, 'candidates', candidates)
        object.__setattr__(self, 'metadata', json.loads(json.dumps(self.metadata, allow_nan=False)))
        object.__setattr__(self, 'audit', json.loads(json.dumps(self.audit, allow_nan=False)))


def _write_json(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _check_seed(value, name):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f'{name} must be an unsigned 32-bit integer')


def _check_digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a SHA256 hex digest')


def episode_family(metadata):
    """Reset identity ignores action seed: alternate descendants cannot cross splits."""
    return (metadata['patient_type'], metadata['patient_name'], metadata['seed'])


def prefix_data_digest(prefix):
    """Hash prefix array values, shapes and dtypes independently of metadata."""
    return digest({name: np.asarray(prefix[name]) for name in ARRAY_NAMES})


def branch_data_digest(branch):
    """Stable array digest independent of dictionaries reordered by storage."""
    return digest({name: np.asarray(branch[name]) for name in
                   (*ARRAY_NAMES, 'glucose_delta', 'unforced_proposals', 'explored')})


def _trace_arrays(initial, trace, *, future=False):
    n = len(trace)
    observations = np.asarray([initial, *(row['observation'] for row in trace)], dtype=np.asarray(initial).dtype)
    arrays = {
        'observations': observations,
        'recommended_actions': np.asarray([row['recommended_action'] for row in trace], dtype=np.int64).reshape(n, 2),
        'executed_actions': np.asarray([row['executed_action'] for row in trace], dtype=np.int64).reshape(n, 2),
        'accepted': np.asarray([row['accepted'] for row in trace], dtype=bool).reshape(n, 2),
        'rewards': np.asarray([row['reward'] for row in trace], dtype=np.float64),
        'costs': np.asarray([row['cost'] for row in trace], dtype=np.float64),
        'terminated': np.asarray([row['terminated'] for row in trace], dtype=bool),
        'truncated': np.asarray([row['truncated'] for row in trace], dtype=bool),
    }
    if future:
        arrays.update(glucose_delta=observations[1:, 0] - observations[0, 0],
                      unforced_proposals=np.asarray([row['unforced_proposal'] for row in trace], dtype=np.int64).reshape(n, 2),
                      explored=np.asarray([row['explored'] for row in trace], dtype=bool))
    return arrays


def _validate_arrays(arrays, *, future=False, prefix=False):
    expected = set(ARRAY_NAMES) | ({'glucose_delta', 'unforced_proposals', 'explored'} if future else set())
    if set(arrays) != expected:
        raise ValueError('Unexpected trace array fields')
    n = len(arrays['rewards'])
    if n < 1 or arrays['observations'].shape != (n + 1, 14):
        raise ValueError('Trace observations must include reset/current and every next observation')
    for name in ('observations', 'rewards', 'costs') + (('glucose_delta',) if future else ()):
        a = arrays[name]
        if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
            raise ValueError(f'Non-finite or nonnumeric {name}')
    if arrays['rewards'].shape != (n,) or arrays['costs'].shape != (n,) or np.any(arrays['costs'] < 0):
        raise ValueError('Invalid reward/cost arrays')
    for name in ('recommended_actions', 'executed_actions') + (('unforced_proposals',) if future else ()):
        a = arrays[name]
        if a.shape != (n, 2) or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a > 4):
            raise ValueError(f'Invalid categorical {name}')
    for name, shape in [('accepted', (n, 2)), ('terminated', (n,)), ('truncated', (n,))] + ([('explored', (n,))] if future else []):
        if arrays[name].shape != shape or arrays[name].dtype.kind != 'b':
            raise ValueError(f'Invalid boolean {name}')
    if not np.array_equal(arrays['executed_actions'], np.where(arrays['accepted'], arrays['recommended_actions'], 0)):
        raise ValueError('Executed indices disagree with acceptance flags')
    ended = arrays['terminated'] | arrays['truncated']
    if np.any(ended[:-1]) or (prefix and np.any(ended)):
        raise ValueError('Cannot branch after a terminal prefix or retain post-terminal steps')
    if future:
        delta = arrays['observations'][1:, 0] - arrays['observations'][0, 0]
        if arrays['glucose_delta'].shape != (n,) or not np.array_equal(delta, arrays['glucose_delta']):
            raise ValueError('Branch targets do not equal future CGM minus anchor CGM')
    return n


def validate_branch_group(group, *, required_split=None):
    if not isinstance(group, BranchGroup):
        raise ValueError('Expected a BranchGroup')
    m = group.metadata
    required = {'schema', 'group_id', 'episode_id', 'patient_type', 'patient_name', 'seed', 'action_seed',
                'exploration_seed', 'split', 'anchor', 'horizon_steps', 'history_length', 'context_size',
                'prefix_horizon_steps', 'prefix_behavior', 'observation_features', 'execution_semantics',
                'policy', 'behavior', 'continuation_policy', 'runtime', 'first_acceptance', 'source_identity',
                'controller_interval_minutes', 'prefix_data_sha256'}
    if required - m.keys():
        raise ValueError(f'Branch metadata missing {sorted(required - m.keys())}')
    if m['schema'] != BRANCH_SCHEMA or m['prefix_behavior'] != PREFIX_BEHAVIOR:
        raise ValueError('Unsupported branch schema/prefix behavior')
    if m['split'] not in SPLITS or (required_split is not None and m['split'] != required_split):
        raise ValueError('Branch split does not match the requested split')
    if m['patient_type'] not in ('t1d', 't2d', 't2d_no_pump'):
        raise ValueError('Unsupported diabetes type')
    import re
    if not isinstance(m['patient_name'], str) or re.fullmatch(r'(child|adolescent|adult)#(00[1-9]|010)', m['patient_name']) is None:
        raise ValueError('Invalid patient identity')
    for name in ('seed', 'action_seed', 'exploration_seed'):
        _check_seed(m[name], name)
    for name in ('anchor', 'horizon_steps', 'history_length', 'context_size', 'prefix_horizon_steps'):
        _positive(m[name], name)
    needed = m['history_length'] + m['horizon_steps'] + m['context_size'] - 2
    if m['anchor'] < needed or m['anchor'] >= m['prefix_horizon_steps']:
        raise ValueError('Anchor lacks fully observed causal context or precedes no remaining prefix')
    if tuple(m['observation_features']) != FEATURES or m['execution_semantics'] != EXECUTION_SEMANTICS:
        raise ValueError('Observation or action semantics mismatch')
    if type(m['controller_interval_minutes']) not in (int, float) or m['controller_interval_minutes'] != 5:
        raise ValueError('Controller interval must be five minutes')
    for key in ('group_id', 'episode_id', 'continuation_policy'):
        if not isinstance(m[key], str) or not m[key].strip():
            raise ValueError(f'Empty {key}')
    if any(not isinstance(m[key], dict) for key in ('policy', 'behavior', 'runtime', 'source_identity')):
        raise ValueError('Policy, behavior and provenance must be mappings')
    for key in ('checkpoint', 'config'):
        if not isinstance(m['policy'].get(key), str) or not m['policy'][key]:
            raise ValueError('Missing policy paths')
        _check_digest(m['policy'].get(key + '_sha256'))
    if m['behavior'].get('action_mode') not in ('stochastic', 'deterministic'):
        raise ValueError('Invalid continuation action mode')
    probability = m['behavior'].get('exploration_probability')
    if type(probability) not in (int, float) or not 0 <= probability <= 1 or m['behavior'].get('exploration_rng') != 'numpy.PCG64':
        raise ValueError('Invalid continuation exploration')
    _check_digest(m['runtime'].get('glucosim_content_sha256'))
    if not isinstance(m['runtime'].get('glucosim_commit'), str) or not m['runtime']['glucosim_commit']:
        raise ValueError('Missing simulator commit')
    n = _validate_arrays(group.prefix, prefix=True)
    if prefix_data_digest(group.prefix) != m['prefix_data_sha256']:
        raise ValueError('Prefix array digest mismatch')
    if n != m['anchor'] or np.any(group.prefix['recommended_actions']) or np.any(group.prefix['executed_actions']):
        raise ValueError('Prefix must contain exactly the no-controller-action history to the anchor')
    expected = np.asarray(CANDIDATES, dtype=np.int64)
    if group.candidates.dtype.kind not in 'iu' or not np.array_equal(group.candidates, expected) or len(group.branches) != 25:
        raise ValueError('A group must contain all 25 ordered joint recommendations exactly once')
    if (not isinstance(m['first_acceptance'], list) or len(m['first_acceptance']) != 25
            or any(not isinstance(item, dict) or not {'accepted', 'executed_action'} <= item.keys()
                   for item in m['first_acceptance'])):
        raise ValueError('Missing first-action outcome diagnostics')
    for candidate, branch, outcome in zip(group.candidates, group.branches, m['first_acceptance']):
        length = _validate_arrays(branch, future=True)
        if length > m['horizon_steps'] or (length < m['horizon_steps'] and not (branch['terminated'][-1] or branch['truncated'][-1])):
            raise ValueError('Short branches must end at a simulator terminal event')
        if not np.array_equal(branch['observations'][0], group.prefix['observations'][-1]):
            raise ValueError('Branch starts from a different observation')
        if not np.array_equal(branch['recommended_actions'][0], candidate):
            raise ValueError('First recommendation was not forced to the declared candidate')
        if outcome['accepted'] != branch['accepted'][0].tolist() or outcome['executed_action'] != branch['executed_actions'][0].tolist():
            raise ValueError('First acceptance labels disagree with the branch trace')
    audit = group.audit
    _check_digest(audit.get('snapshot_sha256'))
    if audit.get('restored_after_sha256') != audit['snapshot_sha256'] or audit.get('duplicate_control_verified') is not True:
        raise ValueError('Missing successful restoration/duplicate-control audit')
    if audit.get('candidate_count') != 25 or len(audit.get('branches', [])) != 25:
        raise ValueError('Incomplete branch audit')
    for trace, recorded in zip(group.branches, audit['branches']):
        _check_digest(recorded['trace_sha256'])
        _check_digest(recorded['end_state_sha256'])
        if recorded['length'] != len(trace['rewards']):
            raise ValueError('Audit length disagrees with trace')
        if recorded.get('data_sha256') != branch_data_digest(trace):
            raise ValueError('Audit array digest disagrees with branch data')
    control = audit['control_repeat']
    control_index = audit['control_index']
    if type(control_index) is not int or control_index != 0:
        raise ValueError('Invalid duplicate control index')
    verify_control(audit['branches'][control_index], control)
    repeated = _trace_arrays(group.prefix['observations'][-1], control['trace'], future=True)
    if any(not np.array_equal(repeated[key], group.branches[control_index][key]) for key in repeated):
        raise ValueError('Duplicate control arrays disagree')
    if trace_digest(control['trace']) != control['trace_sha256']:
        raise ValueError('Duplicate control trace hash mismatch')
    json.dumps(m, allow_nan=False)
    json.dumps(audit, allow_nan=False)


def _pack_group(group):
    packed = {'prefix_' + key: value for key, value in group.prefix.items()}
    lengths = np.array([len(branch['rewards']) for branch in group.branches], dtype=np.int64)
    packed.update(candidate_recommendations=group.candidates,
                  branch_offsets=np.concatenate(([0], np.cumsum(lengths))),
                  observation_offsets=np.concatenate(([0], np.cumsum(lengths + 1))))
    for name in group.branches[0]:
        packed['branch_' + name] = np.concatenate([branch[name] for branch in group.branches], axis=0)
    return packed


def save_branch_group(path, group):
    validate_branch_group(group)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    with (path / 'data.npz').open('xb') as stream:
        np.savez_compressed(stream, **_pack_group(group))
    _write_json(path / 'metadata.json', group.metadata)
    _write_json(path / 'audit.json', group.audit)
    _write_json(path / 'manifest.json', {'schema': BRANCH_SCHEMA, 'group_id': group.metadata['group_id'],
                'files': {name: sha256_file(path / name) for name in ('data.npz', 'metadata.json', 'audit.json')}})
    return path


def load_branch_group(path, *, required_split=None):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema') != BRANCH_SCHEMA or set(manifest.get('files', {})) != {'data.npz', 'metadata.json', 'audit.json'}:
        raise ValueError('Unsupported or incomplete branch manifest')
    if any(sha256_file(path / name) != value for name, value in manifest['files'].items()):
        raise ValueError('Branch artifact hash mismatch')
    metadata = json.loads((path / 'metadata.json').read_text())
    metadata.update(data_sha256=manifest['files']['data.npz'], metadata_sha256=manifest['files']['metadata.json'],
                    manifest_sha256=sha256_file(path / 'manifest.json'))
    audit = json.loads((path / 'audit.json').read_text())
    with np.load(path / 'data.npz', allow_pickle=False) as loaded:
        data = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    future_names = (*ARRAY_NAMES, 'glucose_delta', 'unforced_proposals', 'explored')
    expected = {'prefix_' + key for key in ARRAY_NAMES} | {'branch_' + key for key in future_names} | {'candidate_recommendations', 'branch_offsets', 'observation_offsets'}
    if set(data) != expected:
        raise ValueError('Unexpected packed branch arrays')
    offsets, obs_offsets = data['branch_offsets'], data['observation_offsets']
    for offsets_value in (offsets, obs_offsets):
        if offsets_value.shape != (26,) or offsets_value.dtype.kind not in 'iu' or offsets_value[0] != 0 or np.any(np.diff(offsets_value) <= 0):
            raise ValueError('Invalid ragged branch offsets')
    if not np.array_equal(np.diff(obs_offsets), np.diff(offsets) + 1):
        raise ValueError('Ragged observation offsets disagree with transitions')
    for name in future_names:
        expected_length = obs_offsets[-1] if name == 'observations' else offsets[-1]
        if len(data['branch_' + name]) != expected_length:
            raise ValueError('Packed branch length disagrees with offsets')
    branches = []
    for index in range(25):
        branches.append({name: data['branch_' + name][
            (obs_offsets if name == 'observations' else offsets)[index]:
            (obs_offsets if name == 'observations' else offsets)[index + 1]] for name in future_names})
    group = BranchGroup(prefix={key: data['prefix_' + key] for key in ARRAY_NAMES},
                        candidates=data['candidate_recommendations'], branches=tuple(branches),
                        metadata=metadata, audit=audit, path=path.resolve())
    if manifest['group_id'] != metadata['group_id']:
        raise ValueError('Manifest group identity mismatch')
    validate_branch_group(group, required_split=required_split)
    return group


def load_branch_groups(paths, *, required_split=None):
    """Load group or complete collection directories; reject split/identity reuse."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    groups, identities, families, prefix_splits = [], set(), {}, {}
    group_ids, data_hashes = set(), set()
    for item in paths:
        collection_groups = []
        root = Path(item)
        if (root / 'collection.json').exists():
            collection = json.loads((root / 'collection.json').read_text())
            if collection.get('schema') != COLLECTION_SCHEMA or collection.get('status') != 'complete':
                raise ValueError('Branch collection did not complete')
            if (collection.get('split') not in SPLITS
                    or (required_split is not None and collection['split'] != required_split)):
                raise ValueError('Collection split does not match the requested split')
            base_fields = ('split', 'patient_type', 'patient_name', 'seed', 'action_seed', 'exploration_seed',
                           'prefix_behavior', 'policy')
            if any(collection.get(key) != collection.get('metadata', {}).get(key) for key in base_fields):
                raise ValueError('Collection identity/seed/protocol differs from its metadata')
            run = json.loads((root / 'run.json').read_text())
            if run.get('status') != 'complete' or run.get('collection_sha256') != sha256_file(root / 'collection.json'):
                raise ValueError('Branch collection completion hash mismatch')
            if sha256_file(root / 'prefix.npz') != collection.get('prefix_sha256'):
                raise ValueError('Collection prefix hash mismatch')
            with np.load(root / 'prefix.npz', allow_pickle=False) as loaded:
                prefix = {name: np.array(loaded[name], copy=True) for name in loaded.files}
            _validate_arrays(prefix)
            if (np.any(prefix['recommended_actions']) or np.any(prefix['executed_actions'])
                    or not (prefix['terminated'][-1] or prefix['truncated'][-1])):
                raise ValueError('Collection prefix must be a completed no-action trajectory')
            entries = collection['groups']
            directories = []
            for entry in entries:
                relative = Path(entry['path'])
                if relative.is_absolute() or len(relative.parts) != 1 or relative.parts[0] in ('.', '..'):
                    raise ValueError('Invalid branch group path')
                directory = root / relative
                if directory.resolve().parent != root.resolve() or sha256_file(directory / 'manifest.json') != entry['manifest_sha256']:
                    raise ValueError('Collection group path/hash mismatch')
                directories.append(directory)
            planned = collection['anchors_requested']
            retained = [int(directory.name.removeprefix('anchor-')) for directory in directories]
            skipped = [item['anchor'] for item in collection['skipped_anchors']]
            if (len(set(planned)) != len(planned) or len(set(retained + skipped)) != len(retained + skipped)
                    or sorted(retained + skipped) != sorted(planned)):
                raise ValueError('Planned anchors are omitted, duplicated or invented')
            prefix_length = len(prefix['rewards'])
            needed = sum(collection['metadata'][key] for key in ('history_length', 'horizon_steps', 'context_size')) - 2
            for omitted in collection['skipped_anchors']:
                anchor = omitted['anchor']
                expected_reason = ('prefix_ended' if anchor >= prefix_length else
                                   'insufficient_fully_observed_context' if anchor < needed else None)
                if expected_reason is None or omitted['reason'] != expected_reason:
                    raise ValueError('Invalid reason for skipping a planned anchor')
        else:
            directories = [root]
            collection = None
        for index, directory in enumerate(directories):
            group = load_branch_group(directory, required_split=required_split)
            if collection is not None:
                if group.metadata['split'] != collection['split']:
                    raise ValueError('Collection/group split mismatch')
                if (episode_family(group.metadata) != episode_family(collection)
                        or group.metadata['group_id'] != entries[index]['group_id']):
                    raise ValueError('Collection/group identity mismatch')
                if any(group.metadata.get(key) != value for key, value in collection['metadata'].items()):
                    raise ValueError('Collection/group provenance mismatch')
                anchor = group.metadata['anchor']
                if anchor != retained[index] or any(not np.array_equal(value, prefix[key][:(anchor + 1 if key == 'observations' else anchor)])
                                                    for key, value in group.prefix.items()):
                    raise ValueError('Group prefix differs from the continuing no-action trajectory')
            identity = (episode_family(group.metadata), group.metadata['anchor'],
                        group.metadata['action_seed'], group.metadata['exploration_seed'])
            if identity in identities:
                raise ValueError('Duplicate branch group')
            identities.add(identity)
            if group.metadata['group_id'] in group_ids or group.metadata['data_sha256'] in data_hashes:
                raise ValueError('Duplicate branch group identifier or data bytes')
            group_ids.add(group.metadata['group_id'])
            data_hashes.add(group.metadata['data_sha256'])
            family = episode_family(group.metadata)
            previous_split = families.setdefault(family, group.metadata['split'])
            if previous_split != group.metadata['split']:
                raise ValueError('Same reset episode has descendants in multiple splits')
            previous_prefix_split = prefix_splits.setdefault(group.metadata['prefix_data_sha256'], group.metadata['split'])
            if previous_prefix_split != group.metadata['split']:
                raise ValueError('Identical prefix bytes appear in multiple splits')
            groups.append(group)
            collection_groups.append(group)
        if collection is not None:
            expected_metrics = _collection_metrics(collection_groups, prefix, collection['anchors_requested'],
                collection['metadata']['horizon_steps'], collection['metadata']['prefix_horizon_steps'])
            if collection.get('metrics') != expected_metrics:
                raise ValueError('Collection coverage metrics disagree with retained trajectories')
    return tuple(groups)


def _collection_metrics(groups, prefix, anchors, horizon_steps, prefix_horizon):
    return {'anchors_requested': len(anchors), 'anchors_completed': len(groups),
            'anchors_skipped': len(anchors) - len(groups),
            'requested_candidate_branches': 25 * len(anchors),
            'requested_future_steps': 25 * len(anchors) * horizon_steps,
            'candidate_branches': 25 * len(groups), 'duplicate_controls_verified': len(groups),
            'prefix_steps': len(prefix['rewards']),
            'prefix_early_termination': int(len(prefix['rewards']) < prefix_horizon),
            'full_horizon_branches': sum(len(branch['rewards']) == horizon_steps for group in groups for branch in group.branches),
            'future_steps': sum(len(branch['rewards']) for group in groups for branch in group.branches),
            'accepted_first_bolus': sum(int(branch['accepted'][0, 0]) for group in groups for branch in group.branches),
            'accepted_first_meal': sum(int(branch['accepted'][0, 1]) for group in groups for branch in group.branches),
            'overlap_count': 0}


def collect_branch_episode(env, actor, normalizer, *, metadata, anchors=DEFAULT_ANCHORS,
                           horizon_steps=12, history_length=12, context_size=5, publish=None):
    """Collect one no-action prefix; branches never modify its continuation.

    The caller closes the environment. A publish callback can persist each
    completed group immediately. Prefix RNG streams consume ordinary policy
    and exploration draws before forcing [0,0], keeping anchor streams aligned
    with elapsed decisions without allowing branches to advance the prefix.
    """
    from glucoalg.dynamics.collect import _observation, _scalar
    anchors = tuple(anchors)
    if not anchors or any(type(anchor) is not int or anchor < 1 for anchor in anchors) or tuple(sorted(set(anchors))) != anchors:
        raise ValueError('Anchors must be distinct positive increasing integers')
    for value, name in ((horizon_steps, 'horizon_steps'), (history_length, 'history_length'), (context_size, 'context_size')):
        _positive(value, name)
    prefix_horizon = _positive(metadata['prefix_horizon_steps'], 'prefix_horizon_steps')
    if anchors[-1] > prefix_horizon:
        raise ValueError('Anchor exceeds the declared prefix horizon')
    if getattr(actor, 'shield', None) is not None:
        raise ValueError('Branch continuation requires an unshielded fixed policy')
    if tuple(np.asarray(env.action_space.nvec).tolist()) != (5, 5) or _scalar(env.sample_time, 'sample_time') != 5:
        raise ValueError('Branch collection requires 5-minute GlucoSim (5,5) actions')
    if _scalar(env.simulation_minutes, 'simulation_minutes') != 5 * prefix_horizon:
        raise ValueError('Simulator prefix horizon mismatch')
    for key in ('seed', 'action_seed', 'exploration_seed'):
        _check_seed(metadata[key], key)
    mode = metadata['behavior']['action_mode']
    probability = metadata['behavior']['exploration_probability']
    if mode not in ('stochastic', 'deterministic') or type(probability) not in (int, float) or not 0 <= probability <= 1:
        raise ValueError('Invalid fixed continuation behavior')
    actor_before = digest(actor.state_dict())
    normalization_before = digest(vars(normalizer)) if normalizer is not None else None
    policy_rng = torch.Generator(device='cpu').manual_seed(metadata['action_seed'])
    exploration = np.random.Generator(np.random.PCG64(metadata['exploration_seed']))
    observation, _ = env.reset(seed=metadata['seed'])
    initial = current = _observation(observation)
    trace, groups, skipped = [], [], []
    needed = history_length + horizon_steps + context_size - 2
    ended = False
    for t in range(prefix_horizon + 1):
        if t in anchors:
            if ended:
                skipped.append({'anchor': t, 'reason': 'prefix_ended'})
            elif t < needed:
                skipped.append({'anchor': t, 'reason': 'insufficient_fully_observed_context', 'required_steps': needed})
            else:
                snapshot = capture(env, policy_rng, exploration)
                branches = []
                try:
                    control_index = 0  # Protocol-fixed duplicate [0,0], independent of outcomes.
                    for candidate in CANDIDATES:
                        restore(env, policy_rng, exploration, snapshot)
                        branches.append(run_branch(env, actor, normalizer, current, candidate, horizon_steps,
                                                   mode, probability, policy_rng, exploration))
                    restore(env, policy_rng, exploration, snapshot)
                    repeat = run_branch(env, actor, normalizer, current, CANDIDATES[control_index], horizon_steps,
                                        mode, probability, policy_rng, exploration)
                    verify_control(branches[control_index], repeat)
                finally:
                    restore(env, policy_rng, exploration, snapshot)
                prefix_arrays = _trace_arrays(initial, trace)
                group_metadata = dict(metadata, schema=BRANCH_SCHEMA, group_id=metadata['episode_id'] + f':anchor={t}',
                                      anchor=t, horizon_steps=horizon_steps, history_length=history_length, context_size=context_size,
                                      prefix_behavior=PREFIX_BEHAVIOR, prefix_data_sha256=prefix_data_digest(prefix_arrays),
                                      first_acceptance=[{'accepted': branch['trace'][0]['accepted'],
                                                         'executed_action': branch['trace'][0]['executed_action'],
                                                         'termination_cause': branch['trace'][-1]['termination_cause'],
                                                         'physical_first_step': {key: branch['trace'][0][key] for key in
                                                             ('meal_total_g', 'insulin_total_U', 'scenario_meal_avg') if key in branch['trace'][0]}}
                                                        for branch in branches])
                branch_arrays = tuple(_trace_arrays(current, branch['trace'], future=True) for branch in branches)
                audit = {'snapshot_sha256': snapshot.sha256, 'snapshot_inventory': snapshot.inventory,
                         'restored_after_sha256': state_digest(env, policy_rng, exploration),
                         'candidate_count': 25, 'duplicate_control_verified': True, 'control_index': control_index,
                         'control_repeat': repeat,
                         'branches': [dict({key: value for key, value in branch.items() if key != 'trace'},
                                           data_sha256=branch_data_digest(arrays))
                                      for branch, arrays in zip(branches, branch_arrays)]}
                group = BranchGroup(prefix=prefix_arrays, candidates=np.asarray(CANDIDATES, dtype=np.int64),
                                    branches=branch_arrays,
                                    metadata=group_metadata, audit=audit)
                validate_branch_group(group)
                if publish is not None:
                    publish(group)
                groups.append(group)
        if ended or t == prefix_horizon:
            break
        # A latent ordinary draw advances each stream once; the actual prefix
        # recommendation is always zero, independent of any candidate outcome.
        recommend(actor, normalizer, current, mode, probability, policy_rng, exploration)
        row = step(env, (0, 0))
        trace.append(row)
        current = np.asarray(row['observation'], dtype=initial.dtype)
        ended = row['terminated'] or row['truncated']
    for anchor in anchors:
        if anchor > len(trace) and not any(item['anchor'] == anchor for item in skipped):
            skipped.append({'anchor': anchor, 'reason': 'prefix_ended'})
    if not ended:
        raise ValueError('Simulator did not end at its declared prefix horizon')
    if digest(actor.state_dict()) != actor_before or (normalizer is not None and digest(vars(normalizer)) != normalization_before):
        raise ValueError('Fixed continuation policy or normalization changed')
    prefix = _trace_arrays(initial, trace)
    return tuple(groups), prefix, tuple(sorted(skipped, key=lambda item: item['anchor']))


def run_collection(*, checkpoint, config, simulator_root, output_dir, patient_type, patient_name,
                   split, env_seed, action_seed, exploration_seed, anchors=DEFAULT_ANCHORS,
                   horizon_steps=12, history_length=12, context_size=5, prefix_horizon_days=1,
                   action_mode='stochastic', exploration_probability=0.1, exclude_collections=()):
    """One complete reset family per new output directory; publish failure status."""
    from glucoalg.dynamics.collect import _load_policy
    from glucoalg.eval_grid import validate_protocol
    from glucoalg.runtime import configure_runtime, initialize_simulator
    from glucoalg.tuning.plan import _content_hash, probe_sources
    validate_protocol(patient_type=patient_type, patients=[patient_name], episodes=1,
                      horizon_days=prefix_horizon_days, eval_seed_base=env_seed, action_mode=action_mode)
    anchors = tuple(anchors)
    if (not anchors or any(type(anchor) is not int or anchor < 1 for anchor in anchors)
            or tuple(sorted(set(anchors))) != anchors or anchors[-1] > round(prefix_horizon_days * 288)):
        raise ValueError('Anchors must be distinct positive increasing integers within the prefix horizon')
    for value, name in ((horizon_steps, 'horizon_steps'), (history_length, 'history_length'), (context_size, 'context_size')):
        _positive(value, name)
    if split not in SPLITS:
        raise ValueError('Invalid branch split')
    for value, name in ((env_seed, 'env_seed'), (action_seed, 'action_seed'), (exploration_seed, 'exploration_seed')):
        _check_seed(value, name)
    if type(exploration_probability) not in (int, float) or not 0 <= exploration_probability <= 1:
        raise ValueError('Invalid exploration probability')
    patient_type = {'t2dnp': 't2d_no_pump'}.get(patient_type, patient_type)
    prior = load_branch_groups(exclude_collections)
    if any(episode_family(group.metadata) == (patient_type, patient_name, env_seed) for group in prior):
        raise ValueError('Reset episode overlaps an excluded branch collection')
    checkpoint, config = Path(checkpoint).resolve(), Path(config).resolve()
    policy = {'checkpoint': str(checkpoint), 'config': str(config),
              'checkpoint_sha256': sha256_file(checkpoint), 'config_sha256': sha256_file(config)}
    configuration = json.loads(config.read_text())
    description = (f"Unshielded fixed categorical checkpoint {policy['checkpoint_sha256']}, configuration "
                   f"{policy['config_sha256']}; {action_mode} policy proposal drawn each step, then independently "
                   f"replaced with probability {exploration_probability:g} by uniform independent bolus/meal "
                   "indices in [0,4]. Simulator acceptance, dose noise and autonomous meals remain active.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    collection = {'schema': COLLECTION_SCHEMA, 'status': 'running', 'split': split,
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'groups': [], 'errors': [],
                  'anchors_requested': list(anchors), 'prefix_behavior': PREFIX_BEHAVIOR,
                  'prefix_description': 'All prefix recommendations [0,0]; basal/scenario/simulator unchanged. Latent policy and exploration draws advance once per prefix decision; branches never feed back.',
                  'policy': policy, 'patient_type': patient_type, 'patient_name': patient_name,
                  'seed': env_seed, 'action_seed': action_seed, 'exploration_seed': exploration_seed}
    _write_json(output / 'collection.json', collection)
    _write_json(output / 'run.json', {'status': 'started'})
    env = None
    try:
        configure_runtime()
        torch.set_num_threads(1)
        runtime = initialize_simulator(simulator_root)
        runtime['glucosim_content_sha256'] = _content_hash(Path(runtime['glucosim_file']).parent.parent, ('glucosim',))
        sources = probe_sources(simulator_root)
        metadata = {'episode_id': f'{patient_type}:{patient_name}:reset={env_seed}:action={action_seed}:explore={exploration_seed}',
                    'patient_type': patient_type, 'patient_name': patient_name, 'seed': env_seed,
                    'action_seed': action_seed, 'exploration_seed': exploration_seed, 'split': split,
                    'prefix_horizon_steps': int(prefix_horizon_days * 288), 'prefix_behavior': PREFIX_BEHAVIOR,
                    'horizon_steps': horizon_steps, 'history_length': history_length, 'context_size': context_size,
                    'controller_interval_minutes': 5,
                    'observation_features': list(FEATURES), 'execution_semantics': EXECUTION_SEMANTICS,
                    'policy': policy, 'behavior': {'action_mode': action_mode, 'exploration_probability': exploration_probability,
                                                 'exploration_rng': 'numpy.PCG64', 'description': description},
                    'continuation_policy': description, 'runtime': runtime, 'source_identity': sources}
        collection['metadata'] = metadata
        _write_json(output / 'collection.json', collection)
        env, actor, normalizer = _load_policy(checkpoint, configuration, patient_type, patient_name, env_seed, prefix_horizon_days)
        actor.eval()

        def publish(group):
            directory = save_branch_group(output / f'anchor-{group.metadata["anchor"]:06d}', group)
            collection['groups'].append({'path': directory.name, 'group_id': group.metadata['group_id'],
                                         'manifest_sha256': sha256_file(directory / 'manifest.json')})
            _write_json(output / 'collection.json', collection)

        groups, prefix, skipped = collect_branch_episode(env, actor, normalizer, metadata=metadata,
            anchors=anchors, horizon_steps=horizon_steps, history_length=history_length, context_size=context_size, publish=publish)
        with (output / 'prefix.npz').open('xb') as stream:
            np.savez_compressed(stream, **prefix)
        sources_after = probe_sources(simulator_root)
        if any(sources[key]['content_sha256'] != sources_after[key]['content_sha256'] for key in ('repo', 'simulator')):
            raise ValueError('Source or simulator changed during branch collection')
        if any(sha256_file(path) != policy[name + '_sha256'] for name, path in (('checkpoint', checkpoint), ('config', config))):
            raise ValueError('Policy files changed during branch collection')
        metrics = _collection_metrics(groups, prefix, anchors, horizon_steps, metadata['prefix_horizon_steps'])
        _write_json(output / 'metrics.json', metrics)
        _write_json(output / 'split_check.json', {'overlap_count': 0, 'split': split,
                     'reset_family': list(episode_family(metadata)), 'group_count': len(groups),
                     'excluded_groups_checked': len(prior)})
        collection.update(status='complete', skipped_anchors=list(skipped), metrics=metrics,
                          source_after=sources_after, prefix_sha256=sha256_file(output / 'prefix.npz'))
        _write_json(output / 'collection.json', collection)
        _write_json(output / 'run.json', {'status': 'complete', 'collection_sha256': sha256_file(output / 'collection.json')})
        return collection
    except BaseException as error:
        collection['status'] = 'failed'
        collection['errors'].append(f'{type(error).__name__}: {error}')
        _write_json(output / 'collection.json', collection)
        _write_json(output / 'run.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'simulator-root', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--patient-type', default='t1d')
    parser.add_argument('--patient-name', required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    for name in ('env-seed', 'action-seed', 'exploration-seed'):
        parser.add_argument('--' + name, type=_seed, required=True)
    parser.add_argument('--anchors', nargs='+', type=int, default=DEFAULT_ANCHORS)
    parser.add_argument('--horizon-steps', type=int, default=12)
    parser.add_argument('--history-length', type=int, default=12)
    parser.add_argument('--context-size', type=int, default=5)
    parser.add_argument('--prefix-horizon-days', type=int, default=1)
    parser.add_argument('--action-mode', choices=('stochastic', 'deterministic'), default='stochastic')
    parser.add_argument('--exploration-probability', type=float, default=0.1)
    parser.add_argument('--exclude-collections', nargs='*', type=Path, default=[])
    args = parser.parse_args(argv)
    result = run_collection(**vars(args))
    print(json.dumps({'status': result['status'], 'groups': len(result['groups']), 'metrics': result['metrics']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
