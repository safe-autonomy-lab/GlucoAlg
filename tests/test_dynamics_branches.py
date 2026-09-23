"""External engineering tests; fake transitions are never research evidence."""
import copy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glucoalg.dynamics import branches as probe


class Core:
    _warmup_cache = {'fixture': np.arange(3)}

    def __init__(self):
        self._jax_state = {'cgm': np.array([100.]), 'step': np.array(0)}
        self.key = np.array([2, 7], dtype=np.uint32)
        self.episode_steps = 0
        self.episode_return = 0.
        self.log_file = None
        self._np_random = np.random.Generator(np.random.PCG64(30))


def environment():
    core = Core()
    observation = np.array([[100.] + [0.] * 13])
    vector = SimpleNamespace(envs=[core], _env_obs=[observation[0].copy()], _observations=observation,
                             _rewards=np.zeros(1), _costs=np.zeros(1), _terminations=np.zeros(1, dtype=bool),
                             _truncations=np.zeros(1, dtype=bool), _autoreset_envs=np.zeros(1, dtype=bool))
    return SimpleNamespace(env=vector, _num_envs=1, _device=torch.device('cpu'))


def randoms():
    return torch.Generator(device='cpu').manual_seed(100), np.random.Generator(np.random.PCG64(200))


def fake_draw(actor, normalizer, raw, mode, probability, policy_rng, exploration):
    policy = torch.randint(0, 5, (2,), generator=policy_rng).numpy()
    exploring = exploration.random() < probability
    return (exploration.integers(0, 5, 2) if exploring else policy), bool(exploring)


def fake_step(env, action):
    core = env.env.envs[0]
    assert not env.env._autoreset_envs.any(), 'stepped after termination'
    core.episode_steps += 1
    core.key += np.array([1, 3], dtype=np.uint32)
    core._jax_state['step'] += 1
    core._jax_state['cgm'] += float(action[1] - action[0]) + core._np_random.random()
    core.episode_return += float(core._jax_state['cgm'][0])
    env.env._observations[0, 0] = core._jax_state['cgm'][0]
    env.env._rewards[0] = core.episode_return
    terminal = core.episode_steps == 4
    env.env._terminations[0] = terminal
    env.env._autoreset_envs[0] = terminal
    return {'observation': env.env._observations[0].tolist(), 'terminated': terminal,
            'truncated': False, 'recommended_action': action.tolist()}


def branch(env, rng, explore, candidate=(0, 0), horizon=3):
    return probe.run_branch(env, None, None, np.array([100.] + [0.] * 13), candidate, horizon,
                            'stochastic', .5, rng, explore, draw=fake_draw, step_function=fake_step)


def test_restore_covers_nested_state_vector_buffers_and_both_owned_random_streams():
    env, (rng, explore) = environment(), randoms()
    snapshot = probe.capture(env, rng, explore)
    before = snapshot.sha256
    first = branch(env, rng, explore)
    assert probe.state_digest(env, rng, explore) != before
    probe.restore(env, rng, explore, snapshot)
    assert probe.state_digest(env, rng, explore) == before
    assert probe.buffer_aliases(env.env) == snapshot.inventory['buffer_aliases']
    second = branch(env, rng, explore)
    probe.verify_control(first, second)
    assert probe.digest(snapshot.state) == before


def test_each_candidate_starts_with_same_draws_and_changes_observed_outcome():
    env, (rng, explore) = environment(), randoms()
    snapshot = probe.capture(env, rng, explore)
    first = branch(env, rng, explore, (0, 0))
    probe.restore(env, rng, explore, snapshot)
    second = branch(env, rng, explore, (4, 0))
    assert [r['unforced_proposal'] for r in first['trace']] == [r['unforced_proposal'] for r in second['trace']]
    assert first['trace'][0]['observation'] != second['trace'][0]['observation']
    assert first['trace_sha256'] != second['trace_sha256']


def test_branch_retains_float32_before_policy_normalization_after_json_conversion():
    env, (rng, explore) = environment(), randoms()
    observed_dtypes = []

    def draw(actor, normalizer, raw, *args):
        observed_dtypes.append(raw.dtype)
        return fake_draw(actor, normalizer, raw, *args)

    probe.run_branch(env, None, None, np.array([100.] + [0.] * 13, dtype=np.float32),
                     (0, 0), 3, 'stochastic', .5, rng, explore, draw=draw, step_function=fake_step)
    assert observed_dtypes == [np.dtype('float32')] * 3


def test_terminal_branches_stop_before_autoreset_and_can_restore_anchor():
    env, (rng, explore) = environment(), randoms()
    snapshot = probe.capture(env, rng, explore)
    result = branch(env, rng, explore, horizon=12)
    assert result['length'] == 4 and result['coverage_fraction'] == 4 / 12
    probe.restore(env, rng, explore, snapshot)
    assert not env.env._autoreset_envs.any()


@pytest.mark.parametrize('change', ['trace_sha256', 'end_state_sha256'])
def test_duplicate_control_rejects_either_trace_or_end_state_disagreement(change):
    row = {'trace_sha256': 'trace', 'end_state_sha256': 'state'}
    other = dict(row, **{change: 'different'})
    with pytest.raises(ValueError, match='duplicate control'):
        probe.verify_control(row, other)


def test_warmup_cache_change_is_rejected_instead_of_silently_restored():
    env, (rng, explore) = environment(), randoms()
    snapshot = probe.capture(env, rng, explore)
    saved = copy.deepcopy(Core._warmup_cache)
    try:
        Core._warmup_cache['fixture'][0] += 1
        with pytest.raises(ValueError, match='warmup cache changed'):
            probe.restore(env, rng, explore, snapshot)
    finally:
        Core._warmup_cache = saved


@pytest.mark.parametrize('change', ['logging', 'autoreset', 'multiple', 'not_reset'])
def test_snapshot_rejects_unsupported_operational_state(change):
    env, (rng, explore) = environment(), randoms()
    if change == 'logging':
        env.env.envs[0].log_file = 'should-not-write.csv'
    elif change == 'autoreset':
        env.env._autoreset_envs[0] = True
    elif change == 'multiple':
        env.env.envs.append(Core())
    else:
        env.env.envs[0]._jax_state = None
    with pytest.raises(ValueError):
        probe.capture(env, rng, explore)


def test_unknown_state_is_not_ignored():
    with pytest.raises(TypeError, match='unsupported simulator state'):
        probe.digest(object())
    with pytest.raises(TypeError, match='object arrays'):
        probe.digest(np.array([object()]))


def test_buffer_views_that_deepcopy_cannot_preserve_fail_before_branching():
    env, (rng, explore) = environment(), randoms()
    env.env._env_obs[0] = env.env._observations[0]
    with pytest.raises(ValueError, match='alias relationships'):
        probe.capture(env, rng, explore)


class ParityActor:
    shield = None

    def logits_net(self, normalized):
        result = torch.arange(10, dtype=torch.float32)[None] / 10
        return result + normalized[:, :1] * torch.linspace(-.005, .005, 10)[None]

    def __call__(self, normalized, raw):
        from glucoalg.evaluation import MultiCategoricalDistribution
        return MultiCategoricalDistribution([torch.distributions.Categorical(logits=part)
                                             for part in self.logits_net(normalized).split((5, 5), dim=-1)])


class ParityEnv:
    action_space = SimpleNamespace(nvec=np.array([5, 5]))
    sample_time = 5
    simulation_minutes = 1440

    def __init__(self):
        self.closed = False

    def reset(self, *, seed):
        self.n = 0
        self.cgm = 110.
        self.rng = np.random.Generator(np.random.PCG64(seed))
        return torch.tensor([[self.cgm] + [0.] * 13]), {}

    def step(self, action):
        self.n += 1
        assert self.n <= 10
        self.cgm += float(action[0, 1] - action[0, 0]) + self.rng.random()
        info = {'bolus_accepted': np.array([self.n % 2 == 0]), 'meal_accepted': np.array([True]),
                'termination_cause': np.array([int(self.n == 10)])}
        return (torch.tensor([[self.cgm] + [0.] * 13]), torch.tensor([1.]), torch.tensor([0.]),
                torch.tensor([self.n == 10]), torch.tensor([False]), info)

    def close(self):
        self.closed = True


def metadata(mode, epsilon):
    from glucoalg.dynamics.data import EXECUTION_SEMANTICS, FEATURES, SCHEMA, SCHEMA_VERSION
    return {'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'episode_id': 'fake-parity-only',
            'patient_type': 't1d', 'patient_name': 'adolescent#001', 'split': 'test',
            'seed': 1042, 'action_seed': 1042, 'exploration_seed': 11042, 'controller_interval_minutes': 5,
            'horizon_steps': 288, 'observation_features': list(FEATURES), 'execution_semantics': EXECUTION_SEMANTICS,
            'continuation_policy': 'fake engineering fixture',
            'policy': {'checkpoint': 'fixture.pt', 'config': 'fixture.json', 'checkpoint_sha256': 'a' * 64,
                       'config_sha256': 'b' * 64},
            'behavior': {'action_mode': mode, 'exploration_probability': epsilon,
                         'exploration_rng': 'numpy.PCG64', 'description': 'fixture'},
            'runtime': {'glucosim_commit': 'fake', 'glucosim_content_sha256': 'c' * 64}, 'metrics': {}}


@pytest.mark.parametrize('mode', ['stochastic', 'deterministic'])
@pytest.mark.parametrize('epsilon', [0., .5, 1.])
def test_actual_collection_and_rollout_apis_match_and_preserve_caller_rng(mode, epsilon):
    environments = []

    def factory():
        result = ParityEnv()
        environments.append(result)
        return result

    normalizer = SimpleNamespace(normalize=lambda raw: raw / 50.)
    before = probe.digest((random.getstate(), np.random.get_state(), torch.random.get_rng_state()))
    report, first, second = probe.collection_rollout_parity(factory, ParityActor(), normalizer, metadata(mode, epsilon))
    assert probe.digest((random.getstate(), np.random.get_state(), torch.random.get_rng_state())) == before
    assert report['all_arrays_bitwise_identical'] and len(environments) == 2
    assert all(env.closed for env in environments)
    assert len(first['observations']) == 11
    assert not np.array_equal(first['executed_actions'], first['recommended_actions'])
    assert len(second['observations']) == 11


def test_byte_level_array_gate_detects_signed_zero_difference():
    from glucoalg.dynamics.data import ARRAY_NAMES
    first = {name: np.array([0.]) for name in ARRAY_NAMES}
    second = dict(first, costs=np.array([-0.]))
    with pytest.raises(ValueError, match='bitwise mismatch: costs'):
        probe.verify_episode_arrays(first, second)


def test_owned_sampler_matches_collection_and_does_not_change_global_torch_rng():
    from glucoalg.dynamics.collect import _recommend
    actor, normalizer = ParityActor(), SimpleNamespace(normalize=lambda raw: raw / 50.)
    raw = np.array([101.] + [0.] * 13)
    rng, explore = randoms()
    before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(rng.get_state())
        expected = _recommend(actor, normalizer, raw, 'stochastic')
        expected_after = torch.random.get_rng_state().clone()
    actual, explored = probe.recommend(actor, normalizer, raw, 'stochastic', 0., rng, explore)
    np.testing.assert_array_equal(actual, expected)
    assert torch.equal(rng.get_state(), expected_after) and not explored
    assert torch.equal(before, torch.random.get_rng_state())


class CorpusActor(ParityActor):
    def state_dict(self):
        return {'fixture': torch.arange(10)}

    def eval(self):
        return self


class CorpusEnv:
    """Stateful fake with real owned RNG, vector buffers and acceptance gates."""
    sample_time = 5
    action_space = SimpleNamespace(nvec=np.array([5, 5]))

    def __init__(self, horizon=30, terminal=None):
        self.horizon = horizon
        self.terminal = terminal or horizon
        self.simulation_minutes = 5 * horizon
        self.closed = False
        self.env = environment().env

    def reset(self, *, seed):
        self.env = environment().env
        self.env._observations = self.env._observations.astype(np.float32)
        self.env._env_obs = [self.env._observations[0].copy()]
        self.env.envs[0]._np_random = np.random.Generator(np.random.PCG64(seed))
        return self.env._observations.copy(), {}

    def step(self, action):
        action = np.asarray(action)[0]
        core = self.env.envs[0]
        assert core.episode_steps < self.terminal
        # Gate uses the saved current step, independently of future outcomes.
        accepted = np.array([core.episode_steps % 2 == 1, True]) & (action > 0)
        executed = np.where(accepted, action, 0)
        core.episode_steps += 1
        core.key += np.array([1, 3], dtype=np.uint32)
        core._jax_state['step'] += 1
        core._jax_state['cgm'] += float(executed[1] - executed[0]) + core._np_random.random()
        core.episode_return += 1.
        self.env._observations[0, 0] = core._jax_state['cgm'][0]
        self.env._env_obs = [self.env._observations[0].copy()]
        self.env._rewards[:] = 1.
        ended = core.episode_steps == self.terminal
        self.env._terminations[:] = ended and self.terminal < self.horizon
        self.env._truncations[:] = ended and self.terminal == self.horizon
        self.env._autoreset_envs[:] = ended
        info = {'bolus_accepted': np.array([accepted[0]]), 'meal_accepted': np.array([accepted[1]]),
                'termination_cause': np.array([2 if ended and self.terminal < self.horizon else 0]),
                'meal_total_g': np.array([float(executed[1])]), 'insulin_total_U': np.array([float(executed[0])])}
        return (self.env._observations.copy(), self.env._rewards.copy(), self.env._costs.copy(),
                self.env._terminations.copy(), self.env._truncations.copy(), info)

    def close(self):
        self.closed = True


def corpus_metadata(*, split='train', seed=2000, horizon=30):
    result = metadata('stochastic', .1)
    result.update(episode_id=f't1d:adolescent#001:reset={seed}', split=split,
                  seed=seed, action_seed=seed, exploration_seed=seed + 10000,
                  prefix_horizon_steps=horizon, prefix_behavior=probe.PREFIX_BEHAVIOR,
                  source_identity={'fixture': True})
    return result


def collect_fixture(*, terminal=None, anchors=(7, 17, 25)):
    env = CorpusEnv(30, terminal)
    result = probe.collect_branch_episode(env, CorpusActor(), None, metadata=corpus_metadata(),
        anchors=anchors, horizon_steps=4, history_length=3, context_size=2)
    return env, result


@pytest.fixture(scope='module')
def corpus():
    return collect_fixture()[1]


def test_full_joint_corpus_restores_every_candidate_and_keeps_no_action_prefix(corpus):
    groups, prefix, skipped = corpus
    assert len(groups) == 3 and skipped == ()
    assert not prefix['recommended_actions'].any() and not prefix['executed_actions'].any()
    assert len(prefix['rewards']) == 30 and prefix['truncated'][-1]
    direct = CorpusEnv(30)
    initial, _ = direct.reset(seed=2000)
    trace = [probe.step(direct, (0, 0)) for _ in range(30)]
    expected = probe._trace_arrays(initial[0], trace)
    probe.verify_episode_arrays(expected, prefix)
    for group in groups:
        probe.validate_branch_group(group, required_split='train')
        assert group.candidates.tolist() == [list(candidate) for candidate in probe.CANDIDATES]
        assert group.audit['snapshot_sha256'] == group.audit['restored_after_sha256']
        assert group.audit['duplicate_control_verified'] and group.audit['control_index'] == 0
        assert len({tuple(branch['unforced_proposals'][0]) for branch in group.branches}) == 1
        assert all(branch['observations'].dtype == np.float32 for branch in group.branches)
        for candidate, branch in zip(group.candidates, group.branches):
            assert len(branch['rewards']) == 4
            np.testing.assert_array_equal(branch['recommended_actions'][0], candidate)
            np.testing.assert_array_equal(branch['glucose_delta'], branch['observations'][1:, 0] - group.prefix['observations'][-1, 0])


def test_early_endings_are_ragged_and_unreached_anchors_retained():
    env, (groups, prefix, skipped) = collect_fixture(terminal=19)
    assert [group.metadata['anchor'] for group in groups] == [7, 17]
    assert len(prefix['rewards']) == 19 and prefix['terminated'][-1]
    assert skipped == ({'anchor': 25, 'reason': 'prefix_ended'},)
    assert all(len(branch['glucose_delta']) == 2 for branch in groups[-1].branches)
    assert all(branch['terminated'][-1] for branch in groups[-1].branches)
    assert env.env.envs[0].episode_steps == 19


def test_insufficient_context_is_recorded_without_inventing_a_group():
    _, (groups, _, skipped) = collect_fixture(anchors=(3, 7))
    assert len(groups) == 1
    assert skipped == ({'anchor': 3, 'reason': 'insufficient_fully_observed_context', 'required_steps': 7},)


def test_group_roundtrip_binds_arrays_hashes_and_metadata_and_owns_memory(tmp_path, corpus):
    group = corpus[0][0]
    path = probe.save_branch_group(tmp_path / 'group', group)
    restored = probe.load_branch_groups([path], required_split='train')[0]
    assert restored.path == path.resolve()
    for name in ('data_sha256', 'metadata_sha256', 'manifest_sha256', 'prefix_data_sha256'):
        assert len(restored.metadata[name]) == 64
    for before, after in zip(group.branches, restored.branches):
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
            assert not after[name].flags.writeable
    with pytest.raises(FileExistsError):
        probe.save_branch_group(path, group)
    (path / 'data.npz').write_bytes(b'corruption')
    with pytest.raises(ValueError, match='hash mismatch'):
        probe.load_branch_group(path)


@pytest.mark.parametrize('mutation,error', [
    ('delta', 'targets'), ('first_action', 'forced'), ('prefix', 'digest'),
    ('duplicate', 'control'), ('snapshot', 'restoration'), ('candidates', '25 ordered')])
def test_invalid_group_data_and_audit_fail_closed(corpus, mutation, error):
    group = corpus[0][0]
    prefix = {name: array.copy() for name, array in group.prefix.items()}
    branches = [{name: array.copy() for name, array in branch.items()} for branch in group.branches]
    audit = copy.deepcopy(group.audit)
    candidates = group.candidates.copy()
    if mutation == 'delta':
        branches[1]['glucose_delta'][0] += 1
    elif mutation == 'first_action':
        branches[1]['recommended_actions'][0] = [0, 2]
        branches[1]['executed_actions'][0] = np.where(branches[1]['accepted'][0], [0, 2], 0)
    elif mutation == 'prefix':
        prefix['observations'][0, 0] += 1
    elif mutation == 'duplicate':
        audit['control_repeat']['trace_sha256'] = 'd' * 64
    elif mutation == 'snapshot':
        audit['restored_after_sha256'] = 'd' * 64
    else:
        candidates[-1] = candidates[0]
    changed = replace(group, prefix=prefix, branches=tuple(branches), candidates=candidates, audit=audit)
    with pytest.raises(ValueError, match=error):
        probe.validate_branch_group(changed)


def test_same_reset_family_cannot_cross_splits_even_with_new_action_seed(tmp_path, corpus):
    first, second = corpus[0][:2]
    changed = replace(second, metadata=dict(second.metadata, split='validation', action_seed=9090))
    paths = [probe.save_branch_group(tmp_path / 'train', first), probe.save_branch_group(tmp_path / 'val', changed)]
    with pytest.raises(ValueError, match='multiple splits'):
        probe.load_branch_groups(paths)


def test_identical_prefix_bytes_cannot_cross_splits_under_relabeled_patient(tmp_path, corpus):
    first = corpus[0][0]
    changed = replace(first, metadata=dict(first.metadata, split='validation', patient_name='adolescent#002',
        seed=9000, episode_id='relabeled', group_id='relabeled-anchor'))
    # Use a harmless reward change to keep whole data hashes distinct while
    # deliberately replaying exactly the same causal prefix.
    branches = [{name: value.copy() for name, value in branch.items()} for branch in changed.branches]
    branches[1]['rewards'][0] += 1
    audit = copy.deepcopy(changed.audit)
    audit['branches'][1]['data_sha256'] = probe.branch_data_digest(branches[1])
    changed = replace(changed, branches=tuple(branches), audit=audit)
    paths = [probe.save_branch_group(tmp_path / 'train', first), probe.save_branch_group(tmp_path / 'val', changed)]
    with pytest.raises(ValueError, match='Identical prefix'):
        probe.load_branch_groups(paths)


def test_duplicate_group_or_data_are_rejected(tmp_path, corpus):
    first = corpus[0][0]
    path = probe.save_branch_group(tmp_path / 'group', first)
    with pytest.raises(ValueError, match='Duplicate'):
        probe.load_branch_groups([path, path])
    changed = replace(first, metadata=dict(first.metadata, seed=9000, episode_id='relabeled', group_id='relabeled'))
    path2 = probe.save_branch_group(tmp_path / 'relabeled', changed)
    with pytest.raises(ValueError, match='Duplicate'):
        probe.load_branch_groups([path, path2])


def test_duplicate_control_failure_restores_prefix_and_publishes_nothing(monkeypatch):
    env = CorpusEnv()
    published = []
    def fail(first, second):
        raise ValueError('forced duplicate control failure')
    monkeypatch.setattr(probe, 'verify_control', fail)
    with pytest.raises(ValueError, match='forced duplicate'):
        probe.collect_branch_episode(env, CorpusActor(), None, metadata=corpus_metadata(), anchors=(7,),
            horizon_steps=4, history_length=3, context_size=2, publish=published.append)
    assert env.env.envs[0].episode_steps == 7 and published == []
    direct = CorpusEnv()
    direct.reset(seed=2000)
    for _ in range(7):
        probe.step(direct, (0, 0))
    assert probe.digest(vars(env)) == probe.digest(vars(direct))


@pytest.mark.parametrize('anchors', [(7, 7), (8, 7), (), (0,), (31,), (True,)])
def test_anchor_schedule_rejects_invalid_values_before_reset(anchors):
    env = CorpusEnv()
    with pytest.raises(ValueError, match='[Aa]nchor'):
        probe.collect_branch_episode(env, CorpusActor(), None, metadata=corpus_metadata(), anchors=anchors,
            horizon_steps=4, history_length=3, context_size=2)
    assert env.env.envs[0].episode_steps == 0


def test_default_protocol_anchor_budget():
    assert probe.DEFAULT_ANCHORS == tuple(range(27, 268, 24))
    assert len(probe.DEFAULT_ANCHORS) == 11
    assert len(probe.CANDIDATES) == 25
    assert 12 + 12 + 5 - 2 == probe.DEFAULT_ANCHORS[0]


def mock_collection_runtime(monkeypatch, tmp_path, *, fail=False, drift=False, terminal=None):
    from glucoalg import runtime
    from glucoalg.dynamics import collect
    from glucoalg.tuning import plan
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture checkpoint')
    config = tmp_path / 'config.json'
    config.write_text('{}')
    env = CorpusEnv(288, terminal)
    def load(*args):
        if fail:
            raise ValueError('forced policy load failure')
        return env, CorpusActor(), None
    calls = []
    def sources(*args):
        calls.append(1)
        value = 'd' if drift and len(calls) > 1 else 'c'
        return {key: {'content_sha256': value * 64} for key in ('repo', 'simulator')}
    monkeypatch.setattr(collect, '_load_policy', load)
    monkeypatch.setattr(runtime, 'configure_runtime', lambda: None)
    monkeypatch.setattr(runtime, 'initialize_simulator', lambda root: {
        'glucosim_file': str(tmp_path / 'glucosim' / '__init__.py'), 'glucosim_commit': 'fixture'})
    monkeypatch.setattr(plan, '_content_hash', lambda *args: 'c' * 64)
    monkeypatch.setattr(plan, 'probe_sources', sources)
    arguments = dict(checkpoint=checkpoint, config=config, simulator_root=tmp_path,
        output_dir=tmp_path / 'collection', patient_type='t1d', patient_name='adolescent#001',
        split='train', env_seed=2000, action_seed=2000, exploration_seed=12000, anchors=(27, 51))
    return env, arguments


def test_collection_completes_roundtrips_and_counts_unreached_denominator(monkeypatch, tmp_path):
    env, arguments = mock_collection_runtime(monkeypatch, tmp_path, terminal=35)
    record = probe.run_collection(**arguments)
    assert record['status'] == 'complete' and env.closed
    assert record['metrics']['requested_candidate_branches'] == 50
    assert record['metrics']['requested_future_steps'] == 600
    assert record['metrics']['candidate_branches'] == 25
    assert record['metrics']['future_steps'] == 200
    assert record['skipped_anchors'] == [{'anchor': 51, 'reason': 'prefix_ended'}]
    groups = probe.load_branch_groups(arguments['output_dir'], required_split='train')
    assert len(groups) == 1 and groups[0].metadata['anchor'] == 27
    assert all(len(branch['glucose_delta']) == 8 for branch in groups[0].branches)
    with pytest.raises(FileExistsError):
        probe.run_collection(**arguments)


@pytest.mark.parametrize('failure', ['load', 'source'])
def test_collection_failure_publishes_receipt_and_cannot_load_as_complete(monkeypatch, tmp_path, failure):
    env, arguments = mock_collection_runtime(monkeypatch, tmp_path, fail=failure == 'load', drift=failure == 'source')
    with pytest.raises(ValueError, match='failure|changed'):
        probe.run_collection(**arguments)
    record = json.loads((arguments['output_dir'] / 'collection.json').read_text())
    assert record['status'] == 'failed' and record['errors']
    assert json.loads((arguments['output_dir'] / 'run.json').read_text())['status'] == 'failed'
    if failure == 'source':
        assert env.closed and len(record['groups']) == 2
    with pytest.raises(ValueError, match='did not complete'):
        probe.load_branch_groups(arguments['output_dir'])


def test_collection_manifest_cannot_relabel_group_or_load_wrong_split(monkeypatch, tmp_path):
    _, arguments = mock_collection_runtime(monkeypatch, tmp_path, terminal=35)
    probe.run_collection(**arguments)
    with pytest.raises(ValueError, match='split'):
        probe.load_branch_groups(arguments['output_dir'], required_split='test')
    record_path = arguments['output_dir'] / 'collection.json'
    record = json.loads(record_path.read_text())
    record['groups'][0]['group_id'] = 'relabeled'
    probe._write_json(record_path, record)
    probe._write_json(arguments['output_dir'] / 'run.json', {'status': 'complete', 'collection_sha256': probe.sha256_file(record_path)})
    with pytest.raises(ValueError, match='identity'):
        probe.load_branch_groups(arguments['output_dir'])


@pytest.mark.parametrize('mutation,error', [('drop', 'omitted'), ('coverage', 'coverage'),
    ('skip', 'skipping'), ('prefix', 'prefix hash'), ('action_seed', 'metadata'),
    ('top_action_seed', 'metadata')])
def test_collection_resealed_inconsistencies_are_rejected(monkeypatch, tmp_path, mutation, error):
    _, arguments = mock_collection_runtime(monkeypatch, tmp_path, terminal=35)
    probe.run_collection(**arguments)
    path = arguments['output_dir']
    record = json.loads((path / 'collection.json').read_text())
    if mutation == 'drop':
        record['groups'] = []
    elif mutation == 'coverage':
        record['metrics']['requested_future_steps'] -= 300
    elif mutation == 'skip':
        record['skipped_anchors'][0]['reason'] = 'excluded_low_acceptance'
    elif mutation == 'prefix':
        (path / 'prefix.npz').write_bytes(b'changed')
    elif mutation == 'action_seed':
        record['metadata']['action_seed'] += 1
    else:
        record['action_seed'] += 1
    probe._write_json(path / 'collection.json', record)
    probe._write_json(path / 'run.json', {'status': 'complete', 'collection_sha256': probe.sha256_file(path / 'collection.json')})
    with pytest.raises(ValueError, match=error):
        probe.load_branch_groups(path)


def test_completely_unreached_collection_keeps_requested_coverage(monkeypatch, tmp_path):
    _, arguments = mock_collection_runtime(monkeypatch, tmp_path, terminal=20)
    record = probe.run_collection(**arguments)
    assert record['metrics']['requested_future_steps'] == 600
    assert record['metrics']['future_steps'] == 0
    assert record['metrics']['anchors_skipped'] == 2
    assert probe.load_branch_groups(arguments['output_dir']) == ()
    with pytest.raises(ValueError, match='split'):
        probe.load_branch_groups(arguments['output_dir'], required_split='test')
