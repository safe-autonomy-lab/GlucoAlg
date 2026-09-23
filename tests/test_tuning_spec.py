"""Spec validation and search-space expansion (no launches, no simulator)."""
import subprocess
import sys
from pathlib import Path

import pytest

from glucoalg.tuning.spec import (
    SpecError,
    expand_configs,
    expected_epochs,
    load_json_nodupes,
    normalize_spec,
    spec_hash,
)

ROOT = Path(__file__).resolve().parents[1]


def base_spec(**over):
    spec = {
        "study_name": "demo",
        "seeds": [100, 101],
        "cost_limit": 100.0,
        "factors": {"actor-lr": [0.0003, 3e-05]},
        "train": {
            "algo": "PPOLag",
            "env-id": "t1d-v0",
            "cohort": "adolescent",
            "total-steps": 640,
            "steps-per-epoch": 64,
            "vector-env-nums": 1,
            "device": "cpu",
        },
    }
    spec.update(over)
    return spec


def test_defaults_and_no_launch_determinism():
    spec = normalize_spec(base_spec())
    assert spec["seeds"] == [100, 101]
    minimal = {"study_name": "m", "cost_limit": 50,
               "factors": {"actor-lr": [1e-4]},
               "train": {"total-steps": 640, "steps-per-epoch": 64}}
    normed = normalize_spec(minimal)
    assert normed["seeds"] == [100, 101, 102]
    assert normed["search"] == {"mode": "grid"}
    assert normed["runner"] == "auto"
    first = expand_configs(spec)
    second = expand_configs(normalize_spec(base_spec()))
    assert first == second
    assert [c["config_id"] for c in first] == ["cfg00", "cfg01"]
    assert first[0]["params"] == {"actor-lr": 0.0003}


def test_factor_key_order_does_not_change_expansion_or_hash():
    a = normalize_spec(base_spec(factors={"actor-lr": [3e-4], "critic-lr": [1e-4, 5e-4]}))
    b = normalize_spec(base_spec(factors={"critic-lr": [1e-4, 5e-4], "actor-lr": [3e-4]}))
    assert expand_configs(a) == expand_configs(b)
    assert spec_hash(a) == spec_hash(b)


def test_random_subset_is_deterministic_and_bounded():
    spec = normalize_spec(base_spec(
        factors={"actor-lr": [1e-5, 3e-5, 1e-4, 3e-4], "critic-lr": [1e-4, 5e-4]},
        search={"mode": "random_subset", "size": 3, "seed": 7}))
    again = normalize_spec(base_spec(
        factors={"actor-lr": [1e-5, 3e-5, 1e-4, 3e-4], "critic-lr": [1e-4, 5e-4]},
        search={"mode": "random_subset", "size": 3, "seed": 7}))
    assert expand_configs(spec) == expand_configs(again)
    assert len(expand_configs(spec)) == 3
    other = normalize_spec(base_spec(
        factors={"actor-lr": [1e-5, 3e-5, 1e-4, 3e-4], "critic-lr": [1e-4, 5e-4]},
        search={"mode": "random_subset", "size": 3, "seed": 8}))
    assert expand_configs(spec) != expand_configs(other)
    with pytest.raises(SpecError):
        normalize_spec(base_spec(search={"mode": "random_subset", "size": 999, "seed": 0}))


def test_rejects_continuous_distributions():
    with pytest.raises(SpecError, match="categorical"):
        normalize_spec(base_spec(factors={"actor-lr": {"min": 1e-5, "max": 1e-3}}))


@pytest.mark.parametrize("mutate", [
    lambda s: s.update({"bogus": 1}),
    lambda s: s.update({"search": {"mode": "grid", "bogus": 1}}),
    lambda s: s["train"].update({"not-a-flag": 1}),
    lambda s: s["train"].update({"algo.cfgs.foo": 1}),
    lambda s: s.update({"search": {"mode": "fancy"}}),
    lambda s: s.update({"runner": "optuna"}),
])
def test_rejects_unknown_keys(mutate):
    spec = base_spec()
    mutate(spec)
    with pytest.raises(SpecError):
        normalize_spec(spec)


@pytest.mark.parametrize("key", ["seed", "log-dir", "simulator-root", "dry-run", "log_dir"])
def test_rejects_managed_options(key):
    with pytest.raises(SpecError, match="managed"):
        normalize_spec(base_spec(factors={key: [1]}))
    train = dict(base_spec()["train"])
    train[key] = 1
    with pytest.raises(SpecError, match="managed"):
        normalize_spec(base_spec(train=train))


def test_rejects_duplicates_and_bad_values(tmp_path):
    with pytest.raises(SpecError):
        normalize_spec(base_spec(seeds=[100, 100]))
    with pytest.raises(SpecError):  # same float, different spelling
        normalize_spec(base_spec(factors={"actor-lr": [1e-4, 0.0001]}))
    with pytest.raises(SpecError):
        normalize_spec(base_spec(factors={"actor-lr": []}))
    with pytest.raises(SpecError):
        normalize_spec(base_spec(factors={"actor-lr": [1e-4], "actor_lr": [3e-4]}))
    with pytest.raises(SpecError, match="both a fixed"):
        train = dict(base_spec()["train"])
        train["actor-lr"] = 1e-4
        normalize_spec(base_spec(train=train))
    for bad in (float("nan"), float("inf"), None, [1], {"a": 1}, ""):
        with pytest.raises(SpecError):
            normalize_spec(base_spec(factors={"actor-lr": [bad]}))
    with pytest.raises(SpecError):
        normalize_spec(base_spec(cost_limit=float("nan")))
    dup = tmp_path / "dup.json"
    dup.write_text('{"a": 1, "a": 2}')
    with pytest.raises(SpecError, match="duplicate key"):
        load_json_nodupes(str(dup))


def test_rejects_missing_or_bad_budget():
    train = {k: v for k, v in base_spec()["train"].items() if k != "total-steps"}
    with pytest.raises(SpecError, match="total-steps"):
        normalize_spec(base_spec(train=train))
    train = dict(base_spec()["train"])
    train["steps-per-epoch"] = 0
    with pytest.raises(SpecError):
        normalize_spec(base_spec(train=train))
    with pytest.raises(SpecError):
        normalize_spec(base_spec(seeds=[True]))
    with pytest.raises(SpecError):
        normalize_spec(base_spec(cost_limit=None))
    with pytest.raises(SpecError, match="size"):
        normalize_spec(base_spec(search={"mode": "grid", "size": 2}))


def test_aliases_and_expected_epochs():
    train = dict(base_spec()["train"])
    train["cost-limit"] = 100.0
    train["target-kl"] = 0.1
    spec = normalize_spec(base_spec(train=train))
    assert spec["train"]["cost-limit"] == 100.0
    assert expected_epochs(2000000, 2048) == 976
    assert expected_epochs(100, 10) == 10


def test_import_pulls_no_heavy_dependencies():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "import glucoalg.tuning; "
        "heavy = [m for m in ('glucosim', 'torch', 'omnisafe', 'matplotlib') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

def test_rejects_normalized_top_level_collision():
    spec = base_spec()
    spec["cost-limit"] = 50.0  # collides with cost_limit after normalization
    with pytest.raises(SpecError, match="duplicate top-level"):
        normalize_spec(spec)


@pytest.mark.parametrize("changes", [
    {"seeds": [-1, 100]}, {"seeds": [2**32, 100]}, {"cost_limit": -1},
    {"factors": {"actor-lr": [-0.1]}}, {"factors": {"actor-lr": ["0.0001"]}},
    {"factors": {"algo": ["DDPGLag"]}}, {"factors": {"device": ["cuda:0"]}},
    {"factors": {"critic-lr": [1, 1.0]}},
    {"factors": {"actor-lr": [10**400]}},
    {"cost_limit": 10**400},
    {"factors": {"use-wandb": [True, 1, "true"]}},
])
def test_invalid_training_values_rejected_before_planning(changes):
    spec = base_spec(**changes)
    # Remove a fixed key when testing that key as a factor.
    for key in spec["factors"]:
        spec["train"].pop(key, None)
    with pytest.raises(SpecError):
        normalize_spec(spec)


def test_subepoch_budget_rejected():
    spec = base_spec()
    spec["train"]["total-steps"] = 32
    with pytest.raises(SpecError, match="epoch"):
        normalize_spec(spec)
