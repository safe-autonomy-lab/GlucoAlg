"""Resource profiles reject unsafe directives and preserve explicit overrides."""
import json
from pathlib import Path

import pytest

from glucoalg.tuning.execution_profiles import resolve_profile, validate_profile
from glucoalg.tuning.spec import SpecError

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "execution"


def test_generic_and_slurm_example_profiles():
    assert resolve_profile() == resolve_profile(CONFIGS / "generic.json") == {}
    assert resolve_profile(CONFIGS / "slurm_cpu.json") == {
        "time": "24:00:00", "cpus_per_task": 4, "memory": "8G", "max_concurrent": 2,
    }


def test_overrides_are_explicit_and_do_not_mutate_profile(tmp_path):
    path = tmp_path / "profile.json"
    original = {"partition": "compute", "exclusive": True, "max_concurrent": 4}
    path.write_text(json.dumps(original))
    assert resolve_profile(path, partition=None, exclusive=False, max_concurrent=2) == {
        "partition": "compute", "exclusive": False, "max_concurrent": 2,
    }
    assert json.loads(path.read_text()) == original


@pytest.mark.parametrize("raw", [
    [], None, {"gpu": 1}, {"python": "/personal/env/bin/python"},
    {"extra_directives": "--exclusive"}, {"cpus-per-task": 2},
    {"cpus_per_task": True}, {"cpus_per_task": 0}, {"cpus_per_task": 1.5},
    {"max_concurrent": -1}, {"max_concurrent": "2"}, {"max_concurrent": False},
    {"exclusive": 1}, {"exclusive": "false"}, {"exclusive": None},
    {"partition": ""}, {"partition": "c pu"}, {"partition": "cpu\n#SBATCH --exclusive"},
    {"partition": "$(touch injected)"}, {"account": "a;echo"}, {"qos": "batch\x00"},
    {"partition": ",cpu"}, {"account": "-Aother"}, {"qos": "a,b"},
    {"time": "unlimited"}, {"time": "00:00:00"}, {"time": "1:60"},
    {"time": "-1"}, {"time": "1:2:3:4"}, {"time": "01:00:00\n"},
    {"memory": "0"}, {"memory": "-8G"}, {"memory": "1.5G"},
    {"memory": "2GB"}, {"memory": 8192}, {"memory": "8G --exclusive"},
])
def test_invalid_resource_profiles_rejected(raw):
    with pytest.raises(SpecError):
        validate_profile(raw)


@pytest.mark.parametrize("duration", ["60", "90:30", "24:00:00", "2-3", "2-3:04", "2-3:04:05"])
def test_numeric_slurm_durations(duration):
    assert validate_profile({"time": duration}) == {"time": duration}


def test_supported_resources():
    raw = {
        "partition": "cpu,compute-long", "account": "project_01", "qos": "normal",
        "cpus_per_task": 4, "memory": "8g", "max_concurrent": 2, "exclusive": False,
    }
    assert validate_profile(raw) == {**raw, "memory": "8G"}


def test_file_errors_and_duplicate_keys_are_spec_errors(tmp_path):
    with pytest.raises(SpecError, match="cannot parse"):
        resolve_profile(tmp_path / "missing.json")
    path = tmp_path / "profile.json"
    path.write_text('{"partition":"cpu","partition":"compute"}')
    with pytest.raises(SpecError, match="duplicate key"):
        resolve_profile(path)
    path.write_text('{"cpus_per_task":false}')
    with pytest.raises(SpecError, match="positive integer"):
        resolve_profile(path, cpus_per_task=2)
    with pytest.raises(SpecError, match="unknown.*override"):
        resolve_profile(unknown=None)
