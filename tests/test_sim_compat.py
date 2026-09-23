"""Fresh-process checks against an installed or adjacent GlucoSim checkout."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def _simulator_root():
    configured = os.environ.get("GLUCOSIM_PATH")
    if configured:
        root = Path(configured).resolve()
        if not (root / "glucosim" / "__init__.py").is_file():
            pytest.fail(f"GLUCOSIM_PATH is not a simulator checkout: {root}")
        return root
    adjacent = REPO.parent / "GlucoSim"
    if (adjacent / "glucosim" / "__init__.py").is_file():
        return adjacent
    spec = importlib.util.find_spec("glucosim")
    if spec is None or spec.origin is None:
        pytest.skip("GlucoSim is not installed; set GLUCOSIM_PATH to a checkout")
    return Path(spec.origin).resolve().parents[1]


@pytest.mark.parametrize("first_import", ["glucobench", "glucosim", "omnisafe"])
def test_legacy_namespace_preserves_simulator_identity(first_import, tmp_path):
    simulator_root = _simulator_root()
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(REPO), str(simulator_root))),
        PYTHONDONTWRITEBYTECODE="1",
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
    )
    code = r'''
import importlib
from pathlib import Path
import sys

importlib.import_module(sys.argv[1])
current = importlib.import_module("glucosim")
legacy = importlib.import_module("glucobench")
assert current is legacy
assert Path(current.__file__).resolve().parents[1] == Path(sys.argv[2]).resolve()

for name in (
    "gym_env.utils.registration",
    "gym_env.wrappers.common",
    "safety_gymnasium.wrappers",
    "safety_gymnasium.utils.registration",
    "safety_gymnasium.vector",
    "safety_gymnasium.vector.sync_vector_env",
):
    old = importlib.import_module("glucobench." + name)
    new = importlib.import_module("glucosim." + name)
    assert old is new, name
    assert new.__name__ == "glucosim." + name, name
    assert new.__spec__.name == new.__name__, name

registry = importlib.import_module("glucosim.gym_env.utils.registration")
assert {"t1d-v0", "t2d-v0", "t2d_no_pump-v0"} <= registry.registry.keys()
old_registry = importlib.import_module("glucobench.gym_env.utils.registration")
assert old_registry.registry is registry.registry

import omnisafe
from glucosim.diabetes_cmdp import DiabetesEnvs
from glucosim.safety_gymnasium.vector import SafetySyncVectorEnv
old_vector = importlib.import_module("glucobench.safety_gymnasium.vector")
assert old_vector.SafetySyncVectorEnv is SafetySyncVectorEnv
assert set(DiabetesEnvs.support_envs()) <= set(omnisafe.envs.support_envs())
try:
    importlib.import_module("glucobench._glucoalg_nonexistent_module")
except ModuleNotFoundError as exc:
    assert exc.name == "glucosim._glucoalg_nonexistent_module"
else:
    raise AssertionError("missing module unexpectedly imported")
'''
    result = subprocess.run(
        [sys.executable, "-c", code, first_import, str(simulator_root)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
