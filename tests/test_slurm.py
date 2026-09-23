"""Render and exercise scripts with test substitutes; never contact Slurm."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from glucoalg.tuning import plan as planmod
from glucoalg.tuning import slurm
from glucoalg.tuning.__main__ import main
from glucoalg.tuning.plan import create_plan
from glucoalg.tuning.spec import SpecError

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "execution"


@pytest.fixture
def study(tmp_path, monkeypatch):
    repo = tmp_path / "repo 'quoted' $(echo injected)"
    (repo / "glucoalg").mkdir(parents=True)
    (repo / "glucoalg" / "train.py").write_text("# frozen source\n")
    sim = tmp_path / "simulator with spaces"
    sim.mkdir()
    monkeypatch.setattr(planmod, "repo_root", lambda: repo)
    spec = {
        "study_name": "slurm demo", "seeds": [100, 101], "cost_limit": 100,
        "simulator_root": str(sim), "factors": {"actor-lr": [0.0003]},
        "train": {"total-steps": 100, "steps-per-epoch": 10, "batch-size": 10},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    out = tmp_path / "study 'quoted' $(echo injected)"
    create_plan(str(spec_path), str(out))
    return out


def metadata(text):
    prefix = "# Execution metadata: "
    return json.loads(next(line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)))


def test_default_export_is_portable_and_preserves_plan(study):
    originals = {path: path.read_bytes() for path in study.iterdir() if path.is_file()}
    out = study / "array.sbatch"
    slurm.export_slurm(study, str(out))
    text = out.read_text()
    for directive in ("partition", "account", "qos", "time", "exclusive", "cpus-per-task", "mem"):
        assert f"#SBATCH --{directive}" not in text
    assert "#SBATCH --array=0-1" in text
    assert metadata(text)["effective_profile"] == {}
    assert metadata(text)["profile_path"] is None
    assert metadata(text)["python"] == sys.executable
    assert "'--job', job_id" in text and "'--all'" not in text
    assert "check_source_provenance" in text
    assert "\nsbatch " not in text
    for path, original in originals.items():
        assert path.read_bytes() == original
    subprocess.run(["bash", "-n", str(out)], check=True, capture_output=True)


def test_slurm_profile_and_cli_precedence(study):
    out = study / "array.sbatch"
    common = ["export-slurm", "--plan", str(study), "--out", str(out),
              "--profile", str(CONFIGS / "slurm_cpu.json")]
    assert main(common) == 0
    text = out.read_text()
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --exclusive" not in text
    assert main(common + ["--partition", "compute", "--time", "2:00:00", "--no-exclusive",
                          "--account", "project", "--qos", "normal", "--cpus-per-task", "4",
                          "--memory", "8G", "--max-concurrent", "2"]) == 0
    text = out.read_text()
    for directive in ("partition=compute", "time=2:00:00", "account=project", "qos=normal",
                      "cpus-per-task=4", "mem=8G", "array=0-1%2"):
        assert f"#SBATCH --{directive}" in text
    assert "#SBATCH --exclusive" not in text
    assert metadata(text)["effective_profile"]["exclusive"] is False
    assert metadata(text)["profile_path"] == str(CONFIGS / "slurm_cpu.json")


def test_shell_paths_are_single_literal_arguments(study, tmp_path, monkeypatch):
    """Actually execute Bash up to a fake interpreter, including hostile paths."""
    capture = tmp_path / "captured.json"
    fake_python = tmp_path / "python with 'quotes' $(echo injected)"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(capture)!r}).write_text(json.dumps({{\n"
        "    'argv': sys.argv, 'path': os.environ['PYTHONPATH'], 'code': sys.stdin.read(),\n"
        "    'jax': os.environ['JAX_PLATFORMS'], 'legacy_jax': os.environ['JAX_PLATFORM_NAME'],\n"
        "    'cwd': os.getcwd(), 'cuda': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
    )
    fake_python.chmod(0o700)
    out = study / "array script.sbatch"
    with monkeypatch.context() as patch:
        patch.setattr(slurm.sys, "executable", str(fake_python))
        slurm.export_slurm(study, str(out))
    subprocess.run(["bash", str(out)], env={**os.environ, "SLURM_ARRAY_TASK_ID": "0", "JAX_PLATFORM_NAME": "gpu"},
                   capture_output=True, check=True, text=True)
    recorded = json.loads(capture.read_text())
    manifest = planmod.load_manifest(study)
    assert recorded["argv"] == [str(fake_python), "-", str(study)]
    assert recorded["path"] == manifest["repo_root"] + os.pathsep + manifest["simulator_root"]
    assert recorded["jax"] == "cpu" and recorded["cuda"] == ""
    assert recorded["legacy_jax"] == "cpu"
    assert recorded["cwd"] == manifest["repo_root"]
    assert metadata(out.read_text())["python"] == str(fake_python)
    compile(recorded["code"], "<array-dispatch>", "exec")


@pytest.mark.parametrize("index", ["0", "1", "-1", "2", "nonnumeric"])
def test_array_dispatches_exactly_one_planned_job(study, monkeypatch, index):
    out = study / "array.sbatch"
    slurm.export_slurm(study, str(out))
    code = out.read_text().split("<<'ARRAY_EOF'\n", 1)[1].rsplit("ARRAY_EOF\n", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(study)])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", index)
    monkeypatch.setitem(sys.modules, "glucoalg.runtime", SimpleNamespace(initialize_simulator=lambda root: None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: [SimpleNamespace(platform="cpu")]))
    calls = []
    monkeypatch.setattr(os, "execv", lambda *args: calls.append(args))
    if index in {"0", "1"}:
        exec(compile(code, "<array-dispatch>", "exec"), {})
        assert calls == [(sys.executable, [sys.executable, "-m", "glucoalg.tuning", "run",
                          "--plan", str(study), "--job", f"cfg00-s{100 + int(index)}", "--resume"])]
    else:
        with pytest.raises(SystemExit, match="invalid SLURM_ARRAY_TASK_ID"):
            exec(compile(code, "<array-dispatch>", "exec"), {})
        assert calls == []


def test_invalid_directives_do_not_write_script(study):
    out = study / "array.sbatch"
    for kwargs in ({"partition": "cpu\n#SBATCH --exclusive"}, {"job_name": "demo\ncommand"},
                   {"account": "p --exclusive"}, {"cpus_per_task": 0}):
        with pytest.raises(SpecError):
            slurm.export_slurm(study, str(out), **kwargs)
        assert not out.exists()


def test_cli_errors_and_protected_metadata(study, capsys):
    out = study / "array.sbatch"
    assert main(["export-slurm", "--plan", str(study), "--out", str(out),
                 "--max-concurrent", "0"]) == 2
    assert "positive integer" in capsys.readouterr().err
    assert main(["export-slurm", "--plan", str(study), "--out", str(study / "absent" / "array")]) == 1
    assert "cannot write Slurm script" in capsys.readouterr().err
    seal = (study / "plan.json").read_bytes()
    with pytest.raises(SpecError, match="study metadata"):
        slurm.export_slurm(study, str(study / "plan.json"))
    assert (study / "plan.json").read_bytes() == seal


def test_local_cli_needs_no_scheduler_import_or_binary(study):
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\nfrom glucoalg.tuning.__main__ import main\n"
        f"assert main(['run', '--plan', {str(study)!r}, '--all', '--dry-run']) == 0\n"
        "assert 'glucoalg.tuning.slurm' not in sys.modules\n"
        "assert 'glucoalg.tuning.execution_profiles' not in sys.modules\n"
    )
    completed = subprocess.run([sys.executable, "-c", code],
                               env={**os.environ, "PATH": "", "PYTHONPATH": str(root)},
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("[dry-run]") == 2
