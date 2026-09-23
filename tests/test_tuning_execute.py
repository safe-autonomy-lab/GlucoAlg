"""Runs via fake trainers + subprocess mocking (no simulator, no Slurm submit)."""
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from glucoalg.tuning import execute as execmod
from glucoalg.tuning import plan as planmod
from glucoalg.tuning.plan import RunError, create_plan, load_manifest, read_ledger
from glucoalg.tuning.spec import SpecError

FAKE_TRAINER = """
import argparse, csv, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", required=True)
ap.add_argument("--total-steps", type=int, required=True)
ap.add_argument("--steps-per-epoch", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--actor-lr", type=float, default=1e-5)
ap.add_argument("--critic-lr", type=float, default=5e-5)
args, _ = ap.parse_known_args()
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "fail":
    print("boom", file=sys.stderr)
    sys.exit(3)
n = args.total_steps // args.steps_per_epoch
if mode == "short":
    n = n // 2
d = os.path.join(args.log_dir, "runs", "fake", "seed-%d" % args.seed)
os.makedirs(d, exist_ok=True)
if mode == "nocsv":
    sys.exit(0)
rows = []
for _ in range(n):
    rows.append((1000.0 * args.actor_lr + args.seed,
                 100000.0 * args.critic_lr + args.seed * 0.001))
if mode == "nonfinite" and rows:
    rows[-1] = (float("nan"), rows[-1][1])
with open(os.path.join(d, "progress.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Metrics/EpRet", "Metrics/EpCost"])
    w.writerows(rows)
"""


def base_spec(sim):
    return {
        "study_name": "demo",
        "seeds": [100, 101],
        "cost_limit": 100.0,
        "factors": {"actor-lr": [0.0003, 3e-05]},
        "train": {
            "algo": "PPOLag",
            "env-id": "t1d-v0",
            "cohort": "adolescent",
            "total-steps": 100,
            "steps-per-epoch": 10,
            "batch-size": 10,
            "vector-env-nums": 1,
            "device": "cpu",
            "project-name": "demo [brackets] spaces",
        },
    }


@pytest.fixture()
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "ok")
    sim = tmp_path / "sim"
    sim.mkdir()
    # Freeze a tiny source tree so unrelated worker edits cannot affect tests.
    repo = tmp_path / "repo"
    (repo / "glucoalg").mkdir(parents=True)
    (repo / "glucoalg" / "train.py").write_text("# frozen test source\n")
    monkeypatch.setattr(planmod, "repo_root", lambda: repo)
    spec_path = tmp_path / "spec.json"
    spec = base_spec(str(sim))
    spec["simulator_root"] = str(sim)
    spec_path.write_text(json.dumps(spec))
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(FAKE_TRAINER)
    out = tmp_path / "study"
    create_plan(str(spec_path), str(out))
    return {"dir": out, "trainer": str(trainer), "sim": str(sim), "spec": str(spec_path), "repo": repo}


def test_run_all_success_and_ledger(study):
    summary = execmod.run_study(study["dir"], all_jobs=True, runner_cmd=study["trainer"])
    assert len(summary["succeeded"]) == 4
    records = read_ledger(study["dir"])
    assert len(records) == 8
    assert records[0]["status"] == "started"
    first = records[1]
    assert first["status"] == "success" and first["returncode"] == 0
    assert first["argv"][0] == sys.executable
    assert "--seed" in first["argv"] and "--simulator-root" in first["argv"]
    assert "--log-dir" in first["argv"] and "--actor-lr" in first["argv"]
    # values with spaces travel as single argv items (shell-free)
    assert "demo [brackets] spaces" in first["argv"]
    assert first["env"]["PYTHONPATH"].endswith(study["sim"])
    assert first["env"]["JAX_PLATFORMS"] == "cpu"
    assert first["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert first["runner_used"].startswith("test:")
    assert first["metrics"]["n_rows"] == 10
    assert len(first["progress_sha256"]) == 64
    log = Path(study["dir"]) / first["log_file"]
    assert log.exists()


def test_run_single_job_then_resume_skips_success(study):
    execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    assert len(read_ledger(study["dir"])) == 2
    with pytest.raises(RunError, match="already started"):
        execmod.run_study(study["dir"], all_jobs=True, runner_cmd=study["trainer"])
    summary = execmod.run_study(study["dir"], all_jobs=True, resume=True,
                                runner_cmd=study["trainer"])
    assert len(summary["skipped_success"]) == 1
    assert len(summary["succeeded"]) == 3
    assert len(read_ledger(study["dir"])) == 8


def test_failed_attempt_bookkeeping_and_explicit_retry(study, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(RunError, match="4 job"):
        execmod.run_study(study["dir"], all_jobs=True, runner_cmd=study["trainer"])
    records = read_ledger(study["dir"])
    assert len(records) == 8
    assert all(r["status"] == "failed" and r["returncode"] == 3 for r in records[1::2])
    # without --retry-failed nothing reruns
    with pytest.raises(RunError, match="retry-failed"):
        execmod.run_study(study["dir"], all_jobs=True, resume=True, runner_cmd=study["trainer"])
    assert len(read_ledger(study["dir"])) == 8
    with pytest.raises(RunError, match="retry-failed"):
        execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    # explicit retry with a working trainer records attempt 2
    monkeypatch.setenv("FAKE_MODE", "ok")
    summary = execmod.run_study(study["dir"], all_jobs=True, resume=True,
                                retry_failed=True, runner_cmd=study["trainer"])
    assert len(summary["succeeded"]) == 4
    records = read_ledger(study["dir"])
    assert len(records) == 16
    assert [r["attempt_no"] for r in planmod.job_attempts(records, "cfg00-s100")] == [1, 2]
    # verified successes are never rerun, even with --retry-failed
    summary = execmod.run_study(study["dir"], all_jobs=True, resume=True,
                                retry_failed=True, runner_cmd=study["trainer"])
    assert len(summary["skipped_success"]) == 4
    assert len(read_ledger(study["dir"])) == 16


def test_unverifiable_output_counts_as_failed(study, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "nocsv")
    with pytest.raises(RunError):
        execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    record = read_ledger(study["dir"])[-1]
    assert record["status"] == "failed"
    assert "progress.csv" in record["fail_reason"]


def test_invalidated_success_becomes_failed(study):
    execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    manifest = load_manifest(study["dir"])
    state, _ = execmod.job_state(manifest, study["dir"], read_ledger(study["dir"]), "cfg00-s100")
    assert state == "success"
    csv_path = next(Path(study["dir"]).glob("jobs/cfg00-s100/attempt1/**/progress.csv"))
    csv_path.unlink()
    state, reason = execmod.job_state(manifest, study["dir"], read_ledger(study["dir"]), "cfg00-s100")
    assert state == "failed" and "no longer verifies" in reason


def test_dry_run_launches_nothing(study):
    summary = execmod.run_study(study["dir"], all_jobs=True, dry_run=True)
    assert len(summary["ran"]) == 4
    assert read_ledger(study["dir"]) == []
    assert list(Path(study["dir"], "jobs").iterdir()) == []


def test_local_child_overrides_inherited_gpu_environment(study, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    trainer = Path(study["trainer"])
    trainer.write_text(FAKE_TRAINER + (
        "\nprint('child-platforms=' + os.environ['JAX_PLATFORMS'])\n"
        "print('child-platform=' + os.environ['JAX_PLATFORM_NAME'])\n"
        "print('child-cuda=' + repr(os.environ['CUDA_VISIBLE_DEVICES']))\n"
    ))
    execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=str(trainer))
    record = read_ledger(study["dir"])[-1]
    assert record["env"]["JAX_PLATFORM_NAME"] == "cpu"
    log = (study["dir"] / record["log_file"]).read_text()
    assert "child-platforms=cpu" in log
    assert "child-platform=cpu" in log
    assert "child-cuda=''" in log


def test_launch_guards(study):
    with pytest.raises(SpecError, match="unknown job"):
        execmod.run_study(study["dir"], job_id="cfg99-s100", dry_run=True)
    with pytest.raises(SpecError, match="mismatches"):
        execmod.run_study(study["dir"], all_jobs=True, dry_run=True,
                          simulator_root="/definitely/not/the/plan")
    with pytest.raises(RunError, match="runner-cmd"):
        execmod.run_study(study["dir"], all_jobs=True, runner_cmd="/nope.py")
    with pytest.raises(SpecError, match="exactly one"):
        execmod.run_study(study["dir"], dry_run=True)


def test_foreign_ledger_rejected(study):
    execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    ledger = Path(study["dir"]) / "attempts.jsonl"
    ledger.write_text(ledger.read_text().replace(
        json.loads(ledger.read_text().splitlines()[0])["plan_hash"], "0" * 64))
    with pytest.raises(RunError, match="another plan"):
        execmod.run_study(study["dir"], all_jobs=True, resume=True,
                          runner_cmd=study["trainer"])


def test_runner_auto_detect_matches_files(study):
    manifest = load_manifest(study["dir"])
    root = Path(manifest["repo_root"])
    expected = "glucoalg.train" if (root / "glucoalg" / "train.py").exists() else "run.py"
    assert execmod.resolve_runner(manifest) == expected
    argv, label = execmod.build_argv(manifest, "cfg00-s100", simulator_root=study["sim"],
                                    runner="run.py", log_dir="/tmp/x")
    assert argv[:2] == [sys.executable, str(root / "run.py")]
    assert label == "run.py"
    argv, _ = execmod.build_argv(manifest, "cfg00-s100", simulator_root=study["sim"],
                                runner="glucoalg.train", log_dir="/tmp/x")
    assert argv[:3] == [sys.executable, "-m", "glucoalg.train"]


def test_subprocess_invoked_shell_free(study):
    manifest = load_manifest(study["dir"])
    with mock.patch.object(execmod.subprocess, "run") as mocked:
        mocked.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        record = execmod.launch_one(manifest, study["dir"], "cfg00-s100", 1,
                                    [sys.executable, "fake"], "test:fake", study["sim"])
    call = next(call for call in mocked.call_args_list if call.args[0] == [sys.executable, "fake"])
    kwargs = call.kwargs
    assert "shell" not in kwargs
    assert isinstance(call.args[0], list)
    assert len(kwargs["pass_fds"]) == 1
    # exit 0 but no progress.csv written by the mock => failed, still ledgered
    assert record["status"] == "failed" and "progress.csv" in record["fail_reason"]
    assert len(read_ledger(study["dir"])) == 2


def test_export_slurm_renders_quoted_script_never_submits(study):
    out = str(Path(study["dir"]) / "tune.sbatch")
    execmod.export_slurm(study["dir"], out, partition="cpu", time="01:00:00", exclusive=True)
    text = Path(out).read_text()
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --exclusive" in text
    assert "PYTHONPATH=" in text and study["sim"] in text
    assert "#SBATCH --array=0-3" in text
    assert "--cpus-per-task" not in text
    assert "'--job', job_id" in text and "--all" not in text
    assert "check_source_provenance" in text
    assert "never submits" in text
    assert "\nsbatch " not in text and text.startswith("#!/bin/bash")
    if shutil.which("sh"):
        subprocess.run(["sh", "-n", out], check=True, capture_output=True)

def test_export_slurm_sanitizes_directives(study):
    out = str(Path(study["dir"]) / "x.sbatch")
    execmod.export_slurm(study["dir"], out, job_name="my job")
    assert "#SBATCH -J my_job" in Path(out).read_text()
    with pytest.raises(SpecError, match="single token"):
        execmod.export_slurm(study["dir"], out, partition="c pu")


def test_changed_finite_metrics_invalidate_recorded_success(study):
    execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    record = read_ledger(study["dir"])[-1]
    csv_path = Path(study["dir"]) / record["progress_csv"]
    csv_path.write_text("Metrics/EpRet,Metrics/EpCost\n" + "999,1\n" * 10)
    state, reason = execmod.job_state(load_manifest(study["dir"]), study["dir"],
                                    read_ledger(study["dir"]), "cfg00-s100")
    assert state == "failed" and "hash" in reason.lower()


def test_source_change_blocks_launch_without_new_attempt(study):
    (study["repo"] / "glucoalg" / "train.py").write_text("# changed after sealing\n")
    with pytest.raises(RunError, match="source content changed"):
        execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    assert read_ledger(study["dir"]) == []


def test_source_change_during_execution_records_failure(study):
    source = study["repo"] / "glucoalg" / "train.py"
    trainer = Path(study["trainer"])
    trainer.write_text(FAKE_TRAINER + f"\nopen({str(source)!r}, 'w').write('# changed during run')\n")
    with pytest.raises(RunError, match="1 job"):
        execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    record = read_ledger(study["dir"])[-1]
    assert record["status"] == "failed"
    assert "source content changed" in record["fail_reason"]
    assert record["metrics"] is None


def test_existing_attempt_output_is_never_overwritten(study):
    attempt = Path(study["dir"]) / "jobs" / "cfg00-s100" / "attempt1"
    attempt.mkdir(parents=True)
    marker = attempt / "old-output"
    marker.write_text("preserve")
    with pytest.raises(RunError, match="refusing to overwrite"):
        execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    assert marker.read_text() == "preserve"
    assert read_ledger(study["dir"]) == []


def test_start_is_journaled_before_subprocess(study, monkeypatch):
    original = subprocess.run

    def run(argv, **kwargs):
        if argv[:2] == [sys.executable, study["trainer"]]:
            record = read_ledger(study["dir"])[-1]
            assert record["status"] == "started" and record["attempt_no"] == 1
            assert len(kwargs["pass_fds"]) == 1
            os.fstat(kwargs["pass_fds"][0])
        return original(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
    records = read_ledger(study["dir"])
    assert [record["status"] for record in records] == ["started", "success"]
    assert len(planmod.job_attempts(records, "cfg00-s100")) == 1


def test_killed_driver_retains_child_lock_then_requires_fresh_retry(study, tmp_path):
    """A duplicate launch cannot overlap an orphan trainer; other seed jobs can run."""
    marker = tmp_path / "child-started"
    slow = tmp_path / "slow_trainer.py"
    slow.write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n" + FAKE_TRAINER
    )
    driver_code = (
        "from glucoalg.tuning.execute import run_study\n"
        f"run_study({str(study['dir'])!r}, job_id='cfg00-s100', runner_cmd={str(slow)!r})\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    driver = subprocess.Popen([sys.executable, "-c", driver_code], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline and driver.poll() is None:
            time.sleep(0.02)
        assert marker.exists(), driver.communicate(timeout=1)
        driver.kill()
        driver.communicate(timeout=5)
        records = read_ledger(study["dir"])
        assert len(records) == 1 and records[0]["status"] == "started"
        with pytest.raises(RunError, match="already running"):
            execmod.run_study(study["dir"], job_id="cfg00-s100", retry_failed=True,
                              runner_cmd=study["trainer"])
        # A separate job has its own lock and may complete while the first lives.
        other = execmod.run_study(study["dir"], job_id="cfg00-s101", runner_cmd=study["trainer"])
        assert other["succeeded"] == ["cfg00-s101"]
        os.killpg(driver.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while True:
            try:
                with execmod._job_lock(study["dir"], "cfg00-s100"):
                    break
            except RunError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        with pytest.raises(RunError, match="retry-failed"):
            execmod.run_study(study["dir"], job_id="cfg00-s100", runner_cmd=study["trainer"])
        execmod.run_study(study["dir"], job_id="cfg00-s100", retry_failed=True,
                          runner_cmd=study["trainer"])
        attempts = planmod.job_attempts(read_ledger(study["dir"]), "cfg00-s100")
        assert [(a["attempt_no"], a["status"]) for a in attempts] == [(1, "started"), (2, "success")]
        assert (Path(study["dir"]) / "jobs" / "cfg00-s100" / "attempt1").is_dir()
        assert (Path(study["dir"]) / "jobs" / "cfg00-s100" / "attempt2").is_dir()
    finally:
        try:
            os.killpg(driver.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if driver.poll() is None:
            driver.communicate(timeout=5)
