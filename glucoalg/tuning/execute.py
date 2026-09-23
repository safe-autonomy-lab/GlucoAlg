"""Launch planned seed jobs: subprocess runs, ledger, retry/resume."""
from __future__ import annotations

import os
import fcntl
import shlex
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

from . import plan as planmod
from .plan import MetricError, RunError, job_attempt_dir, job_log_file
from .spec import KNOWN_OPTIONS, SpecError, encode_level
from .summarize import verify_progress

TEST_RUNNER_PREFIX = "test:"


def check_ledger_plan(records: list[dict], manifest: dict) -> None:
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    for rec in records:
        if rec.get("plan_hash") != manifest["_plan_hash"]:
            raise RunError(
                "attempt ledger holds trials from another plan (plan_hash mismatch); "
                "old trials are never reused silently — replan into a fresh directory"
            )
        job = jobs.get(rec.get("job_id"))
        if job is None or any(rec.get(key) != job[key] for key in ("config_id", "seed")):
            raise RunError("attempt ledger job/config/seed does not match the sealed plan")
        if type(rec.get("attempt_no")) is not int or rec["attempt_no"] <= 0:
            raise RunError("attempt ledger has an invalid attempt number")


def check_source_provenance(manifest: dict, provenance: dict) -> None:
    """Require the same runtime source bytes that were sealed in the plan."""
    for name in ("repo", "simulator"):
        expected = manifest.get("provenance", {}).get(name, {})
        observed = provenance.get(name, {})
        if "content_sha256" not in expected or "content_sha256" not in observed:
            raise RunError("source content fingerprints missing; create a fresh plan")
        if observed["content_sha256"] != expected["content_sha256"]:
            raise RunError(f"{name} source content changed from sealed plan; create a fresh plan")


@contextmanager
def _job_lock(study_dir: str | Path, job_id: str):
    directory = Path(study_dir) / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ".launch.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunError(f"{job_id}: already running (job lock held)") from exc
        yield fd
    finally:
        # Do not explicitly LOCK_UN: a live child inherits this open description
        # and must retain the claim if the parent is interrupted or killed.
        os.close(fd)


def _job_by_id(manifest: dict, job_id: str) -> dict:
    for job in manifest["jobs"]:
        if job["job_id"] == job_id:
            return job
    raise SpecError(f"unknown job ID {job_id!r}")


def _config_by_id(manifest: dict, config_id: str) -> dict:
    for cfg in manifest["configs"]:
        if cfg["config_id"] == config_id:
            return cfg
    raise RunError(f"plan corrupt: unknown config ID {config_id!r}")


def verify_recorded_success(manifest: dict, study_dir: str | Path, record: dict) -> tuple[bool, str]:
    """Re-verify the files behind a ledger success claim (resume gate)."""
    if record.get("status") != "success" or record.get("returncode") != 0:
        return False, "ledger does not claim success"
    try:
        check_ledger_plan([record], manifest)
        check_source_provenance(manifest, record.get("provenance", {}))
        check_source_provenance(manifest, record.get("provenance_after", {}))
        job = _job_by_id(manifest, record["job_id"])
        recorded_hash = record.get("progress_sha256")
        if not recorded_hash:
            raise MetricError("successful attempt has no progress.csv hash")
        verify_progress(str(Path(study_dir) / record["log_dir"]), job["expected_epochs"],
                        expected_sha256=recorded_hash)
    except RunError as exc:
        return False, f"prior success no longer verifies: {exc}"
    return True, "ledger success re-verified on disk"


def job_state(
    manifest: dict, study_dir: str | Path, records: list[dict], job_id: str
) -> tuple[str, str]:
    """Return (pending|success|failed, reason) for one job."""
    attempts = planmod.job_attempts(records, job_id)
    if not attempts:
        return "pending", "never attempted"
    latest = attempts[-1]
    if latest.get("status") == "started":
        return "failed", "unfinished attempt; if no job is running, use --retry-failed for a fresh attempt"
    if latest.get("status") != "success" or latest.get("returncode") != 0:
        return "failed", latest.get("fail_reason") or f"exit={latest.get('returncode')}"
    ok, reason = verify_recorded_success(manifest, study_dir, latest)
    return ("success", reason) if ok else ("failed", reason)


def resolve_runner(manifest: dict, cli_runner: str | None = None) -> str:
    choice = cli_runner or manifest.get("runner", "auto")
    if choice not in ("auto", "glucoalg.train", "run.py"):
        raise SpecError(f"runner must be one of auto/glucoalg.train/run.py, got {choice!r}")
    if choice != "auto":
        return choice
    root = Path(manifest["repo_root"])
    if (root / "glucoalg" / "train.py").exists():
        return "glucoalg.train"
    if (root / "run.py").exists():
        return "run.py"
    raise RunError(f"no training runner found under repo root {root}")


def build_argv(
    manifest: dict,
    job_id: str,
    *,
    simulator_root: str,
    runner: str,
    log_dir: str,
    runner_cmd: str | None = None,
) -> tuple[list[str], str]:
    """Build a safe (shell-free) training argv plus the recorded runner label."""
    job = _job_by_id(manifest, job_id)
    cfg = _config_by_id(manifest, job["config_id"])
    if runner_cmd:
        argv = [sys.executable, runner_cmd]
        label = f"{TEST_RUNNER_PREFIX}{runner_cmd}"
    elif runner == "glucoalg.train":
        argv = [sys.executable, "-m", "glucoalg.train"]
        label = runner
    else:
        argv = [sys.executable, str(Path(manifest["repo_root"]) / "run.py")]
        label = runner
    opts = dict(manifest["train"])
    opts.update(cfg["params"])
    for key in sorted(opts):
        argv += [KNOWN_OPTIONS[key], encode_level(opts[key])]
    argv += ["--seed", str(job["seed"]), "--simulator-root", simulator_root,
             "--log-dir", str(log_dir)]
    return argv, label


def launch_one(
    manifest: dict,
    study_dir: str | Path,
    job_id: str,
    attempt_no: int,
    argv: list[str],
    runner_label: str,
    simulator_root: str,
    *,
    lock_fd: int | None = None,
) -> dict:
    if lock_fd is None:
        with _job_lock(study_dir, job_id) as held:
            return launch_one(manifest, study_dir, job_id, attempt_no, argv,
                              runner_label, simulator_root, lock_fd=held)
    study = Path(study_dir)
    log_dir = job_attempt_dir(study, job_id, attempt_no)
    log_file = job_log_file(study, job_id, attempt_no)
    if log_dir.exists() or log_file.exists():
        raise RunError(f"refusing to overwrite existing attempt {job_id}/attempt{attempt_no}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{manifest['repo_root']}{os.pathsep}{simulator_root}"
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    job = _job_by_id(manifest, job_id)
    record: dict = {
        "job_id": job_id,
        "config_id": job["config_id"],
        "seed": job["seed"],
        "attempt_no": attempt_no,
        "plan_hash": manifest["_plan_hash"],
        "started_utc": planmod.utc_now(),
        "runner_used": runner_label,
        "argv": argv,
        "cwd": str(log_dir),
        "env": {key: env[key] for key in (
            "PYTHONPATH", "JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES",
        )},
        "simulator_root": simulator_root,
        "log_dir": str(log_dir.relative_to(study)),
        "log_file": str(log_file.relative_to(study)),
        "provenance": planmod.probe_sources(simulator_root, manifest["repo_root"]),
        "status": "started", "returncode": None, "fail_reason": None,
        "progress_csv": None, "progress_sha256": None, "metrics": None,
    }
    check_source_provenance(manifest, record["provenance"])
    # Persist the claim before creating outputs or starting a process. A killed
    # driver leaves an unfinished attempt that requires an explicit fresh retry.
    planmod.append_ledger(study, record)
    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        with open(log_file, "x", encoding="utf-8") as logfh:
            completed = subprocess.run(argv, cwd=str(log_dir), env=env,
                                       stdout=logfh, stderr=subprocess.STDOUT,
                                       pass_fds=(lock_fd,))
        record["returncode"] = completed.returncode
    except OSError as exc:
        record["returncode"] = None
        record["status"] = "failed"
        record["fail_reason"] = f"spawn failed: {exc}"
        record["ended_utc"] = planmod.utc_now()
        planmod.append_ledger(study, record)
        return record
    except KeyboardInterrupt:
        record.update(status="failed", fail_reason="driver interrupted", ended_utc=planmod.utc_now())
        planmod.append_ledger(study, record)
        raise
    record["ended_utc"] = planmod.utc_now()
    try:
        record["provenance_after"] = planmod.probe_sources(simulator_root, manifest["repo_root"])
        check_source_provenance(manifest, record["provenance_after"])
    except RunError as exc:
        record["status"] = "failed"
        record["fail_reason"] = str(exc)
    else:
        record["status"] = "success" if record["returncode"] == 0 else "failed"
    if record["returncode"] != 0:
        record["status"] = "failed"
        record["fail_reason"] = f"trainer exit={record['returncode']} (see {record['log_file']})"
    elif record["status"] == "success":
        try:
            metrics = verify_progress(str(log_dir), job["expected_epochs"])
        except MetricError as exc:
            record["status"] = "failed"
            record["fail_reason"] = f"unverifiable output: {exc}"
            record["progress_csv"] = None
            record["metrics"] = None
        else:
            record["status"] = "success"
            record["fail_reason"] = None
            record["progress_csv"] = str(Path(metrics["progress_csv"]).relative_to(study))
            record["progress_sha256"] = metrics["progress_sha256"]
            record["metrics"] = {key: metrics[key] for key in ("n_rows", "window", "epret", "epcost")}
    planmod.append_ledger(study, record)
    return record


def run_study(
    study_dir: str | Path,
    *,
    all_jobs: bool = False,
    job_id: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    simulator_root: str | None = None,
    runner: str | None = None,
    runner_cmd: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run pending jobs (verified successes skipped, failures need explicit retry)."""
    manifest = planmod.load_manifest(study_dir)
    records = planmod.read_ledger(study_dir)
    check_ledger_plan(records, manifest)
    if simulator_root and os.path.abspath(simulator_root) != os.path.abspath(
        manifest["simulator_root"]
    ):
        raise SpecError("--simulator-root mismatches the sealed plan")
    if bool(job_id) == bool(all_jobs):
        raise SpecError("pass exactly one of --job JOB_ID or --all")
    if job_id is not None:
        selected = [_job_by_id(manifest, job_id)["job_id"]]
    else:
        selected = [job["job_id"] for job in manifest["jobs"]]
        if records and not resume and not dry_run:
            raise RunError(
                "study already started; pass --resume to continue "
                "(verified successes are skipped, failures still need --retry-failed)"
            )
    sim_root = manifest["simulator_root"]
    if not dry_run and not runner_cmd and not (Path(sim_root) / "glucosim" / "__init__.py").is_file():
        raise RunError(f"simulator root missing glucosim package: {sim_root}")
    if runner_cmd and not os.path.isfile(runner_cmd):
        raise RunError(f"--runner-cmd not found: {runner_cmd}")
    resolved_runner = resolve_runner(manifest, runner)
    summary: dict = {"ran": [], "succeeded": [], "failed": [],
                     "skipped_success": [], "skipped_failed": []}
    for jid in selected:
        with nullcontext(None) if dry_run else _job_lock(study_dir, jid) as held:
            # Another worker may have finished while this caller was preparing.
            records = planmod.read_ledger(study_dir)
            check_ledger_plan(records, manifest)
            state, reason = job_state(manifest, study_dir, records, jid)
            if state == "success":
                summary["skipped_success"].append(jid)
                print(f"[skip] {jid}: verified success ({reason})")
                continue
            if state == "failed" and not retry_failed:
                if job_id is not None:
                    raise RunError(f"{jid}: last attempt failed ({reason}); pass --retry-failed to retry")
                summary["skipped_failed"].append(jid)
                print(f"[skip] {jid}: failed ({reason}); needs --retry-failed")
                continue
            attempts = planmod.job_attempts(records, jid)
            attempt_no = max((rec["attempt_no"] for rec in attempts), default=0) + 1
            log_dir = job_attempt_dir(study_dir, jid, attempt_no)
            argv, label = build_argv(
                manifest, jid, simulator_root=sim_root, runner=resolved_runner,
                log_dir=str(log_dir.absolute()),
                runner_cmd=os.path.abspath(runner_cmd) if runner_cmd else None,
            )
            if dry_run:
                print(f"[dry-run] {jid}: {' '.join(shlex.quote(a) for a in argv)}")
                summary["ran"].append(jid)
                continue
            record = launch_one(manifest, study_dir, jid, attempt_no, argv, label, sim_root, lock_fd=held)
            summary["ran"].append(jid)
            bucket = "succeeded" if record["status"] == "success" else "failed"
            summary[bucket].append(jid)
            detail = record["fail_reason"] or "output verified"
            print(f"[{record['status']}] {jid} attempt{attempt_no}: {detail}")
    failed = summary["failed"] + summary["skipped_failed"]
    if failed:
        raise RunError(f"{len(failed)} job(s) failed or require --retry-failed: {', '.join(failed)}")
    return summary


def export_slurm(
    study_dir: str | Path,
    out_path: str,
    *,
    partition: str | None = None,
    time: str | None = None,
    job_name: str | None = None,
    exclusive: bool | None = None,
    profile: str | Path | None = None,
    account: str | None = None,
    qos: str | None = None,
    cpus_per_task: int | None = None,
    memory: str | None = None,
    max_concurrent: int | None = None,
) -> str:
    """Compatibility entry point; scheduler rendering lives in :mod:`.slurm`."""
    from .slurm import export_slurm as render

    return render(
        study_dir, out_path, profile=profile, partition=partition, time=time,
        job_name=job_name, exclusive=exclusive, account=account, qos=qos,
        cpus_per_task=cpus_per_task, memory=memory, max_concurrent=max_concurrent,
    )
