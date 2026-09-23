"""Sealed plans, source-content provenance, and durable attempt journals."""
import fcntl
import json
import multiprocessing
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from glucoalg.tuning import plan as planmod
from glucoalg.tuning.plan import RunError, create_plan, load_manifest, check_spec_match
from glucoalg.tuning.spec import SpecError


def base_spec(sim=None):
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
            "vector-env-nums": 1,
            "batch-size": 10,
            "device": "cpu",
        },
        "simulator_root": sim,
    }


def write_spec(tmp_path, spec, name="spec.json"):
    path = Path(tmp_path) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec))
    return str(path)


def test_plan_seals_expected_layout(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    manifest = create_plan(write_spec(tmp_path, base_spec(str(sim))), str(tmp_path / "study"))
    study = tmp_path / "study"
    for name in ("spec.json", "plan.json", "plan.sha256", "PREREG.md", "attempts.jsonl"):
        assert (study / name).exists()
    assert len(manifest["jobs"]) == 4
    assert manifest["jobs"][0]["job_id"] == "cfg00-s100"
    assert manifest["jobs"][0]["expected_epochs"] == 10
    assert manifest["verdict_label"] == "provisional"
    prereg = (study / "PREREG.md").read_text()
    assert manifest["spec_hash"] in prereg and "PROVISIONAL" in prereg
    reloaded = load_manifest(study)
    assert reloaded["spec_hash"] == manifest["spec_hash"]
    assert [j["job_id"] for j in reloaded["jobs"]] == [j["job_id"] for j in manifest["jobs"]]


def test_plan_determinism_across_directories(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    spec_path = write_spec(tmp_path, base_spec(str(sim)))
    first = create_plan(spec_path, str(tmp_path / "s1"))
    second = create_plan(spec_path, str(tmp_path / "s2"))
    assert first["spec_hash"] == second["spec_hash"]
    assert first["jobs"] == second["jobs"]
    assert first["configs"] == second["configs"]


def test_plan_refuses_nonempty_directory(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    out = tmp_path / "study"
    out.mkdir()
    (out / "junk.txt").write_text("x")
    with pytest.raises(SpecError, match="non-empty"):
        create_plan(write_spec(tmp_path, base_spec(str(sim))), str(out))


def test_simulator_resolution_rules(tmp_path):
    spec_path = write_spec(tmp_path, base_spec(None))
    with pytest.raises(SpecError, match="simulator root required"):
        create_plan(spec_path, str(tmp_path / "s"))
    sim = tmp_path / "sim"
    sim.mkdir()
    create_plan(spec_path, str(tmp_path / "s1"), simulator_root=str(sim))
    with pytest.raises(SpecError, match="mismatches"):
        create_plan(write_spec(tmp_path, base_spec(str(sim))), str(tmp_path / "s2"),
                    simulator_root=str(tmp_path / "other"))


def test_tampered_manifest_rejected(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    study = tmp_path / "study"
    create_plan(write_spec(tmp_path, base_spec(str(sim))), str(study))
    plan_file = study / "plan.json"
    plan_file.write_text(plan_file.read_text().replace("demo", "evil"))
    with pytest.raises(RunError, match="tampered"):
        load_manifest(study)


def test_spec_plan_mismatch_rejected(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    study = tmp_path / "study"
    spec_path = write_spec(tmp_path, base_spec(str(sim)))
    manifest = create_plan(spec_path, str(study))
    check_spec_match(manifest, spec_path)  # identical spec passes
    altered = base_spec(str(sim))
    altered["factors"] = {"actor-lr": [0.0003]}
    with pytest.raises(SpecError, match="mismatch"):
        check_spec_match(manifest, write_spec(tmp_path, altered, name="other.json"))


def test_empty_plan_seal_rejected(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    study = tmp_path / "study"
    create_plan(write_spec(tmp_path, base_spec(str(sim))), str(study))
    (study / "plan.sha256").write_text("\n")
    with pytest.raises(RunError, match="seal"):
        load_manifest(study)


@pytest.fixture
def source_trees(tmp_path, monkeypatch):
    """Small real Git trees, independent of the checkout's active changes."""
    roots = {"repo": tmp_path / "repo", "simulator": tmp_path / "sim"}
    packages = {"repo": "glucoalg", "simulator": "glucosim"}
    for name, root in roots.items():
        package = root / packages[name]
        package.mkdir(parents=True)
        (package / "model.py").write_text("RATE = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
             "commit", "--quiet", "-m", "fixture"],
            check=True,
        )
    monkeypatch.setattr(planmod, "repo_root", lambda: roots["repo"])
    return roots, packages


def test_actual_source_probe_with_path_arguments_is_json_serializable(source_trees):
    # Real Git probes and source hashing: CLI pathlib arguments previously
    # leaked a PosixPath into provenance and crashed the first real model pilot.
    roots, _ = source_trees
    provenance = planmod.probe_sources(roots['simulator'], roots['repo'])
    restored = json.loads(json.dumps(provenance, allow_nan=False))
    assert restored == provenance
    assert restored['repo']['path'] == str(roots['repo'])
    assert restored['simulator']['path'] == str(roots['simulator'])
    assert all(len(restored[name]['head']) == 40 for name in ('repo', 'simulator'))


@pytest.mark.parametrize("target", ["repo", "simulator"])
def test_dirty_source_content_changes_hash_with_identical_git_status(source_trees, target):
    roots, packages = source_trees
    source = roots[target] / packages[target] / "model.py"
    source.write_text("RATE = 2\n", encoding="utf-8")
    first = planmod.probe_sources(str(roots["simulator"]))
    source.write_text("RATE = 3\n", encoding="utf-8")
    second = planmod.probe_sources(str(roots["simulator"]))
    assert first[target]["head"] == second[target]["head"]
    assert first[target]["status"] == second[target]["status"]
    assert len(first[target]["content_sha256"]) == 64
    assert first[target]["content_sha256"] != second[target]["content_sha256"]
    other = "simulator" if target == "repo" else "repo"
    assert first[other]["content_sha256"] == second[other]["content_sha256"]


@pytest.mark.parametrize("target", ["repo", "simulator"])
@pytest.mark.parametrize("suffix", [".py", ".yaml", ".toml", ".csv"])
def test_untracked_package_source_content_is_fingerprinted(source_trees, target, suffix):
    roots, packages = source_trees
    source = roots[target] / packages[target] / f"untracked{suffix}"
    baseline = planmod.probe_sources(str(roots["simulator"]))[target]
    source.write_text("value = 10\n", encoding="utf-8")
    first = planmod.probe_sources(str(roots["simulator"]))[target]
    source.write_text("value = 20\n", encoding="utf-8")
    second = planmod.probe_sources(str(roots["simulator"]))[target]
    assert first["head"] == second["head"]
    assert first["status"] == second["status"]
    assert baseline["content_sha256"] != first["content_sha256"]
    assert first["content_sha256"] != second["content_sha256"]


@pytest.mark.parametrize("target", ["repo", "simulator"])
def test_generated_cache_and_output_content_does_not_change_source_hash(source_trees, target):
    roots, packages = source_trees
    root = roots[target]
    package = packages[target]
    first = planmod.probe_sources(str(roots["simulator"]))[target]["content_sha256"]
    for relative in (
        f"{package}/__pycache__/generated.py",
        f"{package}/.pytest_cache/generated.py",
        f"{package}/compiled.pyc",
        "outputs/generated.py",
        "results/parameters.yaml",
    ):
        artifact = root / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("generated content\n", encoding="utf-8")
    second = planmod.probe_sources(str(roots["simulator"]))[target]["content_sha256"]
    assert first == second


def test_job_attempts_collapses_lifecycle_events_and_preserves_incomplete_attempts():
    started = {"job_id": "cfg00-s100", "attempt_no": 1, "status": "started"}
    failed = {**started, "status": "failed", "returncode": 3}
    retried = {**started, "attempt_no": 2}
    other = {"job_id": "cfg00-s101", "attempt_no": 1, "status": "started"}
    records = [started, other, failed, retried]
    assert planmod.job_attempts(records, "cfg00-s100") == [failed, retried]
    assert planmod.job_attempts(records, "cfg00-s101") == [other]
    assert planmod.job_attempts(records, "unknown") == []
    success = {**retried, "status": "success", "returncode": 0}
    records.append(success)
    assert planmod.job_attempts(records, "cfg00-s100") == [failed, success]


def _append_worker(study_dir, worker_id):
    for index in range(12):
        planmod.append_ledger(
            study_dir,
            {"job_id": f"worker{worker_id}", "attempt_no": index + 1,
             "payload": f"{worker_id}:{index}:" + "x" * 32768},
        )


def test_concurrent_ledger_appends_preserve_complete_records(tmp_path):
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_append_worker, str(tmp_path), worker) for worker in range(4)]
        for future in futures:
            future.result(timeout=20)
    records = planmod.read_ledger(tmp_path)
    assert len(records) == 48
    assert {(rec["job_id"], rec["attempt_no"]) for rec in records} == {
        (f"worker{worker}", index + 1) for worker in range(4) for index in range(12)
    }
    for record in records:
        worker = record["job_id"].removeprefix("worker")
        index = record["attempt_no"] - 1
        assert record["payload"] == f"{worker}:{index}:" + "x" * 32768


def _ledger_operation(study_dir, operation, started, completed):
    started.set()
    if operation == "read":
        planmod.read_ledger(study_dir)
    else:
        planmod.append_ledger(study_dir, {"job_id": "child", "attempt_no": 1})
    completed.set()


@pytest.mark.parametrize("operation", ["read", "append"])
def test_ledger_operations_wait_for_process_lock(tmp_path, operation):
    planmod.append_ledger(tmp_path, {"job_id": "parent", "attempt_no": 1})
    ctx = multiprocessing.get_context("spawn")
    started, completed = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_ledger_operation,
                        args=(str(tmp_path), operation, started, completed))
    with (tmp_path / planmod.LEDGER_NAME).open("r+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        child.start()
        try:
            assert started.wait(5), "ledger worker did not start"
            assert not completed.wait(0.2), "ledger operation ignored the process lock"
        finally:
            fcntl.flock(ledger, fcntl.LOCK_UN)
            child.join(5)
            if child.is_alive():
                child.terminate()
                child.join(5)
    assert child.exitcode == 0
    assert completed.is_set()


def test_append_flushes_record_before_fsync(tmp_path, monkeypatch):
    synced_records = []
    actual_fsync = os.fsync

    def inspect_fsync(fd):
        synced_records.append((tmp_path / planmod.LEDGER_NAME).read_text(encoding="utf-8"))
        actual_fsync(fd)

    monkeypatch.setattr(planmod.os, "fsync", inspect_fsync)
    record = {"job_id": "cfg00-s100", "attempt_no": 1, "status": "started"}
    planmod.append_ledger(tmp_path, record)
    assert synced_records, "attempt journal was not fsynced"
    assert all(json.loads(blob) == record for blob in synced_records)
