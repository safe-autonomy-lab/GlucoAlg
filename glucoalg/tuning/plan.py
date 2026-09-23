"""Sealed plan manifests, attempt ledger IO, and preregistration template."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .spec import (
    SpecError,
    expand_configs,
    expected_epochs,
    load_json_nodupes,
    normalize_spec,
    spec_hash,
)

PROTOCOL = "glucoalg-tuning/1"

SELECTION_RULE = (
    "Among complete configs (every seed valid) with mean EpCost <= cost_limit, "
    "pick the highest mean EpRet; tie-break by lower mean EpCost, then config_id. "
    "No feasible config => no winner."
)

REQUIRES_CONFIRMATION = [
    "held-out training-seed confirmation (fresh seed, same config and budget)",
    "unseen-patient evaluation (TIR / Risk Index via the separate evaluator)",
]

LEDGER_NAME = "attempts.jsonl"
MANIFEST_NAME = "plan.json"
MANIFEST_HASH_NAME = "plan.sha256"
SPEC_NAME = "spec.json"
PREREG_NAME = "PREREG.md"


class RunError(RuntimeError):
    """Runtime/launch failure (CLI exit code 1)."""


class MetricError(RunError):
    """Metric verification failure: a job is incomplete/unverifiable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_probe(path: str) -> dict:
    probe = {"path": path, "head": None, "status": None}
    try:
        head = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if head.returncode == 0:
            probe["head"] = (head.stdout or "").strip() or None
        status = subprocess.run(
            ["git", "-C", path, "status", "--short"],
            capture_output=True, text=True, timeout=15,
        )
        if status.returncode == 0:
            probe["status"] = (status.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return probe


def _content_hash(root: Path, packages: tuple[str, ...], *, root_sources: bool = False) -> str | None:
    """Hash relative source names and bytes, including untracked package data."""
    if not root.is_dir():
        return None
    paths = set()
    if root_sources:
        paths.update(root.glob("*.py"))
        paths.update(root.glob("*.toml"))
    for package in packages:
        directory = root / package
        if directory.is_dir():
            paths.update(directory.rglob("*"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root)
        if (not path.is_file() or path.suffix in {".pyc", ".pyo"}
                or any(part in {"__pycache__", ".pytest_cache", ".git"}
                       or part.endswith(".egg-info") for part in relative.parts)):
            continue
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise RunError(f"cannot fingerprint source {path}: {exc}") from exc
        digest.update(str(relative).encode("utf-8") + b"\0")
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return digest.hexdigest()


def probe_sources(sim_root: str | Path, repo_path: str | Path | None = None) -> dict:
    root = Path(repo_path) if repo_path is not None else repo_root()
    repo = _git_probe(str(root))
    repo["content_sha256"] = _content_hash(
        root, ("glucoalg", "glucobench", "omnisafe", "FunctionEncoder", "shield"),
        root_sources=True,
    )
    simulator = _git_probe(str(sim_root))
    simulator["content_sha256"] = _content_hash(Path(sim_root), ("glucosim",))
    return {
        "repo": repo,
        "simulator": simulator,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
    }


def resolve_simulator_root(spec: dict, cli_value: str | None) -> str:
    from_spec = spec.get("simulator_root")
    if cli_value and from_spec:
        if os.path.abspath(cli_value) != os.path.abspath(from_spec):
            raise SpecError(
                f"--simulator-root {cli_value!r} mismatches spec simulator_root {from_spec!r}"
            )
    chosen = cli_value or from_spec
    if not chosen:
        raise SpecError("simulator root required: pass --simulator-root or set spec simulator_root")
    return os.path.abspath(chosen)


def job_attempt_dir(study_dir: str | Path, job_id: str, attempt_no: int) -> Path:
    return Path(study_dir) / "jobs" / job_id / f"attempt{attempt_no}"


def job_log_file(study_dir: str | Path, job_id: str, attempt_no: int) -> Path:
    return Path(study_dir) / "jobs" / job_id / f"attempt{attempt_no}.log"


def read_ledger(study_dir: str | Path) -> list[dict]:
    path = Path(study_dir) / LEDGER_NAME
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        text = fh.read()
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            records.append(record)
        except ValueError as exc:
            raise RunError(f"attempt ledger corrupt at line {lineno}: {exc}") from exc
    return records


def append_ledger(study_dir: str | Path, record: dict) -> None:
    path = Path(study_dir) / LEDGER_NAME
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def job_attempts(records: list[dict], job_id: str) -> list[dict]:
    """Collapse started/finished journal events to the latest state per attempt."""
    latest = {}
    for record in records:
        if record.get("job_id") == job_id:
            attempt = record.get("attempt_no")
            if type(attempt) is not int or attempt <= 0:
                raise RunError(f"invalid attempt number for {job_id}: {attempt!r}")
            latest[attempt] = record
    return [latest[number] for number in sorted(latest)]


def create_plan(spec_path: str, output_dir: str, simulator_root: str | None = None) -> dict:
    """Validate the spec and seal a plan directory. Refuses non-empty dirs."""
    spec = normalize_spec(load_json_nodupes(spec_path))
    sim_root = resolve_simulator_root(spec, simulator_root)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise SpecError(f"refusing to plan into non-empty directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs").mkdir(exist_ok=True)
    spec["simulator_root"] = sim_root

    configs = expand_configs(spec)
    manifest_configs = []
    jobs = []
    full_combos = 1
    for levels in spec["factors"].values():
        full_combos *= len(levels)
    for cfg in configs:
        resolved = dict(spec["train"])
        resolved.update(cfg["params"])
        epochs = expected_epochs(resolved["total-steps"], resolved["steps-per-epoch"])
        manifest_configs.append(
            {
                "config_id": cfg["config_id"],
                "slug": cfg["slug"],
                "params": cfg["params"],
                "resolved_train": resolved,
                "expected_epochs": epochs,
            }
        )
        for seed in spec["seeds"]:
            jobs.append(
                {
                    "job_id": f"{cfg['config_id']}-s{seed}",
                    "config_id": cfg["config_id"],
                    "seed": seed,
                    "expected_epochs": epochs,
                }
            )
    search_info = dict(spec["search"])
    search_info["full_combos"] = full_combos
    search_info["n_configs"] = len(configs)

    manifest = {
        "protocol": PROTOCOL,
        "study_name": spec["study_name"],
        "spec_hash": spec_hash(spec),
        "created_utc": utc_now(),
        "search": search_info,
        "seeds": spec["seeds"],
        "cost_limit": spec["cost_limit"],
        "factors": spec["factors"],
        "train": spec["train"],
        "simulator_root": sim_root,
        "runner": spec["runner"],
        "repo_root": str(repo_root()),
        "provenance": probe_sources(sim_root),
        "metric": {
            "epret_col": "Metrics/EpRet",
            "epcost_col": "Metrics/EpCost",
            "window": "mean over last 20% of epoch rows (floor, min 1)",
            "epoch_rows": "floor(total_steps/steps_per_epoch); training truncates the remainder",
            "seed_weighting": "equal weight per training seed; sample std over seed scores",
        },
        "selection_rule": SELECTION_RULE,
        "requires_confirmation": list(REQUIRES_CONFIRMATION),
        "verdict_label": "provisional",
        "configs": manifest_configs,
        "jobs": jobs,
    }
    (out / SPEC_NAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blob = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out / MANIFEST_NAME).write_bytes(blob)
    plan_hash = hashlib.sha256(blob).hexdigest()
    (out / MANIFEST_HASH_NAME).write_text(plan_hash + "\n", encoding="utf-8")
    (out / PREREG_NAME).write_text(render_prereg(manifest, plan_hash), encoding="utf-8")
    (out / LEDGER_NAME).write_text("", encoding="utf-8")
    if not os.path.isdir(sim_root):
        print(f"warning: simulator root does not exist (yet): {sim_root}", file=sys.stderr)
    manifest["_plan_hash"] = plan_hash
    manifest["_study_dir"] = str(out)
    return manifest


def load_manifest(study_dir: str | Path) -> dict:
    """Load a plan manifest, rejecting tampered manifests."""
    study = Path(study_dir)
    blob_path, hash_path = study / MANIFEST_NAME, study / MANIFEST_HASH_NAME
    if not blob_path.exists() or not hash_path.exists():
        raise RunError(f"{study} is not a sealed study (missing {MANIFEST_NAME})")
    blob = blob_path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    sealed_parts = hash_path.read_text(encoding="utf-8").strip().split()
    if not sealed_parts:
        raise RunError(f"plan seal {MANIFEST_HASH_NAME} is missing or corrupt; replan")
    sealed = sealed_parts[0]
    if actual != sealed:
        raise RunError(
            f"plan manifest tampered (sha256 {actual[:12]} != sealed {sealed[:12]}); "
            "replan into a fresh directory instead of editing a sealed plan"
        )
    try:
        manifest = json.loads(blob.decode("utf-8"))
    except ValueError as exc:
        raise RunError(f"cannot parse {MANIFEST_NAME}: {exc}") from exc
    if manifest.get("protocol") != PROTOCOL:
        raise RunError(f"unsupported plan protocol {manifest.get('protocol')!r}")
    manifest["_plan_hash"] = sealed
    manifest["_study_dir"] = str(study)
    return manifest


def check_spec_match(manifest: dict, spec_path: str) -> None:
    """Reject a spec file that does not match the sealed plan."""
    spec = normalize_spec(load_json_nodupes(spec_path))
    digest = spec_hash(spec)
    if digest != manifest["spec_hash"]:
        raise SpecError(
            f"spec/plan mismatch: {spec_path} hashes to {digest[:12]} but the sealed plan "
            f"holds {manifest['spec_hash'][:12]}; replan into a fresh directory"
        )


def render_prereg(manifest: dict, plan_hash: str) -> str:
    factor_rows = "\n".join(
        f"| `{name}` | {', '.join(str(v) for v in levels)} |"
        for name, levels in sorted(manifest["factors"].items())
    )
    fixed_rows = "\n".join(
        f"| `{name}` | `{value}` |" for name, value in sorted(manifest["train"].items())
    )
    search = manifest["search"]
    search_line = f"mode={search['mode']}, {search['n_configs']} configs"
    if search["mode"] == "random_subset":
        search_line += (
            f" (deterministic subset of {search['full_combos']}, "
            f"size={search['size']}, seed={search['seed']})"
        )
    prov = manifest["provenance"]
    return f"""# Preregistration — {manifest["study_name"]} (sealed before running)

Sealed: {manifest["created_utc"]}. Do not edit this plan; any spec change
requires a fresh plan directory. Tuning selection is PROVISIONAL until both
confirmations below pass.

- spec sha256: `{manifest["spec_hash"]}`
- plan sha256: `{plan_hash}`
- search: {search_line}
- seeds: {", ".join(str(s) for s in manifest["seeds"])}
- selection cost limit: `{manifest["cost_limit"]}`

## Factors (categorical levels only)

| option | levels |
|---|---|
{factor_rows}

## Fixed training options

| option | value |
|---|---|
{fixed_rows}

## Metric and selection rule (binding)

- Per seed: mean `Metrics/EpRet` / `Metrics/EpCost` over the last 20% of
  `progress.csv` epoch rows (floor, min 1 row), equal weight per seed.
- A config is complete only if every seed log is present, full-length
  (`floor(total_steps/steps_per_epoch)` rows), and finite. Missing,
  non-finite, short, or ambiguous logs invalidate the seed; metrics are
  never filled with zero.
- {SELECTION_RULE}
- This is the TRAINING reward-cost objective, distinct from TIR / Risk Index
  and unseen-patient generalization (separate evaluator).

## Required confirmations (winner stays provisional without both)

1. {REQUIRES_CONFIRMATION[0]}.
2. {REQUIRES_CONFIRMATION[1]}.

## Kill rule

No automatic kill is configured in this tooling. If a manual kill is applied,
record the job IDs, criterion, and timestamp in the study notes; killed jobs
stay `failed` in the ledger and need explicit `--retry-failed` to rerun.

## Provenance

- repo: `{prov["repo"]["path"]}` head `{prov["repo"]["head"]}`;
  source content sha256 `{prov["repo"]["content_sha256"]}`
- simulator: `{prov["simulator"]["path"]}` head `{prov["simulator"]["head"]}`;
  source content sha256 `{prov["simulator"]["content_sha256"]}`
- python: `{prov["python"]["executable"]}` ({prov["python"]["version"]})

Sealed by: ________________  Date: __________
"""
