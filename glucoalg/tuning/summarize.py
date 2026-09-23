"""Aggregate progress.csv metrics and pick a PROVISIONAL winner.

Stdlib-only except for the optional ``--plot`` path, which imports matplotlib
lazily and fails with a clear message when it is unavailable.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

from . import plan as planmod
from .plan import MetricError, RunError

EPRET_COL = "Metrics/EpRet"
EPCOST_COL = "Metrics/EpCost"

OBJECTIVE_NOTE = (
    "Selection uses the TRAINING reward-cost objective (OmniSafe train rewards "
    "and costs, last-20% epochs, equal training-seed weighting) under the "
    "configured cost limit. This is not TIR / Risk Index and not unseen-patient "
    "generalization: those need the separate evaluator plus held-out confirmation."
)


def find_progress_csv(attempt_dir: str | Path) -> str:
    matches = sorted(glob.glob(str(Path(attempt_dir) / "**" / "progress.csv"), recursive=True))
    if not matches:
        raise MetricError(f"no progress.csv under {attempt_dir}")
    if len(matches) > 1:
        raise MetricError(f"{len(matches)} progress.csv files under {attempt_dir} (expected exactly 1)")
    return matches[0]


def verify_progress(attempt_dir: str | Path, expected_epochs: int, expected_sha256: str | None = None) -> dict:
    """Validate one attempt log; return seed scores. Raises MetricError."""
    csv_path = find_progress_csv(attempt_dir)
    if type(expected_epochs) is not int or expected_epochs < 1:
        raise MetricError("Expected at least one completed epoch")
    try:
        blob = Path(csv_path).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise MetricError(f"{csv_path}: progress.csv hash changed since completion")
        with io.StringIO(blob.decode("utf-8"), newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise MetricError(f"{csv_path}: no header row")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise MetricError(f"{csv_path}: duplicate header columns")
            for col in (EPRET_COL, EPCOST_COL):
                if col not in reader.fieldnames:
                    raise MetricError(f"{csv_path}: missing column {col}")
            rets, costs = [], []
            for index, row in enumerate(reader):
                if None in row:
                    raise MetricError(f"{csv_path}: malformed CSV row")
                if "Train/Epoch" in row:
                    try:
                        epoch = float(row["Train/Epoch"])
                    except (TypeError, ValueError) as exc:
                        raise MetricError(f"{csv_path}: invalid Train/Epoch") from exc
                    if epoch != index:
                        raise MetricError(f"{csv_path}: Train/Epoch must be contiguous from zero")
                rets.append(row.get(EPRET_COL))
                costs.append(row.get(EPCOST_COL))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetricError(f"cannot read {csv_path}: {exc}") from exc
    n_rows = len(rets)
    if n_rows < expected_epochs:
        raise MetricError(f"{csv_path}: short log ({n_rows} epoch rows, expected {expected_epochs})")
    if n_rows > expected_epochs:
        raise MetricError(f"{csv_path}: excess epoch rows ({n_rows}, expected exactly {expected_epochs})")

    def window_mean(values: list, col: str, width: int) -> float:
        numbers = []
        for raw in values:
            try:
                number = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise MetricError(f"{csv_path}: {col} has a missing/non-numeric value")
            if not math.isfinite(number):
                raise MetricError(f"{csv_path}: {col} has a non-finite value")
            if col == EPCOST_COL and number < 0:
                raise MetricError(f"{csv_path}: safety cost must be nonnegative")
            numbers.append(number)
        try:
            result = statistics.mean(numbers[-width:])
        except OverflowError as exc:
            raise MetricError(f"{csv_path}: metric mean overflow") from exc
        if not math.isfinite(result):
            raise MetricError(f"{csv_path}: metric mean must be finite")
        return result

    width = max(1, n_rows // 5)  # last 20% of rows (floor, min 1)
    return {
        "progress_csv": csv_path,
        "progress_sha256": digest,
        "n_rows": n_rows,
        "window": width,
        "epret": window_mean(rets, EPRET_COL, width),
        "epcost": window_mean(costs, EPCOST_COL, width),
    }


def _score_job(manifest: dict, study: Path, records: list[dict], job: dict) -> dict:
    attempts = planmod.job_attempts(records, job["job_id"])
    base = {"job_id": job["job_id"], "seed": job["seed"], "expected_epochs": job["expected_epochs"]}
    if not attempts:
        return {**base, "status": "never-run", "reason": "no attempts recorded"}
    latest = attempts[-1]
    if latest.get("status") != "success" or latest.get("returncode") != 0:
        return {**base, "status": "failed",
                "reason": latest.get("fail_reason") or f"exit={latest.get('returncode')}"}
    if latest.get("runner_used", "").startswith("test:"):
        return {**base, "status": "synthetic", "reason": "Synthetic test-runner output is ineligible for research selection"}
    if latest.get("runner_used") not in {"glucoalg.train", "run.py"}:
        return {**base, "status": "invalid", "reason": "Missing or unrecognized training runner provenance"}
    from .execute import verify_recorded_success

    ok, reason = verify_recorded_success(manifest, study, latest)
    if not ok:
        return {**base, "status": "invalid", "reason": reason}
    try:
        metrics = verify_progress(
            str(study / latest["log_dir"]), job["expected_epochs"], latest["progress_sha256"])
    except MetricError as exc:
        return {**base, "status": "invalid", "reason": str(exc)}
    return {**base, "status": "ok", "reason": None,
            "n_rows": metrics["n_rows"], "window": metrics["window"],
            "epret": metrics["epret"], "epcost": metrics["epcost"],
            "progress_csv": str(Path(metrics["progress_csv"]).relative_to(study))}


def summarize_study(
    study_dir: str | Path,
    out_dir: str | Path | None = None,
    make_plot: bool = False,
    spec_path: str | None = None,
) -> dict:
    """Aggregate the study, write summary.json/csv (+plot), print a report."""
    manifest = planmod.load_manifest(study_dir)
    if spec_path:
        planmod.check_spec_match(manifest, spec_path)
    study = Path(study_dir)
    records = planmod.read_ledger(study)
    from .execute import check_ledger_plan

    check_ledger_plan(records, manifest)
    declared_seeds = manifest["seeds"]
    if not declared_seeds or len(declared_seeds) != len(set(declared_seeds)):
        raise RunError("Plan must declare unique training seeds")
    for cfg in manifest["configs"]:
        job_seeds = [job["seed"] for job in manifest["jobs"] if job["config_id"] == cfg["config_id"]]
        if len(job_seeds) != len(declared_seeds) or set(job_seeds) != set(declared_seeds):
            raise RunError(f"{cfg['config_id']}: planned jobs do not match declared seed coverage")
    warnings: list[str] = []

    config_results = []
    for cfg in manifest["configs"]:
        seeds = [_score_job(manifest, study, records, job)
                 for job in manifest["jobs"] if job["config_id"] == cfg["config_id"]]
        ok = [s for s in seeds if s["status"] == "ok"]
        complete = bool(seeds) and len(ok) == len(seeds)
        for seed in seeds:
            if seed["status"] == "invalid":
                warnings.append(f"{seed['job_id']}: prior success no longer verifies ({seed['reason']})")
        entry: dict = {
            "config_id": cfg["config_id"], "params": cfg["params"], "slug": cfg["slug"],
            "expected_epochs": cfg["expected_epochs"], "n_seeds": len(seeds),
            "seeds_ok": len(ok), "complete": complete, "seeds": seeds,
            "ret_mean": None, "ret_std": None, "cost_mean": None, "cost_std": None,
            "feasible": False, "rank": None,
        }
        if complete:
            rets = [s["epret"] for s in ok]
            costs = [s["epcost"] for s in ok]
            try:
                stats = {"ret_mean": statistics.mean(rets), "cost_mean": statistics.mean(costs),
                         "ret_std": statistics.stdev(rets) if len(rets) >= 2 else 0.0,
                         "cost_std": statistics.stdev(costs) if len(costs) >= 2 else 0.0}
                if not all(math.isfinite(value) for value in stats.values()):
                    raise OverflowError("non-finite aggregate statistic")
            except (OverflowError, ValueError) as exc:
                entry["complete"] = False
                warnings.append(f"{cfg['config_id']}: cannot aggregate finite statistics: {exc}")
            else:
                entry.update(stats)
                entry["feasible"] = bool(entry["cost_mean"] <= manifest["cost_limit"])
        config_results.append(entry)

    complete_cfgs = [c for c in config_results if c["complete"]]
    ranked = sorted(complete_cfgs, key=lambda c: (-c["ret_mean"], c["cost_mean"], c["config_id"]))
    for pos, cfg in enumerate(ranked, 1):
        cfg["rank"] = pos
    feasible = [c for c in ranked if c["feasible"]]
    if feasible:
        winner = feasible[0]
        selected = {"config_id": winner["config_id"], "params": winner["params"],
                    "seeds": manifest["seeds"], "ret_mean": winner["ret_mean"],
                    "cost_mean": winner["cost_mean"]}
        reason = None
    else:
        selected = None
        reason = ("no winner: no complete config satisfies mean cost <= "
                  f"{manifest['cost_limit']}" if complete_cfgs
                  else "no winner: no config is complete (every config lacks a valid seed)")

    summary = {
        "study_name": manifest["study_name"],
        "verdict": "provisional",
        "requires": list(manifest["requires_confirmation"]),
        "objective_note": OBJECTIVE_NOTE,
        "selection_rule": manifest["selection_rule"],
        "cost_limit": manifest["cost_limit"],
        "spec_hash": manifest["spec_hash"],
        "plan_hash": manifest["_plan_hash"],
        "n_configs": len(config_results),
        "n_complete": len(complete_cfgs),
        "n_feasible": len(feasible),
        "selected": selected,
        "no_winner_reason": reason,
        "configs": config_results,
        "warnings": warnings,
        "provenance": manifest["provenance"],
    }
    out = Path(out_dir) if out_dir else study / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                      encoding="utf-8")
    _write_csv(summary, manifest, out / "summary.csv")
    if make_plot:
        plot_tradeoff(summary, out / "tradeoff.png")
    _print_report(summary, out)
    return summary


def _write_csv(summary: dict, manifest: dict, path: Path) -> None:
    factor_names = sorted(manifest["factors"])
    header = (["config_id"] + factor_names
              + ["n_seeds", "seeds_ok", "complete", "feasible", "rank",
                 "R_mean", "R_std", "C_mean", "C_std"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cfg in summary["configs"]:
            writer.writerow(
                [cfg["config_id"]] + [cfg["params"][name] for name in factor_names]
                + [cfg["n_seeds"], cfg["seeds_ok"], cfg["complete"], cfg["feasible"],
                   cfg["rank"] if cfg["rank"] is not None else "",
                   _num(cfg["ret_mean"]), _num(cfg["ret_std"]),
                   _num(cfg["cost_mean"]), _num(cfg["cost_std"])]
            )


def _num(value) -> str:
    return "" if value is None else repr(value)


def _print_report(summary: dict, out: Path) -> None:
    print(f"study {summary['study_name']}: verdict={summary['verdict']} "
          f"({summary['n_complete']}/{summary['n_configs']} complete, "
          f"{summary['n_feasible']} feasible)")
    for cfg in sorted(summary["configs"], key=lambda c: (c["rank"] is None, c["rank"])):
        if cfg["complete"]:
            print(f"  rank {cfg['rank']}: {cfg['config_id']} [{cfg['slug']}] "
                  f"R={cfg['ret_mean']:.3f}±{cfg['ret_std']:.3f} "
                  f"C={cfg['cost_mean']:.3f}±{cfg['cost_std']:.3f} "
                  f"{'FEASIBLE' if cfg['feasible'] else 'over-limit'}")
        else:
            print(f"  incomplete: {cfg['config_id']} [{cfg['slug']}] "
                  f"({cfg['seeds_ok']}/{cfg['n_seeds']} seeds valid)")
    if summary["selected"]:
        sel = summary["selected"]
        print(f"selected (PROVISIONAL): {sel['config_id']} {sel['params']} "
              f"over seeds {sel['seeds']}; still requires: {'; '.join(summary['requires'])}")
    else:
        print(f"{summary['no_winner_reason']}; still requires: {'; '.join(summary['requires'])}")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    print(f"wrote {out / 'summary.json'} and {out / 'summary.csv'}")


def plot_tradeoff(summary: dict, out_png: str | Path) -> str:
    """Render the cost/return tradeoff figure (complete configs only)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RunError("matplotlib is required for --plot but is not installed") from exc
    complete = [c for c in summary["configs"] if c["complete"]]
    if not complete:
        print("warning: no complete configs; skipping tradeoff plot")
        return ""
    selected_id = (summary["selected"] or {}).get("config_id")
    fig, axis = plt.subplots(figsize=(10, 7))
    for cfg in complete:
        marker = "*" if cfg["config_id"] == selected_id else "o"
        size = 14 if cfg["config_id"] == selected_id else 8
        axis.errorbar(cfg["cost_mean"], cfg["ret_mean"],
                      xerr=cfg["cost_std"], yerr=cfg["ret_std"],
                      fmt=marker, markersize=size, capsize=4,
                      label=f"{cfg['config_id']} [{cfg['slug']}]")
    axis.axvline(x=summary["cost_limit"], color="r", linestyle="--", label="cost limit")
    axis.axvspan(0, summary["cost_limit"], color="green", alpha=0.08)
    axis.set_title("Safe RL tradeoff (PROVISIONAL train objective; mean±std over seeds)")
    axis.set_xlabel("Mean cost (lower is safer)")
    axis.set_ylabel("Mean return (higher is better)")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize="small")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png} ({len(complete)} complete configs; incomplete configs never plotted)")
    return str(out_png)
