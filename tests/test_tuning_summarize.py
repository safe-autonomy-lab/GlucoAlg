"""Aggregation, selection, and rejection rules (fake-trainer studies)."""
import csv
import json
import statistics

import pytest

from glucoalg.tuning import execute as execmod
from glucoalg.tuning.plan import MetricError, create_plan
from glucoalg.tuning.summarize import summarize_study, verify_progress

FAKE_TRAINER = """
import argparse, csv, os
ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", required=True)
ap.add_argument("--total-steps", type=int, required=True)
ap.add_argument("--steps-per-epoch", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--actor-lr", type=float, default=1e-5)
ap.add_argument("--critic-lr", type=float, default=5e-5)
args, _ = ap.parse_known_args()
n = args.total_steps // args.steps_per_epoch
d = os.path.join(args.log_dir, "runs", "fake", "seed-%d" % args.seed)
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "progress.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Metrics/EpRet", "Metrics/EpCost"])
    for _ in range(n):
        w.writerow([1000.0 * args.actor_lr + args.seed,
                    100000.0 * args.critic_lr + args.seed * 0.001])
"""


def make_study(tmp_path, factors, cost_limit=100.0, seeds=(100, 101), synthetic=False):
    sim = tmp_path / "sim"
    sim.mkdir(parents=True, exist_ok=True)
    spec = {
        "study_name": "demo",
        "seeds": list(seeds),
        "cost_limit": cost_limit,
        "factors": factors,
        "train": {
            "algo": "PPOLag", "env-id": "t1d-v0", "cohort": "adolescent",
            "total-steps": 640, "steps-per-epoch": 64,
            "vector-env-nums": 1, "device": "cpu",
        },
        "simulator_root": str(sim),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(FAKE_TRAINER)
    out = tmp_path / "study"
    manifest = create_plan(str(spec_path), str(out))
    execmod.run_study(out, all_jobs=True, runner_cmd=str(trainer))
    if not synthetic:
        # Model genuine trainer receipts for arithmetic tests. The explicit
        # synthetic-runner exclusion is exercised separately below.
        path = out / "attempts.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            record["runner_used"] = "glucoalg.train"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return out, manifest


def test_complete_selection_with_seed_stats(tmp_path):
    out, _ = make_study(tmp_path, {"actor-lr": [0.0003, 3e-05]})
    summary = summarize_study(out)
    assert summary["verdict"] == "provisional"
    assert summary["requires"] and summary["n_complete"] == 2
    hi = next(c for c in summary["configs"] if c["params"] == {"actor-lr": 0.0003})
    exp_ret = [1000 * 0.0003 + s for s in (100, 101)]
    exp_cost = [100000 * 5e-5 + s * 0.001 for s in (100, 101)]
    assert hi["ret_mean"] == pytest.approx(statistics.fmean(exp_ret))
    assert hi["ret_std"] == pytest.approx(statistics.stdev(exp_ret))
    assert hi["cost_mean"] == pytest.approx(statistics.fmean(exp_cost))
    assert hi["cost_std"] == pytest.approx(statistics.stdev(exp_cost))
    assert hi["feasible"] and hi["rank"] == 1
    assert summary["selected"]["config_id"] == hi["config_id"]
    assert summary["selected"]["params"] == {"actor-lr": 0.0003}
    assert summary["selected"]["seeds"] == [100, 101]
    assert summary["no_winner_reason"] is None
    assert "TRAINING" in summary["objective_note"]
    assert (out / "summary" / "summary.json").exists()
    rows = list(csv.DictReader(open(out / "summary" / "summary.csv")))
    assert len(rows) == 2
    assert rows[0]["rank"] and rows[0]["R_mean"] and rows[0]["C_mean"]


def test_tiebreak_lower_cost_then_config_id(tmp_path):
    out, _ = make_study(tmp_path, {"critic-lr": [0.0001, 0.0005]})
    summary = summarize_study(out)
    # identical returns (trainer ignores critic for EpRet); cheaper cost wins
    assert summary["selected"]["params"] == {"critic-lr": 0.0001}
    # full tie (trainer ignores batch-size): lowest config_id wins
    out2, _ = make_study(tmp_path / "tied", {"batch-size": [32, 64]})
    assert summarize_study(out2)["selected"]["config_id"] == "cfg00"


def test_no_feasible_config_means_no_winner(tmp_path):
    out, _ = make_study(tmp_path, {"actor-lr": [0.0003, 3e-05]}, cost_limit=0.5)
    summary = summarize_study(out)
    assert summary["selected"] is None
    assert summary["n_feasible"] == 0
    assert "no winner" in summary["no_winner_reason"]
    assert summary["verdict"] == "provisional"
    # complete configs are still ranked
    assert sorted(c["rank"] for c in summary["configs"]) == [1, 2]


def test_incomplete_config_never_wins_and_warns(tmp_path):
    out, _ = make_study(tmp_path, {"actor-lr": [0.0003, 3e-05]})
    victim = next((out / "jobs" / "cfg00-s100" / "attempt1").glob("**/progress.csv"))
    victim.unlink()
    summary = summarize_study(out)
    broken = next(c for c in summary["configs"] if c["config_id"] == "cfg00")
    assert not broken["complete"] and broken["seeds_ok"] == 1
    assert broken["ret_mean"] is None and broken["rank"] is None
    assert summary["warnings"]
    # the other complete config is selected instead
    assert summary["selected"]["config_id"] == "cfg01"
    rows = {r["config_id"]: r for r in csv.DictReader(open(out / "summary" / "summary.csv"))}
    assert rows["cfg00"]["R_mean"] == "" and rows["cfg00"]["C_mean"] == ""
    assert rows["cfg00"]["complete"] == "False"


@pytest.mark.parametrize("corrupt", ["nonfinite", "short", "missing-col", "multi-csv"])
def test_corrupt_logs_rejected_not_zero_filled(tmp_path, corrupt):
    out, _ = make_study(tmp_path, {"actor-lr": [0.0003]})
    csv_path = next((out / "jobs" / "cfg00-s100" / "attempt1").glob("**/progress.csv"))
    if corrupt == "nonfinite":
        lines = csv_path.read_text().splitlines()
        lines[-1] = "nan,1.0"
        csv_path.write_text("\n".join(lines) + "\n")
    elif corrupt == "short":
        lines = csv_path.read_text().splitlines()
        csv_path.write_text("\n".join(lines[:6]) + "\n")
    elif corrupt == "missing-col":
        lines = csv_path.read_text().splitlines()
        lines[0] = "Metrics/EpRet"
        csv_path.write_text("\n".join(lines) + "\n")
    elif corrupt == "multi-csv":
        extra = csv_path.parent / "nested"
        extra.mkdir()
        (extra / "progress.csv").write_text(csv_path.read_text())
    summary = summarize_study(out)
    cfg = summary["configs"][0]
    assert not cfg["complete"]
    assert cfg["ret_mean"] is None
    assert summary["selected"] is None
    assert "no config is complete" in summary["no_winner_reason"]


def test_verify_progress_window_math(tmp_path):
    csv_path = tmp_path / "progress.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Metrics/EpRet", "Metrics/EpCost"])
        for i in range(10):
            w.writerow([float(i), 100.0])
    got = verify_progress(tmp_path, 10)
    assert got["window"] == 2  # floor(10/5)
    assert got["epret"] == pytest.approx(8.5)
    with pytest.raises(MetricError, match="short"):
        verify_progress(tmp_path, 11)
    tiny = tmp_path / "tiny"
    tiny.mkdir()
    with open(tiny / "progress.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Metrics/EpRet", "Metrics/EpCost"])
        w.writerow([1.0, 2.0])
    assert verify_progress(tiny, 1)["window"] == 1  # min 1 row


def test_plot_needs_no_zero_fill(tmp_path):
    pytest.importorskip("matplotlib")
    out, _ = make_study(tmp_path, {"actor-lr": [0.0003, 3e-05]})
    summary = summarize_study(out, make_plot=True)
    png = out / "summary" / "tradeoff.png"
    assert png.exists() and png.stat().st_size > 0
    assert summary["n_complete"] == 2


def test_synthetic_runner_cannot_win(tmp_path):
    out, _ = make_study(tmp_path, {"actor-lr": [3e-4]}, synthetic=True)
    summary = summarize_study(out)
    assert summary["selected"] is None
    assert summary["configs"][0]["seeds"][0]["status"] == "synthetic"


def test_finite_modified_results_cannot_win(tmp_path):
    out, _ = make_study(tmp_path, {"actor-lr": [3e-4]})
    path = next((out / "jobs" / "cfg00-s100").rglob("progress.csv"))
    lines = path.read_text().splitlines()
    lines[-1] = "999,1"
    path.write_text("\n".join(lines) + "\n")
    summary = summarize_study(out)
    assert summary["selected"] is None
    assert "hash" in summary["configs"][0]["seeds"][0]["reason"]


def test_mixed_plan_ledger_rejected_by_summary(tmp_path):
    from glucoalg.tuning.plan import RunError

    out, _ = make_study(tmp_path, {"actor-lr": [3e-4]})
    path = out / "attempts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["plan_hash"] = "different-plan"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RunError, match="plan"):
        summarize_study(out)


def test_extra_epochs_and_nonfinite_outside_tail_rejected(tmp_path):
    path = tmp_path / "progress.csv"
    path.write_text("Metrics/EpRet,Metrics/EpCost\n" + "1,2\n" * 10)
    with pytest.raises(MetricError, match="excess"):
        verify_progress(tmp_path, 9)
    path.write_text("Metrics/EpRet,Metrics/EpCost\nnan,2\n" + "1,2\n" * 9)
    with pytest.raises(MetricError, match="non-finite"):
        verify_progress(tmp_path, 10)


def test_epoch_sequence_and_cost_domain(tmp_path):
    path = tmp_path / "progress.csv"
    path.write_text("Metrics/EpRet,Metrics/EpCost,Train/Epoch\n1,2,0\n3,4,0\n")
    with pytest.raises(MetricError, match="contiguous"):
        verify_progress(tmp_path, 2)
    path.write_text("Metrics/EpRet,Metrics/EpCost\n1,-2\n")
    with pytest.raises(MetricError, match="nonnegative"):
        verify_progress(tmp_path, 1)


def test_large_finite_window_scores_do_not_overflow(tmp_path):
    path = tmp_path / "progress.csv"
    path.write_text("Metrics/EpRet,Metrics/EpCost\n" + "1e308,2\n" * 10)
    assert verify_progress(tmp_path, 10)["epret"] == 1e308
