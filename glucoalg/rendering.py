"""Render GlucoAlg's episode_*_controller.csv without rerunning a policy.

python -m glucoalg.rendering --trace evaluation/adolescent#002/episode_0_controller.csv \
    --output episode.gif
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image


def read_trace(path: Path, glucose_source=None):
    for directory in [path.parent, path.parent.parent]:
        manifest = directory / "EVAL_MANIFEST.json"
        if manifest.is_file():
            recorded_source = json.loads(manifest.read_text()).get("glucose_source")
            if glucose_source is not None and recorded_source != glucose_source:
                raise ValueError("Requested glucose alignment conflicts with the evaluation manifest")
            glucose_source = recorded_source
            break
    if glucose_source != "post-step":
        raise ValueError("This renderer requires post-step CGM. Evaluate with --glucose-source post-step; "
                         "for a CSV without its manifest, explicitly pass --glucose-source post-step.")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"time_minutes", "CGM", "bolus_index", "meal_index",
                "bolus_accepted", "meal_accepted"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected a nonempty GlucoAlg controller CSV with {sorted(required)}")
    time = np.array([float(row["time_minutes"]) for row in rows])
    glucose = np.array([float(row["CGM"]) for row in rows])
    if not np.isfinite(time).all() or not np.isfinite(glucose).all():
        raise ValueError("Non-finite time/glucose in trace")
    # The supported evaluator uses five-minute controller steps. CSV timestamps
    # mark the observation AFTER the action; requests occur five minutes earlier.
    if not np.allclose(time, 5 * np.arange(1, len(rows) + 1), rtol=0, atol=1e-6):
        raise ValueError("Expected consecutive five-minute post-step timestamps")
    if np.any(glucose <= 0):
        raise ValueError("Glucose must be positive")
    actions = np.array([[float(row["bolus_index"]), float(row["meal_index"])] for row in rows])
    if not np.isfinite(actions).all() or np.any(actions != np.floor(actions)) or np.any((actions < 0) | (actions > 4)):
        raise ValueError("Expected integer bolus/meal recommendation levels 0–4")

    def boolean(value):
        if value.lower() not in {"true", "false", "1", "0"}:
            raise ValueError(f"Invalid acceptance flag: {value}")
        return value.lower() in {"true", "1"}

    accepted = np.array([[boolean(row["bolus_accepted"]), boolean(row["meal_accepted"])] for row in rows])
    return time, glucose, actions.astype(int), accepted


def render(trace: Path, output: Path, title="Trained policy in GlucoSim", seconds=20.0, glucose_source=None):
    if not np.isfinite(seconds) or seconds < 1:
        raise ValueError("Animation duration must be at least one second")
    poster = output.with_suffix(".png")
    receipt = output.with_suffix(".render.json")
    if output.suffix.lower() != ".gif":
        raise ValueError("Output must end in .gif")
    for path in [output, poster, receipt]:
        if path.exists():
            raise FileExistsError(f"Choose a fresh output name; refusing to overwrite {path}")
    minutes, glucose, actions, accepted = read_trace(trace, glucose_source)
    hours = minutes / 60
    requests_at = (minutes - 5) / 60
    output.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive claim also protects concurrent standalone renders. Keep it
    # alongside the receipt so a failed/partial output is never silently reused.
    with output.with_suffix(".gif.lock").open("x") as claim:
        claim.write(str(trace.resolve()) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=80, facecolor="white")
    fig.text(.07, .93, title, fontsize=23, weight="bold", color="#17212b")
    fig.text(.07, .875, "LATEST REQUEST", fontsize=10, color="#59636e")
    request = fig.text(.07, .805, "No nonzero request yet", fontsize=14, color="#2864a5")
    fig.text(.66, .875, "SIMULATED TIME", fontsize=10, color="#59636e")
    clock = fig.text(.66, .815, "", fontsize=20, color="#087f7d")
    ax = fig.add_axes([.14, .37, .80, .35])
    ax.set_xlim(0, max(1, hours[-1]))
    low = min(50, np.floor((glucose.min() - 10) / 10) * 10)
    high = max(205, np.ceil((glucose.max() + 10) / 10) * 10)
    ax.set_ylim(low, high)
    ax.axhspan(70, 180, color="#edf5ef")
    for threshold in [70, 180]:
        ax.axhline(threshold, color="#aec8b4", linestyle="--", linewidth=1)
    ax.set_ylabel("CGM (mg/dL)")
    ax.tick_params(labelbottom=False)
    line, = ax.plot([], [], color="#2864a5", linewidth=2.5)
    point, = ax.plot([], [], "o", color="#2864a5")
    cursor = ax.axvline(0, color="#087f7d", linewidth=1)
    lane = fig.add_axes([.14, .17, .80, .15], sharex=ax)
    lane.set_ylim(-.5, 1.5)
    lane.set_yticks([1, 0], ["Bolus request", "Meal request"])
    lane.set_xlabel("Simulated hours")
    for y in [0, 1]:
        lane.axhline(y, color="#e8edf1", zorder=0)
    approved = lane.scatter([], [], color="#087f7d", s=35)
    declined = lane.scatter([], [], color="#bc433c", marker="x", s=40)
    fig.text(.14, .075, "● Accepted request", color="#087f7d", fontsize=11)
    fig.text(.38, .075, "× Declined request", color="#bc433c", fontsize=11)
    fig.text(.07, .025, "Recorded simulator CGM • recommendation levels 0–4 • acceptance is not a delivered physical dose",
             fontsize=9, color="#59636e")

    def update(index):
        line.set_data(hours[:index + 1], glucose[:index + 1])
        point.set_data([hours[index]], [glucose[index]])
        cursor.set_xdata([hours[index], hours[index]])
        clock.set_text(f"Day {int(minutes[index] // 1440) + 1}, {int(minutes[index] % 1440) // 60:02d}:{int(minutes[index] % 60):02d}")
        yes, no = [], []
        for step in range(index + 1):
            for head, y in [(0, 1), (1, 0)]:
                if actions[step, head] > 0:
                    (yes if accepted[step, head] else no).append((requests_at[step], y))
        approved.set_offsets(np.asarray(yes).reshape(-1, 2))
        declined.set_offsets(np.asarray(no).reshape(-1, 2))
        requested = np.flatnonzero(np.any(actions[:index + 1] > 0, axis=1))
        if requested.size:
            step = requested[-1]
            labels = [f"{name}: level {actions[step, head]} ({'accepted' if accepted[step, head] else 'declined'})"
                      for head, name in enumerate(["Bolus", "Meal"]) if actions[step, head] > 0]
            request.set_text("\n".join(labels) + f"\nat {requests_at[step]:.2f} h")
        else:
            request.set_text("No nonzero request yet")

    # Every recorded glucose sample remains in the cumulative line. Frame
    # subsampling changes playback speed only; persistent lanes retain all events.
    frame_count = min(len(minutes), 160, int(seconds * 100))
    indices = np.unique(np.linspace(0, len(minutes) - 1, frame_count, dtype=int))
    try:
        animation = FuncAnimation(fig, update, frames=indices, repeat=True, cache_frame_data=False)
        animation.save(output, writer=PillowWriter(fps=min(100.0, len(indices) / seconds)))
        update(len(minutes) - 1)
        fig.savefig(poster, dpi=100)
    finally:
        plt.close(fig)
    with Image.open(output) as gif:
        duration = 0
        for frame in range(gif.n_frames):
            gif.seek(frame)
            duration += gif.info.get("duration", 0)
        media = {"gif_frames": gif.n_frames, "actual_duration_seconds": duration / 1000}
    receipt.write_text(json.dumps({"trace": str(trace.resolve()),
                                  "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                                  "sample_count": len(minutes), "observed_minutes": float(minutes[-1]),
                                  "glucose_axis": [float(low), float(high)],
                                  "final_sample_included": True, **media}, indent=2))
    print(f"GIF: {output.resolve()}\nPoster: {poster.resolve()}\nReceipt: {receipt.resolve()}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Trained policy in GlucoSim")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--glucose-source", choices=["post-step"], help="Required only when the CSV's evaluation manifest is unavailable")
    args = parser.parse_args(argv)
    render(args.trace, args.output, args.title, args.seconds, args.glucose_source)


if __name__ == "__main__":
    main()
