"""Checkpoint demo and real CSV/GIF regressions; no training or network calls."""
import csv
import hashlib
import importlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image


def test_packaged_animation_commands_exist():
    assert importlib.util.find_spec("glucoalg.demo") is not None
    assert importlib.util.find_spec("glucoalg.rendering") is not None


def trace_file(tmp_path, rows=None, source="post-step"):
    rows = rows or [
        [5, 55, 1, 0, "True", "False"],
        [10, 345, 0, 2, "False", "True"],
        [15, 110, 0, 0, "False", "False"],
    ]
    path = tmp_path / "episode_0_controller.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_minutes", "CGM", "bolus_index", "meal_index",
                         "bolus_accepted", "meal_accepted"])
        writer.writerows(rows)
    if source is not None:
        (tmp_path / "EVAL_MANIFEST.json").write_text(json.dumps({"glucose_source": source}))
    return path


@pytest.mark.parametrize("source", [None, "legacy"])
def test_renderer_rejects_unconfirmed_or_legacy_alignment(tmp_path, source):
    rendering = importlib.import_module("glucoalg.rendering")
    with pytest.raises(ValueError, match="post-step"):
        rendering.read_trace(trace_file(tmp_path, source=source))


def test_explicit_alignment_cannot_override_recorded_legacy(tmp_path):
    rendering = importlib.import_module("glucoalg.rendering")
    with pytest.raises(ValueError, match="conflicts"):
        rendering.read_trace(trace_file(tmp_path, source="legacy"), "post-step")


@pytest.mark.parametrize("column,value", [(0, 9), (1, "nan"), (2, 2.5), (3, 5), (4, "maybe")])
def test_invalid_controller_data_is_rejected(tmp_path, column, value):
    rendering = importlib.import_module("glucoalg.rendering")
    rows = [[5, 100, 0, 0, "False", "False"], [10, 110, 1, 2, "True", "False"]]
    rows[1][column] = value
    with pytest.raises(ValueError):
        rendering.read_trace(trace_file(tmp_path, rows))


def test_real_gif_contains_full_trace_without_clipping(tmp_path):
    rendering = importlib.import_module("glucoalg.rendering")
    trace = trace_file(tmp_path)
    original_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    output = tmp_path / "episode.gif"
    rendering.render(trace, output, seconds=1)
    receipt = json.loads(output.with_suffix(".render.json").read_text())
    with Image.open(output) as gif:
        assert gif.n_frames >= 2
        assert gif.size[0] > 0
    assert receipt["sample_count"] == 3
    assert receipt["observed_minutes"] == 15
    assert receipt["glucose_axis"][0] <= 55
    assert receipt["glucose_axis"][1] >= 345
    assert receipt["trace_sha256"] == original_hash
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == original_hash
    assert output.with_suffix(".png").is_file()


@pytest.mark.parametrize("suffix", [".gif", ".png", ".render.json", ".gif.lock"])
def test_existing_or_claimed_outputs_are_preserved(tmp_path, suffix):
    rendering = importlib.import_module("glucoalg.rendering")
    trace = trace_file(tmp_path)
    output = tmp_path / "episode.gif"
    occupied = output.with_suffix(suffix)
    occupied.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        rendering.render(trace, output)
    assert occupied.read_bytes() == b"previous result"
    if suffix != ".gif":
        assert not output.exists()


def test_short_gif_has_nonzero_encoded_delays(tmp_path, monkeypatch):
    rendering = importlib.import_module("glucoalg.rendering")
    trace = trace_file(tmp_path, [[5 * (i + 1), 100 + i % 30, 0, 0, "False", "False"] for i in range(160)])
    # Shrink only the canvas, while exercising the real animation and GIF writer.
    figure = rendering.plt.figure

    def small_canvas(*args, **kwargs):
        kwargs.update(figsize=(2, 1), dpi=30)
        return figure(*args, **kwargs)

    monkeypatch.setattr(rendering.plt, "figure", small_canvas)
    output = tmp_path / "short.gif"
    rendering.render(trace, output, seconds=1.38)
    with Image.open(output) as gif:
        for frame in range(gif.n_frames):
            gif.seek(frame)
            assert gif.info["duration"] >= 10


@pytest.mark.parametrize("flags", [["--days", "0.5"], ["--days", "nan"],
                                  ["--seconds", "0"], ["--seconds", "inf"],
                                  ["--eval-seed", "-1"]])
def test_invalid_demo_options_fail_before_download_or_output(tmp_path, flags):
    demo = importlib.import_module("glucoalg.demo")
    output = tmp_path / "should-not-exist"
    with pytest.raises(SystemExit):
        demo.main(["--hf-repo", "not-contacted", "--simulator-root", str(tmp_path),
                   "--output-dir", str(output), *flags])
    assert not output.exists()


def test_local_policy_requires_matching_config_argument(tmp_path):
    demo = importlib.import_module("glucoalg.demo")
    args = demo.build_parser().parse_args([
        "--checkpoint", str(tmp_path / "epoch-1.pt"),
        "--simulator-root", str(tmp_path), "--output-dir", str(tmp_path / "output"),
    ])
    with pytest.raises(ValueError, match="config"):
        demo.resolve_policy(args)


@pytest.mark.parametrize("tampered", [False, True])
def test_hf_selection_pins_revision_and_verifies_bytes(tmp_path, monkeypatch, tampered):
    demo = importlib.import_module("glucoalg.demo")
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"saved actor state")
    config = tmp_path / "config.json"
    config.write_text('{"seed":100}')
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"simulator_commit": "sim-commit", "policies": [{
        "policy_seed": 100, "checkpoint": "checkpoints/seed100/policy.pt", "config": "config/seed100/config.json",
        "checkpoint_sha256": hashlib.sha256(b"saved actor state").hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }]}))
    if tampered:
        checkpoint.write_bytes(b"different actor state")
    files = {"releases/rerun-20261008/release.json": release,
             "checkpoints/seed100/policy.pt": checkpoint, "config/seed100/config.json": config}
    revisions = []

    class Hub:
        def __init__(self, token):
            pass

        def model_info(self, repo_id, revision):
            return SimpleNamespace(sha="immutable-sha")

    def download(repo_id, filename, *, revision, cache_dir, token):
        revisions.append(revision)
        return str(files[filename])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Hub, hf_hub_download=download))
    args = demo.build_parser().parse_args([
        "--hf-repo", "example/policy", "--simulator-root", str(tmp_path),
        "--output-dir", str(tmp_path / "output"),
    ])
    if tampered:
        with pytest.raises(ValueError, match="hash mismatch"):
            demo.resolve_policy(args)
    else:
        actual_checkpoint, actual_config, source = demo.resolve_policy(args)
        assert actual_checkpoint == checkpoint
        assert actual_config == config
        assert source["revision"] == "immutable-sha"
    assert revisions == ["immutable-sha"] * 3
