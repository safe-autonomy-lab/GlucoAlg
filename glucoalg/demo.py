"""Saved local or Hugging Face policy -> sampled GlucoSim rollout -> GIF.

Requires the current GlucoAlg package and the matching GlucoSim source checkout.
This runs inference only, with no added runtime shield.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")


def resolve_policy(args):
    if args.hf_repo:
        if args.checkpoint or args.config:
            raise ValueError("Choose --hf-repo OR --checkpoint with --config")
        from huggingface_hub import HfApi, hf_hub_download

        revision = HfApi(token=False).model_info(args.hf_repo, revision=args.revision).sha

        def download(filename):
            return Path(hf_hub_download(args.hf_repo, filename, revision=revision,
                                       cache_dir=args.cache_dir, token=False))

        release = json.loads(download(f"releases/{args.release}/release.json").read_text())
        candidates = [p for p in release["policies"] if p["policy_seed"] == args.training_seed]
        if len(candidates) != 1:
            raise ValueError(f"Expected one policy for training seed {args.training_seed}; found {len(candidates)}")
        policy = candidates[0]
        checkpoint, config = download(policy["checkpoint"]), download(policy["config"])
        for path, key in [(checkpoint, "checkpoint_sha256"), (config, "config_sha256")]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != policy[key]:
                raise ValueError(f"Release hash mismatch: {path}")
        source = {"repo": args.hf_repo, "revision": revision, "release": args.release,
                  "release_simulator_commit": release.get("simulator_commit")}
        return checkpoint, config, source
    if not args.checkpoint or not args.config:
        raise ValueError("Supply --checkpoint and its matching --config, or --hf-repo")
    if args.revision:
        raise ValueError("--revision applies only to --hf-repo")
    return args.checkpoint.resolve(), args.config.resolve(), {"source": "local training run"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hf-repo")
    parser.add_argument("--revision", help="Optional Hugging Face revision; resolved once and pinned for all downloads")
    parser.add_argument("--release", default="rerun-20261008")
    parser.add_argument("--training-seed", type=int, default=100, help="Select uploaded policy seed; local runs use config.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--simulator-root", type=Path, required=True)
    parser.add_argument("--patient", default="adolescent#002")
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; previous results are never overwritten")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.days) or args.days < 1 or not math.isclose(args.days * 288, round(args.days * 288)):
        parser.error("--days must be at least one day in whole five-minute steps")
    if not math.isfinite(args.seconds) or args.seconds < 1:
        parser.error("--seconds must be finite and at least one")
    if args.eval_seed < 0 or args.training_seed < 0:
        parser.error("Seeds must be nonnegative")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    checkpoint, config_path, source = resolve_policy(args)
    config = json.loads(config_path.read_text())
    patient_type = config["env_id"].removesuffix("-v0")
    algorithm, training_seed = config["algo"], config["seed"]

    # Official loader restores categorical actor weights and saved observation
    # normalization. No actor reimplementation, random policy, or retraining.
    from glucoalg.eval_grid import run_grid
    import torch
    from glucoalg.rendering import render

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "policy_source.json").write_text(json.dumps(source, indent=2))
    print(f"Checkpoint: {checkpoint}\nConfig: {config_path}", flush=True)
    run_grid(
        checkpoint=checkpoint, config=config_path,
        output_dir=args.output_dir / "evaluation",
        patient_type=patient_type, patients=[args.patient],
        algorithm=algorithm, train_seed=training_seed,
        episodes=1, horizon_days=args.days, eval_seed_base=args.eval_seed,
        action_mode="stochastic", glucose_source="post-step",
        simulator_root=args.simulator_root, shield_type="none", save_traces=True,
    )
    trace = args.output_dir / "evaluation" / args.patient / "episode_0_controller.csv"
    render(trace, args.output_dir / "episode.gif",
           title=f"{algorithm} | {patient_type} {args.patient} | training seed {training_seed}",
           seconds=args.seconds)


if __name__ == "__main__":
    main()
