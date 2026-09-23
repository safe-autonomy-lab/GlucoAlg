"""CLI entry point: python -m glucoalg.tuning plan|run|summarize|export-slurm|validate|status."""
from __future__ import annotations

import argparse
import sys

from . import execute as execmod
from . import plan as planmod
from . import summarize as summod
from .plan import RunError
from .spec import (
    SpecError,
    expand_configs,
    load_json_nodupes,
    normalize_spec,
    spec_hash,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m glucoalg.tuning",
        description="Categorical, multi-seed HPO workflow for GlucoAlg (grid + deterministic random subset).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate a spec file (needs no simulator)")
    p.add_argument("--spec", required=True, help="tuning spec JSON")

    p = sub.add_parser("plan", help="seal a study plan directory (before running)")
    p.add_argument("--spec", required=True, help="tuning spec JSON")
    p.add_argument("--output", required=True, help="study directory to create (must be empty)")
    p.add_argument("--simulator-root", default=None, help="GlucoSim checkout (or set spec simulator_root)")

    p = sub.add_parser("run", help="run planned seed jobs via subprocess")
    p.add_argument("--plan", required=True, help="sealed study directory")
    p.add_argument("--all", dest="all_jobs", action="store_true", help="run every planned job")
    p.add_argument("--job", default=None, help="run one job ID only")
    p.add_argument("--resume", action="store_true", help="continue a started study (verified successes skipped)")
    p.add_argument("--retry-failed", action="store_true", help="also rerun failed jobs")
    p.add_argument("--simulator-root", default=None, help="must match the sealed plan when given")
    p.add_argument("--runner", default=None, choices=("auto", "glucoalg.train", "run.py"))
    p.add_argument("--dry-run", action="store_true", help="print planned commands without launching")

    p = sub.add_parser("summarize", help="aggregate logs and pick a provisional winner")
    p.add_argument("--plan", required=True, help="sealed study directory")
    p.add_argument("--out", default=None, help="output dir (default: <plan>/summary)")
    p.add_argument("--plot", action="store_true", help="also render tradeoff.png (needs matplotlib)")
    p.add_argument("--spec", default=None, help="optional spec file to check against the plan")

    p = sub.add_parser("export-slurm", help="render a batch resume script (never submits)")
    p.add_argument("--plan", required=True, help="sealed study directory")
    p.add_argument("--out", required=True, help="script path to write")
    p.add_argument("--profile", default=None, help="optional execution profile JSON")
    p.add_argument("--partition", default=None, help="override profile partition")
    p.add_argument("--account", default=None, help="override profile accounting allocation")
    p.add_argument("--qos", default=None, help="override profile quality of service")
    p.add_argument("--time", default=None, help="override profile Slurm time limit")
    p.add_argument("--cpus-per-task", type=int, default=None, help="override profile CPU allocation")
    p.add_argument("--memory", default=None, help="override profile memory per node (e.g. 8G)")
    p.add_argument("--max-concurrent", type=int, default=None, help="limit concurrent array tasks")
    p.add_argument("--job-name", default=None)
    exclusive = p.add_mutually_exclusive_group()
    exclusive.add_argument("--exclusive", dest="exclusive", action="store_true", default=None)
    exclusive.add_argument("--no-exclusive", dest="exclusive", action="store_false")

    p = sub.add_parser("status", help="show per-job states for a study")
    p.add_argument("--plan", required=True, help="sealed study directory")
    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    spec = normalize_spec(load_json_nodupes(args.spec))
    configs = expand_configs(spec)
    print(f"spec OK: study={spec['study_name']} mode={spec['search']['mode']} "
          f"configs={len(configs)} seeds={spec['seeds']} jobs={len(configs) * len(spec['seeds'])} "
          f"spec_hash={spec_hash(spec)[:12]}")
    for cfg in configs:
        print(f"  {cfg['config_id']}: {cfg['slug']}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    manifest = planmod.create_plan(args.spec, args.output, simulator_root=args.simulator_root)
    print(f"sealed {args.output}: study={manifest['study_name']} "
          f"configs={manifest['search']['n_configs']} jobs={len(manifest['jobs'])} "
          f"spec={manifest['spec_hash'][:12]} plan={manifest['_plan_hash'][:12]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    summary = execmod.run_study(
        args.plan, all_jobs=args.all_jobs, job_id=args.job, resume=args.resume,
        retry_failed=args.retry_failed, simulator_root=args.simulator_root,
        runner=args.runner, dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"done: ran={len(summary['ran'])} ok={len(summary['succeeded'])} "
              f"skipped_ok={len(summary['skipped_success'])} skipped_failed={len(summary['skipped_failed'])}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summod.summarize_study(args.plan, out_dir=args.out, make_plot=args.plot, spec_path=args.spec)
    return 0


def cmd_export_slurm(args: argparse.Namespace) -> int:
    from .slurm import export_slurm

    path = export_slurm(
        args.plan, args.out, profile=args.profile, partition=args.partition,
        account=args.account, qos=args.qos, time=args.time, cpus_per_task=args.cpus_per_task,
        memory=args.memory, max_concurrent=args.max_concurrent,
        job_name=args.job_name, exclusive=args.exclusive,
    )
    print(f"wrote {path}; review it, then submit by hand (this tool never runs sbatch)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest = planmod.load_manifest(args.plan)
    records = planmod.read_ledger(args.plan)
    execmod.check_ledger_plan(records, manifest)
    counts = {"pending": 0, "success": 0, "failed": 0}
    for job in manifest["jobs"]:
        state, reason = execmod.job_state(manifest, args.plan, records, job["job_id"])
        counts[state] += 1
        print(f"{state:8s} {job['job_id']} ({reason})")
    print(f"{manifest['study_name']}: {counts['success']} ok / {counts['failed']} failed / "
          f"{counts['pending']} pending across {len(manifest['jobs'])} jobs")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {
            "validate": cmd_validate, "plan": cmd_plan, "run": cmd_run,
            "summarize": cmd_summarize, "export-slurm": cmd_export_slurm,
            "status": cmd_status,
        }[args.command](args)
    except SpecError as exc:
        print(f"tuning spec error: {exc}", file=sys.stderr)
        return 2
    except RunError as exc:
        print(f"tuning error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
