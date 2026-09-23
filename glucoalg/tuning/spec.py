"""Categorical HPO spec: schema, validation, and search-space expansion.

Uses the lightweight training configuration builder for semantic validation.
Neither ``validate`` nor ``plan`` initializes the simulator, JAX or PyTorch.
"""
from __future__ import annotations

import hashlib
import contextlib
import io
import json
import math
import random
import re

DEFAULT_SEEDS = [100, 101, 102]

# Normalized flag name (dashes) -> CLI spelling to emit. Spellings match the
# current root run.py (incl. legacy ``--cost_limit`` / ``--target_kl``) so the
# ``run.py`` fallback runner keeps working; the parent training CLI accepts
# these plus the ``--cost-limit`` alias.
KNOWN_OPTIONS = {
    "algo": "--algo",
    "env-id": "--env-id",
    "cohort": "--cohort",
    "total-steps": "--total-steps",
    "device": "--device",
    "vector-env-nums": "--vector-env-nums",
    "entropy-coef": "--entropy-coef",
    "safety-bonus": "--safety-bonus",
    "penalty-type": "--penalty-type",
    "use-wandb": "--use-wandb",
    "steps-per-epoch": "--steps-per-epoch",
    "target-kl": "--target_kl",
    "batch-size": "--batch-size",
    "lagrangian-multiplier-init": "--lagrangian-multiplier-init",
    "lambda-lr": "--lambda-lr",
    "project-name": "--project-name",
    "actor-lr": "--actor-lr",
    "critic-lr": "--critic-lr",
    "parallel": "--parallel",
    "cost-limit": "--cost_limit",
    "simulator-root": "--simulator-root",
    "log-dir": "--log-dir",
}

# Options the tuning runner manages itself; forbidden in spec factors/train.
MANAGED_OPTIONS = {"seed", "log-dir", "simulator-root", "dry-run"}

TOP_LEVEL_KEYS = {
    "study_name", "search", "seeds", "cost_limit",
    "factors", "train", "simulator_root", "runner",
}
SEARCH_KEYS = {"mode", "size", "seed"}
RUNNERS = ("auto", "glucoalg.train", "run.py")
SEARCH_MODES = ("grid", "random_subset")

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


class SpecError(ValueError):
    """Raised for any invalid tuning spec (CLI exit code 2)."""


def normalize_key(key: object, where: str) -> str:
    if not isinstance(key, str):
        raise SpecError(f"{where}: option names must be strings, got {type(key).__name__}")
    norm = key.strip().replace("_", "-")
    if not norm or not _KEY_RE.match(norm):
        raise SpecError(f"{where}: unsafe option name {key!r}")
    return norm


def _no_dupes(pairs: list[tuple[str, object]]) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise SpecError(f"duplicate key {key!r}")
        out[key] = value
    return out


def load_json_nodupes(path: str) -> object:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=_no_dupes)
    except SpecError:
        raise
    except (OSError, ValueError) as exc:
        raise SpecError(f"cannot parse JSON spec {path}: {exc}") from exc


def check_scalar(value: object, where: str) -> None:
    """Allow only finite simple scalars passable as one CLI argument."""
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        try:
            math.isfinite(value)
        except OverflowError as exc:
            raise SpecError(f"{where}: integer too large for training") from exc
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SpecError(f"{where}: non-finite number {value!r}")
        return
    if isinstance(value, str):
        if not value:
            raise SpecError(f"{where}: empty string level")
        return
    raise SpecError(f"{where}: unsupported value {value!r} (want finite number, bool, or string)")


def encode_level(value: object) -> str:
    """Stable one-argument CLI encoding (also the dedup identity)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _check_option_key(norm: str, where: str) -> None:
    if norm in MANAGED_OPTIONS:
        raise SpecError(f"{where}: {norm!r} is managed by the runner and must not be set in the spec")
    if norm not in KNOWN_OPTIONS:
        raise SpecError(f"{where}: unknown training option {norm!r}")


def normalize_spec(raw: object) -> dict:
    """Validate a parsed spec and return the normalized form (raises SpecError)."""
    if not isinstance(raw, dict):
        raise SpecError("spec root must be a JSON object")
    top = {}
    for key, value in raw.items():
        norm = key.replace("-", "_") if isinstance(key, str) else key
        if norm not in TOP_LEVEL_KEYS:
            raise SpecError(f"unknown top-level key {key!r}")
        if norm in top:
            raise SpecError(f"duplicate top-level key {key!r}")
        top[norm] = value

    study = top.get("study_name")
    if not isinstance(study, str) or not study.strip():
        raise SpecError("study_name must be a non-empty string")

    search_raw = top.get("search", {"mode": "grid"})
    if not isinstance(search_raw, dict):
        raise SpecError("search must be an object")
    for key in search_raw:
        if key not in SEARCH_KEYS:
            raise SpecError(f"unknown search key {key!r}")
    mode = search_raw.get("mode", "grid")
    if mode not in SEARCH_MODES:
        raise SpecError(f"search.mode must be one of {SEARCH_MODES}, got {mode!r}")
    search: dict = {"mode": mode}
    if mode == "grid":
        if "size" in search_raw or "seed" in search_raw:
            raise SpecError("search.size/search.seed are only valid for mode random_subset")
    else:
        size = search_raw.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise SpecError("search.size must be a positive int for mode random_subset")
        seed = search_raw.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SpecError("search.seed must be an int")
        search["size"] = size
        search["seed"] = seed

    seeds = top.get("seeds", list(DEFAULT_SEEDS))
    if not isinstance(seeds, list) or not seeds:
        raise SpecError("seeds must be a non-empty list of ints")
    for seed in seeds:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise SpecError(f"seeds must be unsigned 32-bit ints, got {seed!r}")
    if len(set(seeds)) != len(seeds):
        raise SpecError(f"duplicate seeds {seeds!r}")

    cost_limit = top.get("cost_limit")
    if isinstance(cost_limit, bool) or not isinstance(cost_limit, (int, float)):
        raise SpecError("cost_limit (selection gate) is required and must be a number")
    check_scalar(cost_limit, "cost_limit")
    if not math.isfinite(cost_limit) or cost_limit < 0:
        raise SpecError("cost_limit must be finite and nonnegative")

    factors_raw = top.get("factors")
    if not isinstance(factors_raw, dict) or not factors_raw:
        raise SpecError("factors must be a non-empty object of {option: [levels]}")
    factors: dict = {}
    for key, levels in factors_raw.items():
        norm = normalize_key(key, "factors")
        _check_option_key(norm, "factors")
        if norm in factors:
            raise SpecError(f"duplicate factor {norm!r}")
        if isinstance(levels, dict):
            raise SpecError(
                f"factors[{norm!r}]: continuous distributions (min/max) are rejected; "
                "give explicit categorical levels as a list"
            )
        if not isinstance(levels, list) or not levels:
            raise SpecError(f"factors[{norm!r}]: levels must be a non-empty list")
        seen = set()
        for level in levels:
            check_scalar(level, f"factors[{norm!r}]")
            code = float(level) if type(level) in (int, float) else encode_level(level)
            if code in seen:
                raise SpecError(f"factors[{norm!r}]: duplicate level {level!r}")
            seen.add(code)
        factors[norm] = list(levels)

    train_raw = top.get("train")
    if not isinstance(train_raw, dict) or not train_raw:
        raise SpecError("train must be a non-empty object of fixed training options")
    train: dict = {}
    for key, value in train_raw.items():
        norm = normalize_key(key, "train")
        _check_option_key(norm, "train")
        if norm in train:
            raise SpecError(f"duplicate train option {norm!r}")
        if norm in factors:
            raise SpecError(f"{norm!r} is both a fixed train option and a factor")
        check_scalar(value, f"train[{norm!r}]")
        train[norm] = value
    for budget_key in ("total-steps", "steps-per-epoch"):
        if budget_key not in train and budget_key not in factors:
            raise SpecError(
                f"{budget_key!r} must be pinned explicitly in train (or swept as a factor)"
            )
    if "device" not in factors:
        train.setdefault("device", "cpu")
    if "vector-env-nums" not in factors:
        train.setdefault("vector-env-nums", 1)

    sim_root = top.get("simulator_root")
    if sim_root is not None and (not isinstance(sim_root, str) or not sim_root.strip()):
        raise SpecError("simulator_root must be a non-empty path string")

    runner = top.get("runner", "auto")
    if runner not in RUNNERS:
        raise SpecError(f"runner must be one of {RUNNERS}, got {runner!r}")

    spec = {
        "study_name": study.strip(),
        "search": search,
        "seeds": list(seeds),
        "cost_limit": cost_limit,
        "factors": factors,
        "train": train,
        "simulator_root": sim_root.strip() if isinstance(sim_root, str) else None,
        "runner": runner,
    }
    _check_budgets(spec)
    if mode == "random_subset" and search["size"] > _full_combos(spec):
        raise SpecError(
            f"search.size {search['size']} exceeds full grid size {_full_combos(spec)}"
        )
    _validate_training_configs(spec)
    return spec


def _validate_training_configs(spec: dict) -> None:
    """Use the training configuration authority, without importing JAX/torch."""
    from glucoalg.train import build_config, build_parser, parse_bool

    parser = build_parser()
    actions = {option: action for action in parser._actions for option in action.option_strings}
    for key in set(spec["train"]) | set(spec["factors"]):
        values = spec["factors"].get(key, [spec["train"].get(key)])
        converter = actions[KNOWN_OPTIONS[key]].type
        for value in values:
            if converter is int and type(value) is not int:
                raise SpecError(f"{key} requires JSON integers")
            if converter is float and type(value) not in (float, int):
                raise SpecError(f"{key} requires JSON numbers")
            if converter is parse_bool and type(value) is not bool:
                raise SpecError(f"{key} requires JSON booleans")
    for config in expand_configs(spec):
        options = {**spec["train"], **config["params"]}
        if options.get("device") != "cpu":
            raise SpecError("The tuning runner currently supports device=cpu only")
        argv = [f"{KNOWN_OPTIONS[key]}={encode_level(value)}" for key, value in options.items()]
        error = io.StringIO()
        try:
            with contextlib.redirect_stderr(error):
                args = parser.parse_args(argv)
            build_config(args)
        except (ValueError, SystemExit) as exc:
            detail = error.getvalue().strip().splitlines()
            raise SpecError(f"{config['config_id']}: invalid training configuration: {detail[-1] if detail else exc}") from exc


def _check_budgets(spec: dict) -> None:
    for key in ("total-steps", "steps-per-epoch"):
        values = spec["factors"].get(key, [spec["train"].get(key)])
        for value in values:
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SpecError(f"{key!r} levels must be positive ints, got {value!r}")


def _full_combos(spec: dict) -> int:
    total = 1
    for levels in spec["factors"].values():
        total *= len(levels)
    return total


def spec_hash(spec: dict) -> str:
    """Stable hash of the science-bearing spec subset (excludes paths/runner)."""
    core = {key: spec[key] for key in ("study_name", "search", "seeds", "cost_limit", "factors", "train")}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def expand_configs(spec: dict) -> list[dict]:
    """Expand factors into configs: [{config_id, slug, params}]. Deterministic."""
    names = sorted(spec["factors"])
    total = _full_combos(spec)
    if spec["search"]["mode"] == "grid" and total > 100_000:
        raise SpecError("Grid exceeds 100,000 configurations; use random_subset or reduce levels")
    selected_idx = range(total)
    if spec["search"]["mode"] == "random_subset":
        rng = random.Random(spec["search"]["seed"])
        try:
            selected_idx = sorted(rng.sample(selected_idx, spec["search"]["size"]))
        except (OverflowError, ValueError) as exc:
            raise SpecError("Random subset size or search space is too large") from exc
    width = max(2, len(str(len(selected_idx) - 1)))
    configs = []
    for pos, idx in enumerate(selected_idx):
        # Decode one Cartesian index without materializing the entire grid.
        params = {}
        for name in reversed(names):
            idx, level_index = divmod(idx, len(spec["factors"][name]))
            params[name] = spec["factors"][name][level_index]
        params = {name: params[name] for name in names}
        slug = ",".join(f"{name}={encode_level(params[name])}" for name in names)
        configs.append({"config_id": f"cfg{pos:0{width}d}", "slug": slug, "params": params})
    if len({config["config_id"] for config in configs}) != len(configs):
        raise SpecError("config ID collision; cannot plan")
    return configs


def expected_epochs(total_steps: int, steps_per_epoch: int) -> int:
    """Full expected epochs; training truncates the remainder (floor)."""
    return total_steps // steps_per_epoch
