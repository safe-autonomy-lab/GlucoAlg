"""Optional Slurm resource settings, kept separate from scientific study specs."""
from __future__ import annotations

import re
from pathlib import Path

from .spec import SpecError, load_json_nodupes

PROFILE_KEYS = {
    "partition", "account", "qos", "time", "cpus_per_task", "memory",
    "exclusive", "max_concurrent",
}
_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"


def validate_profile(raw: object) -> dict:
    """Validate explicit resources; omitted keys use the scheduler's defaults.

    Profiles deliberately contain no executable commands, interpreter/source
    paths, training settings, or arbitrary additional SBATCH directives.
    """
    if not isinstance(raw, dict):
        raise SpecError("execution profile must be a JSON object")
    for key in raw:
        if key not in PROFILE_KEYS:
            raise SpecError(f"unknown execution profile key {key!r}")
    result = dict(raw)
    for key, value in result.items():
        label = f"execution profile {key}"
        if key in {"cpus_per_task", "max_concurrent"}:
            if type(value) is not int or value <= 0:
                raise SpecError(f"{label} must be a positive integer")
        elif key == "exclusive":
            if type(value) is not bool:
                raise SpecError(f"{label} must be a JSON boolean")
        else:
            if not isinstance(value, str) or not value or re.search(r"\s", value):
                raise SpecError(f"{label} must be a nonempty single token")
            if key in {"partition", "account", "qos"}:
                pattern = rf"{_NAME}(?:,{_NAME})*" if key == "partition" else _NAME
                if re.fullmatch(pattern, value) is None:
                    raise SpecError(f"{label} contains unsupported directive characters")
            elif key == "memory":
                match = re.fullmatch(r"([0-9]+)([KMGTkmgt]?)", value)
                if match is None or int(match[1]) <= 0:
                    raise SpecError(f"{label} must be a positive integer with optional K/M/G/T suffix")
                result[key] = value.upper()
            elif key == "time":
                # Slurm accepts minutes, minutes:seconds, hours:minutes:seconds,
                # and days-hours[:minutes[:seconds]]. Require a bounded duration.
                if re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?::[0-9]{1,2}){0,2}", value) is None:
                    raise SpecError(f"{label} must be a numeric Slurm duration")
                clock = value.rsplit("-", 1)[-1].split(":")
                if any(int(part) >= 60 for part in clock[1:]):
                    raise SpecError(f"{label} minutes/seconds fields must be below 60")
                if not any(int(part) for part in re.split("[-:]", value)):
                    raise SpecError(f"{label} must be a positive duration")
    return result


def resolve_profile(
    profile: str | Path | None = None, **overrides: object
) -> dict:
    """Load a profile and apply only supplied (non-None) invocation overrides."""
    raw = {} if profile is None else load_json_nodupes(str(profile))
    resources = validate_profile(raw)
    # Validate overrides separately: a typo must not disappear because its value
    # is None, and a CLI override must not hide an invalid profile file.
    for key in overrides:
        if key not in PROFILE_KEYS:
            raise SpecError(f"unknown execution profile override {key!r}")
    resources.update({key: value for key, value in overrides.items() if value is not None})
    return validate_profile(resources)
