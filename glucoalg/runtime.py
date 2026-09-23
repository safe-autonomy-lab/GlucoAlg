"""Explicit simulator initialization and source provenance."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys


def configure_runtime() -> None:
    """Default JAX to CPU, respecting an explicitly selected backend."""
    if "JAX_PLATFORM_NAME" not in os.environ:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def initialize_simulator(simulator_root: str | Path | None = None) -> dict:
    """Register the CMDP and reject an already imported, conflicting checkout.

    ``simulator_root`` or ``GLUCOSIM_ROOT`` pins a checkout containing the
    glucosim package. Without either, use the installed package and report its
    actual location. Never mutate a running process's imported simulator.
    """
    configure_runtime()
    requested = simulator_root or os.environ.get("GLUCOSIM_ROOT")
    root = Path(requested).expanduser().resolve() if requested else None
    if root is not None:
        expected = root / "glucosim" / "__init__.py"
        if not expected.is_file():
            raise ValueError(f"Simulator root must contain glucosim/__init__.py: {root}")
        existing = sys.modules.get("glucosim")
        if existing is not None and Path(existing.__file__).resolve() != expected:
            raise RuntimeError(
                f"glucosim already loaded from {existing.__file__}; requested {expected}. "
                "Start a fresh process with the intended PYTHONPATH or --simulator-root."
            )
        # A stale editable checkout may precede a valid root already on the path.
        sys.path[:] = [entry for entry in sys.path if entry != str(root)]
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
    sim = importlib.import_module("glucosim")
    actual = Path(sim.__file__).resolve()
    if root is not None and actual != root / "glucosim" / "__init__.py":
        raise RuntimeError(f"Simulator source mismatch: {actual}; requested {root}")
    importlib.import_module("glucosim.diabetes_cmdp")
    source = actual.parent.parent
    algo_source = Path(__file__).resolve().parents[1]
    return {
        "glucosim_file": str(actual),
        "glucosim_commit": _git(source, "rev-parse", "HEAD"),
        "glucosim_tracked_status": _git(source, "status", "--porcelain", "--untracked-files=no"),
        "glucoalg_root": str(algo_source),
        "glucoalg_commit": _git(algo_source, "rev-parse", "HEAD"),
        "python": sys.executable,
        "python_prefix": sys.prefix,
        "packages": {
            name: {"version": getattr(sys.modules[name], "__version__", None),
                   "file": getattr(sys.modules[name], "__file__", None)}
            for name in ("torch", "jax", "gymnasium", "numpy") if name in sys.modules
        },
        "jax_platforms": os.environ.get("JAX_PLATFORMS", os.environ.get("JAX_PLATFORM_NAME")),
    }
