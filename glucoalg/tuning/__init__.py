"""Categorical, multi-seed HPO workflow for GlucoAlg.

Exhaustive grid and deterministic random-subset search over explicit
categorical levels; sealed plans; subprocess launches; last-20% train-metric
aggregation with a provisional winner. Validation uses the lightweight training
configuration builder and PyYAML; matplotlib is loaded only for plots. Planning
and validation do not initialize the simulator, JAX or PyTorch.
"""

__version__ = "0.1.0"

from .plan import MetricError, RunError, create_plan, load_manifest
from .spec import SpecError, expand_configs, normalize_spec

__all__ = [
    "__version__",
    "MetricError",
    "RunError",
    "SpecError",
    "create_plan",
    "expand_configs",
    "load_manifest",
    "normalize_spec",
]
