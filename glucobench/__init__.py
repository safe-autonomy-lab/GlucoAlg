"""Compatibility name for legacy imports inside GlucoSim.

GlucoSim's bundled Safety-Gymnasium still imports ``glucobench.*``. Forward
those names to canonical ``glucosim.*`` modules without loading the same source
twice: separate copies would create different environment registries and class
identities. No simulator files or unrelated imports are changed.
"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


class _LegacyLoader(Loader):
    def __init__(self, legacy_name: str, canonical_name: str) -> None:
        self.legacy_name = legacy_name
        self.canonical_name = canonical_name

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        # Substitute after import rather than returning the canonical module
        # from create_module: that would overwrite its __spec__ with the alias.
        sys.modules[self.legacy_name] = import_module(self.canonical_name)


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("glucobench."):
            return None
        canonical_name = "glucosim" + fullname[len("glucobench"):]
        canonical_spec = find_spec(canonical_name)
        if canonical_spec is None:
            # Do not let the filesystem loader create a duplicate legacy copy.
            raise ModuleNotFoundError(
                f"No module named {canonical_name!r}", name=canonical_name
            )
        return spec_from_loader(
            fullname,
            _LegacyLoader(fullname, canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


# Install before importing GlucoSim, which may itself need the legacy name.
_finder = _LegacyFinder()
sys.meta_path.insert(0, _finder)
try:
    sys.modules[__name__] = import_module("glucosim")
except BaseException:
    sys.meta_path.remove(_finder)
    raise
