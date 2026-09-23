"""A source pin must never silently replace a simulator already in memory."""

from types import SimpleNamespace
import sys

import pytest

from glucoalg.runtime import initialize_simulator


def test_explicit_source_must_exist(tmp_path):
    with pytest.raises(ValueError, match="must contain"):
        initialize_simulator(tmp_path)


def test_already_imported_conflicting_source_rejected(tmp_path, monkeypatch):
    package = tmp_path / "glucosim"
    package.mkdir()
    (package / "__init__.py").touch()
    monkeypatch.setitem(sys.modules, "glucosim", SimpleNamespace(__file__="/wrong/glucosim/__init__.py"))
    with pytest.raises(RuntimeError, match="already loaded"):
        initialize_simulator(tmp_path)
