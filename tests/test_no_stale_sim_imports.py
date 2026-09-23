"""Regression: GlucoAlg must import the simulator package under its current name."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
LEGACY_IMPORT = re.compile(r'^\s*(import|from)\s+glucobench', re.MULTILINE)


def test_no_legacy_sim_imports():
    offenders = [
        str(path) for path in REPO.rglob('*.py')
        if '__pycache__' not in path.parts and LEGACY_IMPORT.search(path.read_text())
    ]
    assert offenders == [], offenders
