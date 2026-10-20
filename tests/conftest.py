import sys
from pathlib import Path

import pytest

# tests import the package from the repository root without installing it
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", help="run tests that launch Isaac Sim in subprocesses (slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--isaac"):
        return
    skip = pytest.mark.skip(reason="needs --isaac (launches Isaac Sim)")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip)
