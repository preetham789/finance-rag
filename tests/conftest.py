import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def eval_api():
    return _load("eval_api")


@pytest.fixture(scope="session")
def bench():
    return _load("bench_latency")


@pytest.fixture(scope="session")
def purge():
    return _load("purge_source")


@pytest.fixture(scope="session")
def root():
    return ROOT
