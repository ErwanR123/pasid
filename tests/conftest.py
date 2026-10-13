import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def load_script():
    """Import a data script by path : their names start with a digit."""

    def load(name):
        path = Path(__file__).parents[1] / "data_scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
