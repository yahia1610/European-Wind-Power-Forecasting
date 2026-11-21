"""Smoke tests to verify the environment and project layout.

These tests are intentionally simple. They confirm that:
- Python can import the project package.
- Core libraries are installed and importable.
- The expected folders exist.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_numpy_and_pandas_import():
    """NumPy and pandas should be importable and usable."""
    arr = np.array([1, 2, 3])
    series = pd.Series(arr)
    assert series.sum() == 6


def test_project_package_imports():
    """The main package should import without errors."""
    import src

    assert src.__version__ == "0.1.0"


def test_expected_directories_exist():
    """The core project folders should exist."""
    for name in ["data", "data/raw", "src", "notebooks", "tests"]:
        assert (PROJECT_ROOT / name).is_dir(), f"Missing folder: {name}"


def test_dotenv_available():
    """python-dotenv should be installed (used for .env secrets)."""
    from dotenv import load_dotenv

    assert callable(load_dotenv)