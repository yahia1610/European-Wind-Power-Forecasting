"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import pandas as pd

from src.models.metrics import (
    INSTALLED_CAPACITY_AVAILABLE,
    improvement_pct,
    regression_metrics,
)


def test_regression_metrics_known_values():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 4, 3]
    m = regression_metrics(y_true, y_pred)
    assert m["n"] == 4
    assert m["mae"] == 0.5
    assert math.isclose(m["rmse"], math.sqrt(0.5), rel_tol=1e-9)
    assert m["bias"] == 0.0
    assert math.isclose(m["mean_normalized_mae"], 0.2, rel_tol=1e-9)


def test_metrics_ignore_nan_pairs():
    m = regression_metrics([1, 2, None], [1, 3, 5])
    assert m["n"] == 2
    # Remaining pairs: (1,1) error 0 and (2,3) error 1 -> MAE 0.5.
    assert m["mae"] == 0.5


def test_nmae_is_mean_generation_normalized_not_capacity():
    m = regression_metrics([100, 200], [110, 190])
    assert m["normalization_denominator"] == "mean_actual_generation"
    assert m["installed_capacity_nmae_available"] is INSTALLED_CAPACITY_AVAILABLE
    assert INSTALLED_CAPACITY_AVAILABLE is False
    # MAE = 10, mean actual = 150 -> 0.0667
    assert abs(m["mean_normalized_mae"] - (10 / 150)) < 1e-9


def test_improvement_pct():
    assert improvement_pct(10.0, 8.0) == 20.0
    assert improvement_pct(10.0, 12.0) == -20.0