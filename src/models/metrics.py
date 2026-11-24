"""Evaluation metrics for day-ahead wind forecasting.

Normalization note
------------------
A true installed-capacity nMAE requires a reliable (ideally time-dependent)
installed wind capacity for DE-LU. That series is **not** available in this
project, so we do **not** report an installed-capacity nMAE. Instead we report
``mean_normalized_mae`` = MAE / mean(actual generation), which is explicitly a
mean-normalized MAE, not an installed-capacity metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INSTALLED_CAPACITY_AVAILABLE = False
NORMALIZATION_DENOMINATOR = "mean_actual_generation"


def _clean(y_true, y_pred):
    # Align by position: callers pass predictions in the same row order.
    y_true = pd.Series(y_true).reset_index(drop=True).astype(float)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(float)
    mask = y_true.notna() & y_pred.notna()
    return y_true[mask], y_pred[mask]


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, bias, and mean-normalized MAE."""

    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        raise ValueError("no valid (y_true, y_pred) pairs")

    error = y_pred - y_true
    mae = float(error.abs().mean())
    rmse = float(np.sqrt((error**2).mean()))
    bias = float(error.mean())
    mean_actual = float(y_true.mean())
    mean_normalized_mae = (
        float(mae / mean_actual) if mean_actual != 0 else float("nan")
    )

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mean_normalized_mae": mean_normalized_mae,
        "normalization_denominator": NORMALIZATION_DENOMINATOR,
        "installed_capacity_nmae_available": INSTALLED_CAPACITY_AVAILABLE,
        "mean_actual": mean_actual,
    }


def improvement_pct(mae_baseline: float, mae_model: float) -> float:
    """Percentage MAE reduction relative to a baseline (positive = better)."""

    if mae_baseline == 0:
        return float("nan")
    return float((mae_baseline - mae_model) / mae_baseline * 100.0)