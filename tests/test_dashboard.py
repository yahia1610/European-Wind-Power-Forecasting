"""Tests for the Streamlit dashboard data layer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.dashboard.data import (
    DashboardDataError,
    compute_metrics,
    error_by_hour,
    error_by_season,
    filter_by_date,
    load_metadata,
    load_results,
    model_table,
    predictions_aligned,
)

REQUIRED = {
    "timestamp": pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 02:00"], utc=True
    ),
    "actual_wind_mw": [100.0, 110.0, 120.0],
    "ridge_pred_mw": [105.0, 108.0, 130.0],
    "lightgbm_pred_mw": [102.0, 112.0, 118.0],
    "tso_forecast_mw": [101.0, 109.0, 121.0],
    "forecast_horizon_hours": [6.0, 7.0, 8.0],
    "season": ["Winter", "Winter", "Winter"],
}

METADATA = {
    "coverage": {"start": "2019-01-01", "end": "2026-09-06", "rows": 67343},
    "models_test_metrics": [
        {"model": "persistence", "mae": 200.0, "rmse": 250.0, "bias": 5.0,
         "mean_normalized_mae": 0.5, "improvement_vs_persistence_pct": None},
        {"model": "Ridge", "mae": 150.0, "rmse": 200.0, "bias": -3.0,
         "mean_normalized_mae": 0.4, "improvement_vs_persistence_pct": 25.0},
    ],
}


def write_results(path):
    pd.DataFrame(REQUIRED).to_parquet(path, index=False)


def test_load_results_ok(tmp_path):
    path = tmp_path / "results.parquet"
    write_results(path)
    df = load_results(path)
    assert list(df.columns)[:5] == [
        "timestamp", "actual_wind_mw", "ridge_pred_mw", "lightgbm_pred_mw", "tso_forecast_mw",
    ]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].is_monotonic_increasing


def test_missing_results_file_raises(tmp_path):
    with pytest.raises(DashboardDataError):
        load_results(tmp_path / "nope.parquet")


def test_missing_required_columns_raises(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"], utc=True)}).to_parquet(path, index=False)
    with pytest.raises(DashboardDataError):
        load_results(path)


def test_load_metadata_ok(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(METADATA), encoding="utf-8")
    meta = load_metadata(path)
    assert meta["coverage"]["rows"] == 67343


def test_missing_metadata_raises(tmp_path):
    with pytest.raises(DashboardDataError):
        load_metadata(tmp_path / "nope.json")


def test_filter_by_date_inclusive():
    df = pd.DataFrame(REQUIRED)
    filtered = filter_by_date(df, "2025-01-01", "2025-01-01")
    assert len(filtered) == 3
    assert filter_by_date(df, "2025-01-02", "2025-01-02").empty


def test_predictions_aligned():
    df = pd.DataFrame(REQUIRED)
    assert predictions_aligned(df) is True
    bad = df.copy()
    bad.loc[0, "ridge_pred_mw"] = None
    assert predictions_aligned(bad) is False


def test_compute_metrics_values():
    df = pd.DataFrame(REQUIRED)
    metrics = compute_metrics(df)
    # Ridge errors: +5, -2, +10 -> MAE = 17/3
    assert abs(metrics["ridge_pred_mw"]["mae"] - 17 / 3) < 1e-9
    assert metrics["ridge_pred_mw"]["normalization_denominator"] == "mean_actual_generation"
    assert metrics["tso_forecast_mw"]["mae"] < metrics["ridge_pred_mw"]["mae"]


def test_model_table_includes_persistence_and_improvement():
    df = pd.DataFrame(REQUIRED)
    table = model_table(df, METADATA).set_index("model")
    assert "Persistence" in table.index
    assert table.loc["Ridge", "improvement_vs_persistence_pct"] > 0


def test_error_by_hour_and_season():
    df = pd.DataFrame(REQUIRED)
    by_hour = error_by_hour(df, "ridge_pred_mw")
    assert list(by_hour.columns) == ["mean_error", "mae"]
    assert len(by_hour) == 3
    by_season = error_by_season(df, "ridge_pred_mw")
    assert "Winter" in by_season.index


def test_real_dashboard_artifacts_load():
    df = load_results()
    meta = load_metadata()
    assert len(df) > 0
    assert meta["tso_is_best_benchmark"] is True
    assert predictions_aligned(df) is True


def test_real_filter_by_date():
    df = load_results()
    start = df["timestamp"].min().date()
    end = df["timestamp"].max().date()
    filtered = filter_by_date(df, start, end)
    assert len(filtered) == len(df)
    one_day = filter_by_date(df, start, start)
    assert len(one_day) <= 24