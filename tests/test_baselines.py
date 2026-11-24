"""Tests for naive forecasting baselines."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models.baselines import (
    build_baselines,
    persistence_forecast,
    seasonal_naive_forecast,
)

UTC = timezone.utc


def make_frame(periods: int = 200) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="h", tz=UTC)
    df = pd.DataFrame(
        {"wind_total_actual_mw": np.arange(periods, dtype=float)}, index=index
    )
    # Fixed decision time: 2020-01-03 17:00 UTC (18:00 CET).
    df["decision_time_utc"] = pd.Timestamp("2020-01-03 17:00", tz=UTC)
    return df


def test_persistence_uses_last_available_actual():
    df = make_frame()
    pred = persistence_forecast(df)
    # Last interval start with H+1 <= decision is 2020-01-03 16:00 -> value 64.
    assert pred.iloc[100] == 64.0
    assert pred.notna().all()


def test_seasonal_naive_48_uses_lag_48():
    df = make_frame()
    pred = seasonal_naive_forecast(df, 48)
    # Row 84 is 2020-01-04 12:00; 48 h earlier is row 36 -> value 36.
    assert pred.iloc[84] == 36.0


def test_seasonal_naive_168_uses_lag_168():
    df = make_frame()
    pred = seasonal_naive_forecast(df, 168)
    # Row 180 is 2020-01-08 12:00; 168 h earlier is row 12 -> value 12.
    assert pred.iloc[180] == 12.0


def test_seasonal_naive_masks_values_after_decision():
    df = make_frame()
    pred = seasonal_naive_forecast(df, 48)
    # Row 190 source is 2020-01-06 22:00, after the 2020-01-03 decision -> NaN.
    assert pd.isna(pred.iloc[190])


def test_baselines_never_use_future_actuals():
    df = make_frame()
    decision = pd.to_datetime(df["decision_time_utc"], utc=True)
    index = pd.Series(df.index, index=df.index)

    for lag in (48, 168):
        pred = seasonal_naive_forecast(df, lag)
        mask = pred.notna()
        source = index[mask] - pd.Timedelta(hours=lag)
        assert ((source + pd.Timedelta(hours=1)) <= decision[mask]).all()


def test_build_baselines_columns():
    df = make_frame()
    out = build_baselines(df)
    assert list(out.columns) == [
        "persistence",
        "seasonal_naive_48h",
        "seasonal_naive_168h",
    ]
    assert len(out) == len(df)