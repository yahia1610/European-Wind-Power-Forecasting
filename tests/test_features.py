"""Tests for feature engineering and leakage-safe feature selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.features import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    add_calendar_features,
    build_feature_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed" / "de_lu_hourly.parquet"


def test_forbidden_features_are_not_model_inputs():
    for forbidden in FORBIDDEN_FEATURES:
        assert forbidden not in FEATURE_COLUMNS
    # The target itself must never be a feature.
    assert "wind_total_actual_mw" not in FEATURE_COLUMNS


def test_calendar_features_are_added():
    df = pd.read_parquet(PROCESSED)
    out = add_calendar_features(df)
    for col in (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
        "season_winter", "season_spring", "season_summer", "season_autumn",
        "forecast_horizon_hours",
    ):
        assert col in out.columns


def test_cyclical_encodings_are_bounded():
    df = add_calendar_features(pd.read_parquet(PROCESSED))
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"):
        assert df[col].between(-1.0, 1.0).all()


def test_forecast_horizon_matches_decision_time():
    df = add_calendar_features(pd.read_parquet(PROCESSED))
    expected = (
        pd.Series(df.index, index=df.index)
        - pd.to_datetime(df["decision_time_utc"], utc=True)
    ).dt.total_seconds() / 3600.0
    pd.testing.assert_series_equal(
        df["forecast_horizon_hours"], expected, check_names=False
    )


def test_eligible_mask_excludes_warmup_and_target_nan():
    df = add_calendar_features(pd.read_parquet(PROCESSED))
    X, y, eligible = build_feature_matrix(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert eligible.sum() < len(df)  # warm-up rows excluded
    assert not eligible.iloc[0]  # first row has no history
    assert y[eligible].notna().all()
    assert X[eligible].notna().all().all()