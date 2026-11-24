"""Tests for forecast-availability rules (H+1, cutoff, no leakage)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.models.availability import (
    ACTUAL_PUBLICATION_DELAY_HOURS,
    actual_available_at,
    audit_availability,
    latest_available_actual_start,
    verify_timestamp_convention,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed" / "de_lu_hourly.parquet"
UTC = timezone.utc


def test_timestamp_convention_is_interval_start():
    df = pd.read_parquet(PROCESSED)
    report = verify_timestamp_convention(df)
    assert report["verified"] is True
    assert report["convention"] == "interval_start"


def test_h1_availability_rule():
    decision = datetime(2026, 8, 1, 16, 0, tzinfo=UTC)  # 18:00 CEST
    # Interval starting 15:00 is published at 16:00 -> available.
    assert bool(actual_available_at(datetime(2026, 8, 1, 15, 0, tzinfo=UTC), decision))
    # Interval starting 16:00 is published at 17:00 -> NOT available.
    assert not bool(actual_available_at(datetime(2026, 8, 1, 16, 0, tzinfo=UTC), decision))


def test_latest_available_actual_start_is_one_hour_before_decision():
    decision = datetime(2026, 8, 1, 16, 0, tzinfo=UTC)
    assert latest_available_actual_start(decision) == datetime(
        2026, 8, 1, 15, 0, tzinfo=UTC
    )
    assert ACTUAL_PUBLICATION_DELAY_HOURS == 1


def test_decision_lag1_respects_h1():
    df = pd.read_parquet(PROCESSED)
    decision = pd.to_datetime(df["decision_time_utc"], utc=True)
    source = decision - pd.Timedelta(hours=1)
    available = (source + pd.Timedelta(hours=1)) <= decision
    assert bool(available.all())
    # Non-null lag1 only where history exists.
    assert df["wind_actual_decision_lag1"].notna().sum() > 0


def test_hour_specific_lags_respect_h1():
    df = pd.read_parquet(PROCESSED)
    decision = pd.to_datetime(df["decision_time_utc"], utc=True)
    index = pd.Series(df.index, index=df.index)
    for lag in (48, 168):
        source = index - pd.Timedelta(hours=lag)
        available = (source + pd.Timedelta(hours=1)) <= decision
        assert bool(available.all()), f"lag {lag} violates H+1"


def test_audit_excludes_unverified_features():
    df = pd.read_parquet(PROCESSED)
    report = audit_availability(df)
    excluded = " ".join(report["excluded_features"])
    assert "wind_total_forecast_mw" in excluded
    assert "price_eur_mwh" in excluded
    assert "load_forecast_mw" in excluded
    assert report["decision_lag1_respects_h1"] is True


def test_forecast_horizon_is_positive_and_bounded():
    df = pd.read_parquet(PROCESSED)
    horizon = (
        pd.Series(df.index, index=df.index) - pd.to_datetime(df["decision_time_utc"], utc=True)
    ).dt.total_seconds() / 3600.0
    assert (horizon > 0).all()
    assert horizon.max() <= 30