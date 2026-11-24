"""Tests for the DE-LU preprocessing pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    PRICE_METHODOLOGY_CHANGE_UTC,
    DataValidationError,
    add_derived_columns,
    add_lag_features,
    load_snapshot,
    make_chronological_split,
    validate_hourly_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = PROJECT_ROOT / "data" / "raw" / "snapshot.parquet"

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_frame(start: str, periods: int, freq: str = "h") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz=UTC)
    n = len(index)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "delivery_date": [d.date() for d in index],
            "price_eur_mwh": rng.normal(50, 20, n),
            "load_forecast_mw": rng.normal(50000, 5000, n),
            "wind_onshore_forecast_mw": rng.normal(8000, 2000, n),
            "wind_offshore_forecast_mw": rng.normal(2000, 800, n),
            "wind_onshore_actual_mw": rng.normal(8000, 2000, n),
            "wind_offshore_actual_mw": rng.normal(2000, 800, n),
        },
        index=index,
    )
    df.index.name = "timestamp_utc"
    return df


# --------------------------------------------------------------------------- #
# UTC index / frequency / duplicates
# --------------------------------------------------------------------------- #
def test_load_snapshot_sets_utc_index():
    df = load_snapshot(SNAPSHOT)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing


def test_validate_hourly_index_reports_clean_structure():
    df = load_snapshot(SNAPSHOT)
    report = validate_hourly_index(df)
    assert report["duplicate_timestamps"] == 0
    assert report["hourly_frequency"] is True
    assert report["missing_hours"] == 0
    assert report["monotonic_increasing"] is True


def test_validate_hourly_index_detects_duplicates():
    df = make_frame("2020-01-01", 48)
    df = pd.concat([df, df.iloc[[0]]])
    report = validate_hourly_index(df, name="dup")
    assert report["duplicate_timestamps"] == 1


def test_validate_hourly_index_rejects_non_utc():
    df = make_frame("2020-01-01", 24)
    df.index = df.index.tz_convert("Europe/Berlin")
    with pytest.raises(DataValidationError):
        validate_hourly_index(df)


# --------------------------------------------------------------------------- #
# Derived columns
# --------------------------------------------------------------------------- #
def test_total_wind_generation_is_sum_of_components():
    df = add_derived_columns(make_frame("2020-01-01", 48))
    expected = df["wind_onshore_actual_mw"] + df["wind_offshore_actual_mw"]
    pd.testing.assert_series_equal(
        df["wind_total_actual_mw"], expected, check_names=False
    )
    expected_f = df["wind_onshore_forecast_mw"] + df["wind_offshore_forecast_mw"]
    pd.testing.assert_series_equal(
        df["wind_total_forecast_mw"], expected_f, check_names=False
    )


def test_total_wind_generation_propagates_nan():
    df = make_frame("2020-01-01", 24)
    df.loc[df.index[0], "wind_onshore_actual_mw"] = np.nan
    out = add_derived_columns(df)
    assert pd.isna(out["wind_total_actual_mw"].iloc[0])
    assert bool(out["is_target_missing"].iloc[0]) is True


def test_missing_target_preserved_in_real_snapshot():
    df = load_snapshot(SNAPSHOT)
    out = add_derived_columns(df)
    missing = out["wind_total_actual_mw"].isna()
    assert int(missing.sum()) == 1
    assert out.loc[missing, "is_target_missing"].all()


def test_negative_prices_are_kept():
    df = load_snapshot(SNAPSHOT)
    out = add_derived_columns(df)
    assert (out["price_eur_mwh"] < 0).sum() > 0
    assert out["price_eur_mwh"].min() < 0


def test_price_methodology_flag_switches_on_change_date():
    df = make_frame("2025-09-30", 48)
    out = add_derived_columns(df)
    before = out.loc[out.index < PRICE_METHODOLOGY_CHANGE_UTC, "price_is_quarter_mean"]
    after = out.loc[out.index >= PRICE_METHODOLOGY_CHANGE_UTC, "price_is_quarter_mean"]
    assert not before.any()
    assert after.all()


# --------------------------------------------------------------------------- #
# Leakage prevention
# --------------------------------------------------------------------------- #
def test_hour_specific_lag_is_masked_after_decision():
    # 3 days of hourly data, but treat every row as delivery date 2020-01-02,
    # whose decision time is 2020-01-01 17:00 UTC (18:00 CET).
    df = make_frame("2020-01-01", 72)
    df["delivery_date"] = date(2020, 1, 2)
    out = add_lag_features(add_derived_columns(df))

    decision = datetime(2020, 1, 1, 17, 0, tzinfo=UTC)

    # A row whose lag-24 source is before the decision must be populated.
    ok_time = datetime(2020, 1, 2, 12, 0, tzinfo=UTC)  # source 2020-01-01 12:00
    ok_val = out.loc[ok_time, "wind_total_actual_mw_lag24"]
    assert not pd.isna(ok_val)

    # A row whose lag-24 source is after the decision must be masked.
    bad_time = datetime(2020, 1, 2, 18, 0, tzinfo=UTC)  # source 2020-01-01 18:00
    bad_val = out.loc[bad_time, "wind_total_actual_mw_lag24"]
    assert pd.isna(bad_val)


def test_decision_lag1_is_last_observation_before_cutoff():
    df = make_frame("2020-01-01", 72)
    df["delivery_date"] = date(2020, 1, 2)
    out = add_lag_features(add_derived_columns(df))

    decision = datetime(2020, 1, 1, 17, 0, tzinfo=UTC)
    expected = out.loc[out.index < decision, "wind_total_actual_mw"].iloc[-1]
    value = out.loc[decision + timedelta(hours=5), "wind_actual_decision_lag1"]
    assert value == pytest.approx(expected)


def test_decision_features_are_constant_within_delivery_date():
    df = make_frame("2020-01-01", 72)
    df["delivery_date"] = date(2020, 1, 3)
    out = add_lag_features(add_derived_columns(df))
    group = out[out["delivery_date"] == date(2020, 1, 3)]
    assert group["wind_actual_decision_lag1"].nunique(dropna=True) == 1
    assert group["wind_actual_decision_roll24_mean"].nunique(dropna=True) == 1


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def test_chronological_split_boundaries():
    df = make_frame("2022-12-31", 24)
    extra = make_frame("2023-06-01", 24)
    extra2 = make_frame("2025-06-01", 24)
    df = pd.concat([df, extra, extra2])
    out = make_chronological_split(df)
    assert set(out.loc[out.index < "2023-01-01", "split"]) == {"train"}
    assert set(
        out.loc[
            (out.index >= "2023-01-01") & (out.index < "2025-01-01"), "split"
        ]
    ) == {"validation"}
    assert set(out.loc[out.index >= "2025-01-01", "split"]) == {"test"}


def test_split_preserves_chronological_order():
    df = load_snapshot(SNAPSHOT)
    out = make_chronological_split(add_derived_columns(df))
    assert out.index.is_monotonic_increasing
    # Every train row precedes every validation row, etc.
    assert out.index[out["split"] == "train"].max() < out.index[
        out["split"] == "validation"
    ].min()
    assert out.index[out["split"] == "validation"].max() < out.index[
        out["split"] == "test"
    ].min()