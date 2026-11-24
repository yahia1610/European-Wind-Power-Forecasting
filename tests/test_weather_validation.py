"""Unit tests for weather validation helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.weather_validation import (
    REQUIRED_VARIABLES,
    summarize_dataframe,
    validate_raw_response,
    validate_records,
)


def make_payload(times, values, units=None):
    return {
        "hourly_units": units
        or {
            "time": "iso8601",
            "wind_speed_100m": "km/h",
            "wind_direction_100m": "°",
            "temperature_2m": "°C",
            "surface_pressure": "hPa",
        },
        "hourly": {"time": list(times), **{k: list(v) for k, v in values.items()}},
    }


def test_validate_raw_response_ok():
    payload = make_payload(
        ["2026-07-31T06:00", "2026-07-31T07:00"],
        {
            "wind_speed_100m": [10.0, 11.0],
            "wind_direction_100m": [180, 200],
            "temperature_2m": [15.0, 15.5],
            "surface_pressure": [1010.0, 1011.0],
        },
    )
    report = validate_raw_response(payload)
    assert report["ok"] is True
    assert report["n_timesteps"] == 2
    assert report["missing_counts"]["wind_speed_100m"] == 0


def test_validate_raw_response_missing_variable_is_error():
    payload = make_payload(
        ["2026-07-31T06:00"],
        {"wind_speed_100m": [10.0]},
    )
    report = validate_raw_response(payload)
    assert report["ok"] is False
    assert any("missing required variable" in e for e in report["errors"])


def test_validate_raw_response_unit_mismatch_is_warning():
    units = {
        "time": "iso8601",
        "wind_speed_100m": "m/s",  # wrong, expected km/h
        "wind_direction_100m": "°",
        "temperature_2m": "°C",
        "surface_pressure": "hPa",
    }
    payload = make_payload(
        ["2026-07-31T06:00"],
        {
            "wind_speed_100m": [10.0],
            "wind_direction_100m": [180],
            "temperature_2m": [15.0],
            "surface_pressure": [1010.0],
        },
        units=units,
    )
    report = validate_raw_response(payload)
    assert any("unit mismatch" in w for w in report["warnings"])


def test_validate_raw_response_duplicate_timestamps_is_error():
    payload = make_payload(
        ["2026-07-31T06:00", "2026-07-31T06:00"],
        {
            "wind_speed_100m": [10.0, 11.0],
            "wind_direction_100m": [180, 200],
            "temperature_2m": [15.0, 15.5],
            "surface_pressure": [1010.0, 1011.0],
        },
    )
    report = validate_raw_response(payload)
    assert report["ok"] is False
    assert len(report["duplicate_timestamps"]) == 1


def test_validate_records_negative_lead_is_error():
    records = [
        {
            "init_time_utc": datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
            "valid_time_utc": datetime(2026, 7, 31, 5, 0, tzinfo=timezone.utc),
            "lead_hours": -1.0,
            "latitude": 54.3,
            "longitude": 10.1,
            "wind_speed_100m": 10.0,
            "wind_direction_100m": 180,
            "temperature_2m": 15.0,
            "surface_pressure": 1010.0,
        }
    ]
    report = validate_records(records)
    assert report["ok"] is False
    assert any("negative lead_hours" in e for e in report["errors"])


def test_validate_records_missing_values_counted():
    records = [
        {
            "init_time_utc": datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
            "valid_time_utc": datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
            "lead_hours": 0.0,
            "latitude": 54.3,
            "longitude": 10.1,
            "wind_speed_100m": None,
            "wind_direction_100m": 180,
            "temperature_2m": 15.0,
            "surface_pressure": 1010.0,
        }
    ]
    report = validate_records(records)
    assert report["missing_counts"]["wind_speed_100m"] == 1


def test_summarize_dataframe():
    df = pd.DataFrame(
        {
            "delivery_date": ["2026-08-01", "2026-08-01"],
            "location_label": ["a", "a"],
            "init_time_utc": pd.to_datetime(
                ["2026-07-31 06:00:00+00:00", "2026-07-31 06:00:00+00:00"], utc=True
            ),
            "valid_time_utc": pd.to_datetime(
                ["2026-07-31 22:00:00+00:00", "2026-07-31 23:00:00+00:00"], utc=True
            ),
            "lead_hours": [16.0, 17.0],
            "wind_speed_100m": [10.0, 12.0],
            "wind_direction_100m": [180, 200],
            "temperature_2m": [15.0, 15.5],
            "surface_pressure": [1010.0, 1011.0],
        }
    )
    summary = summarize_dataframe(df)
    assert summary["n_rows"] == 2
    assert summary["lead_hours_min"] == 16.0
    assert summary["lead_hours_max"] == 17.0
    assert summary["missing_counts"]["wind_speed_100m"] == 0
    assert summary["duplicate_location_valid_time"] == 0
    assert summary["locations"] == ["a"]