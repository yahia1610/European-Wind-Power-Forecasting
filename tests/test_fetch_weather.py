"""Unit tests for forecast-decision-time and run-selection logic."""

from __future__ import annotations

from datetime import date, datetime, timezone

from src.data.fetch_weather import (
    decision_time_utc,
    delivery_window_utc,
    forecast_hours_needed,
    records_to_dataframe,
    select_run_init_time,
)

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Decision time / delivery window (CET and CEST)
# --------------------------------------------------------------------------- #
def test_decision_time_utc_summer_is_16utc():
    # 18:00 CEST (UTC+2) on the preceding day -> 16:00 UTC
    assert decision_time_utc(date(2026, 8, 1)) == datetime(
        2026, 7, 31, 16, 0, tzinfo=UTC
    )


def test_decision_time_utc_winter_is_17utc():
    # 18:00 CET (UTC+1) on the preceding day -> 17:00 UTC
    assert decision_time_utc(date(2027, 1, 15)) == datetime(
        2027, 1, 14, 17, 0, tzinfo=UTC
    )


def test_delivery_window_summer():
    start, end = delivery_window_utc(date(2026, 8, 1))
    assert start == datetime(2026, 7, 31, 22, 0, tzinfo=UTC)
    assert end == datetime(2026, 8, 1, 21, 0, tzinfo=UTC)


def test_delivery_window_winter():
    start, end = delivery_window_utc(date(2027, 1, 15))
    assert start == datetime(2027, 1, 14, 23, 0, tzinfo=UTC)
    assert end == datetime(2027, 1, 15, 22, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Run selection
# --------------------------------------------------------------------------- #
def test_select_run_is_conservative_06utc_run():
    decision = datetime(2026, 7, 31, 16, 0, tzinfo=UTC)
    selection = select_run_init_time(decision, nominal_delay_hours=6.0)
    # 12 UTC run would be available ~18 UTC (after 16 UTC cutoff) -> not used.
    assert selection.selected_init_utc == datetime(2026, 7, 31, 6, 0, tzinfo=UTC)
    assert selection.estimated_available_utc == datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    assert selection.verified_historical_availability is False


def test_select_run_uses_verified_availability_when_provided():
    decision = datetime(2026, 7, 31, 16, 0, tzinfo=UTC)
    verified = {
        datetime(2026, 7, 31, 12, 0, tzinfo=UTC): datetime(
            2026, 7, 31, 15, 0, tzinfo=UTC
        )
    }
    selection = select_run_init_time(
        decision, nominal_delay_hours=6.0, verified_availability=verified
    )
    # With a verified 15:00 availability, the 12 UTC run becomes usable.
    assert selection.selected_init_utc == datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    assert selection.verified_historical_availability is True


def test_select_run_rejects_unavailable_12utc_run():
    decision = datetime(2026, 7, 31, 16, 0, tzinfo=UTC)
    verified = {
        datetime(2026, 7, 31, 12, 0, tzinfo=UTC): datetime(
            2026, 7, 31, 18, 0, tzinfo=UTC
        )
    }
    selection = select_run_init_time(
        decision, nominal_delay_hours=6.0, verified_availability=verified
    )
    assert selection.selected_init_utc == datetime(2026, 7, 31, 6, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Forecast hours / dataframe
# --------------------------------------------------------------------------- #
def test_forecast_hours_needed():
    init = datetime(2026, 7, 31, 6, 0, tzinfo=UTC)
    start = datetime(2026, 7, 31, 22, 0, tzinfo=UTC)
    end = datetime(2026, 8, 1, 21, 0, tzinfo=UTC)
    # end - init = 39 h, inclusive -> 40 timesteps from init (0..39)
    assert forecast_hours_needed(init, start, end) == 40


def test_records_to_dataframe_orders_columns_and_parses_utc():
    records = [
        {
            "valid_time_utc": datetime(2026, 7, 31, 22, 0, tzinfo=UTC),
            "init_time_utc": datetime(2026, 7, 31, 6, 0, tzinfo=UTC),
            "lead_hours": 16.0,
            "model": "ecmwf_ifs",
            "latitude": 54.3,
            "longitude": 10.1,
            "delivery_date": date(2026, 8, 1),
            "location_label": "north_germany",
            "classification": "land",
            "wind_speed_100m": 12.0,
            "wind_direction_100m": 180,
            "temperature_2m": 15.0,
            "surface_pressure": 1010.0,
        }
    ]
    df = records_to_dataframe(records)
    assert list(df.columns)[:4] == [
        "delivery_date",
        "location_label",
        "classification",
        "latitude",
    ]
    assert str(df["valid_time_utc"].dt.tz) == "UTC"