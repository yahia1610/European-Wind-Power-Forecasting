"""Unit tests for the Open-Meteo Single Runs API client (mocked HTTP)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import requests

from src.data.open_meteo_client import (
    DEFAULT_HOURLY_VARIABLES,
    OpenMeteoClient,
    OpenMeteoParseError,
    OpenMeteoRequestError,
    OpenMeteoRunUnavailableError,
    build_single_run_params,
    format_run,
    parse_single_run_response,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        if text is not None:
            self.text = text
        elif payload is not None:
            self.text = json.dumps(payload)
        else:
            self.text = ""

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class RaisingSession:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        raise self._exc


def make_payload(times, values, units=None):
    payload = {
        "latitude": 54.3,
        "longitude": 10.1,
        "elevation": 40.0,
        "hourly_units": units
        or {
            "time": "iso8601",
            "wind_speed_100m": "km/h",
            "wind_direction_100m": "°",
            "temperature_2m": "°C",
            "surface_pressure": "hPa",
        },
        "hourly": {"time": list(times)},
    }
    for key, series in values.items():
        payload["hourly"][key] = list(series)
    return payload


# --------------------------------------------------------------------------- #
# Parameter / formatting helpers
# --------------------------------------------------------------------------- #
def test_format_run_is_utc_without_seconds():
    run = datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc)
    assert format_run(run) == "2026-07-31T06:00"


def test_build_single_run_params_contains_documented_fields():
    params = build_single_run_params(
        54.3,
        10.1,
        datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
        hourly=DEFAULT_HOURLY_VARIABLES,
        model="ecmwf_ifs",
        forecast_hours=40,
    )
    assert params["latitude"] == 54.3
    assert params["longitude"] == 10.1
    assert params["models"] == "ecmwf_ifs"
    assert params["run"] == "2026-07-31T06:00"
    assert params["timezone"] == "UTC"
    assert params["forecast_hours"] == 40
    assert params["hourly"] == ",".join(DEFAULT_HOURLY_VARIABLES)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_single_run_response_computes_utc_and_lead_hours():
    times = ["2026-07-31T06:00", "2026-07-31T07:00", "2026-07-31T08:00"]
    payload = make_payload(
        times,
        {
            "wind_speed_100m": [10.0, 11.0, 12.0],
            "wind_direction_100m": [180, 200, 220],
            "temperature_2m": [15.0, 15.5, 16.0],
            "surface_pressure": [1010.0, 1011.0, 1012.0],
        },
    )
    parsed = parse_single_run_response(
        payload,
        run_init=datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
        model="ecmwf_ifs",
        latitude=54.3,
        longitude=10.1,
    )
    records = parsed["records"]
    assert len(records) == 3
    assert records[0]["valid_time_utc"] == datetime(
        2026, 7, 31, 6, 0, tzinfo=timezone.utc
    )
    assert records[0]["init_time_utc"] == datetime(
        2026, 7, 31, 6, 0, tzinfo=timezone.utc
    )
    assert records[0]["lead_hours"] == 0.0
    assert records[1]["lead_hours"] == 1.0
    assert records[2]["lead_hours"] == 2.0
    assert records[0]["wind_speed_100m"] == 10.0
    assert parsed["units"]["wind_speed_100m"] == "km/h"
    assert parsed["variables"] == [
        "wind_speed_100m",
        "wind_direction_100m",
        "temperature_2m",
        "surface_pressure",
    ]


def test_parse_single_run_response_handles_missing_values():
    times = ["2026-07-31T06:00", "2026-07-31T07:00"]
    payload = make_payload(
        times,
        {
            "wind_speed_100m": [10.0, None],
            "wind_direction_100m": [180, None],
            "temperature_2m": [15.0, None],
            "surface_pressure": [1010.0, None],
        },
    )
    parsed = parse_single_run_response(
        payload, run_init=datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc)
    )
    assert parsed["records"][1]["wind_speed_100m"] is None


def test_parse_single_run_response_rejects_error_payload():
    with pytest.raises(OpenMeteoParseError):
        parse_single_run_response(
            {"error": True, "reason": "boom"},
            run_init=datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
        )


# --------------------------------------------------------------------------- #
# HTTP behaviour
# --------------------------------------------------------------------------- #
def test_get_single_run_raw_returns_json():
    payload = make_payload(["2026-07-31T06:00"], {"wind_speed_100m": [10.0]})
    session = FakeSession([FakeResponse(200, payload)])
    client = OpenMeteoClient(session=session, sleep_fn=lambda _: None)
    result = client.get_single_run_raw(
        54.3, 10.1, datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc)
    )
    assert result["hourly"]["time"] == ["2026-07-31T06:00"]
    assert session.calls[0]["params"]["run"] == "2026-07-31T06:00"


def test_run_exists_true_on_200():
    payload = make_payload(["2026-07-31T06:00"], {"wind_speed_100m": [10.0]})
    session = FakeSession([FakeResponse(200, payload)])
    client = OpenMeteoClient(session=session, sleep_fn=lambda _: None)
    exists, detail = client.run_exists(
        datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
        latitude=54.3,
        longitude=10.1,
    )
    assert exists is True
    assert detail["exists"] is True


def test_run_exists_false_on_run_unavailable():
    body = {
        "error": True,
        "reason": (
            "The requested model run is not available. Model: ecmwf_ifs, "
            "run: 2020-01-01T00:00Z"
        ),
    }
    session = FakeSession([FakeResponse(400, body)])
    client = OpenMeteoClient(session=session, sleep_fn=lambda _: None)
    exists, detail = client.run_exists(
        datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
        latitude=54.3,
        longitude=10.1,
    )
    assert exists is False
    assert "not available" in detail["reason"].lower()


def test_retries_on_500_then_succeeds():
    payload = make_payload(["2026-07-31T06:00"], {"wind_speed_100m": [10.0]})
    session = FakeSession([FakeResponse(500), FakeResponse(200, payload)])
    client = OpenMeteoClient(session=session, retries=2, sleep_fn=lambda _: None)
    result = client.get_single_run_raw(
        54.3, 10.1, datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc)
    )
    assert result["hourly"]["time"] == ["2026-07-31T06:00"]
    assert len(session.calls) == 2


def test_non_run_400_raises_request_error():
    session = FakeSession([FakeResponse(400, {"error": True, "reason": "bad param"})])
    client = OpenMeteoClient(session=session, sleep_fn=lambda _: None)
    with pytest.raises(OpenMeteoRequestError):
        client.get_single_run_raw(
            54.3, 10.1, datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc)
        )


def test_network_error_raises_after_retries():
    session = RaisingSession(requests.ConnectionError("no network"))
    client = OpenMeteoClient(session=session, retries=2, sleep_fn=lambda _: None)
    with pytest.raises(OpenMeteoRequestError):
        client.get_single_run_raw(
            54.3, 10.1, datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc)
        )
    assert session.calls == 3  # 1 initial + 2 retries