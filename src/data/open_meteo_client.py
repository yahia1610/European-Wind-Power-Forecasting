"""Minimal, dependency-light client for the Open-Meteo **Single Runs API**.

Documentation: https://open-meteo.com/en/docs/single-runs-api

The Single Runs API returns the full forecast horizon of one specific model run,
identified by its UTC initialisation time via the ``run`` parameter. This is the
correct API for historical, leakage-free backtesting because we can request the
exact run that would have been available at forecast decision time.

Key documented facts (verified against the live API during Phase 2):
- Base URL: ``https://single-runs-api.open-meteo.com/v1/forecast``
- ``run`` is required, ISO-8601 without seconds, e.g. ``2026-07-31T06:00``.
- ECMWF IFS HRES 9 km is requested with ``models=ecmwf_ifs``.
- ``forecast_hours`` limits the response to N hours from the run's init time.
- ``start_date`` / ``end_date`` are *not* accepted by this endpoint.
- A missing/invalid run returns HTTP 400 with ``{"error": true, "reason": ...}``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import requests

SINGLE_RUNS_BASE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

DEFAULT_MODEL = "ecmwf_ifs"
DEFAULT_HOURLY_VARIABLES: tuple[str, ...] = (
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
)
# Valid ECMWF IFS HRES initialisation cycles (UTC).
RUN_CYCLE_HOURS_UTC: tuple[int, ...] = (0, 6, 12, 18)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class OpenMeteoError(RuntimeError):
    """Base class for all Open-Meteo client errors."""


class OpenMeteoRequestError(OpenMeteoError):
    """Raised for network failures or non-recoverable HTTP error statuses."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body


class OpenMeteoRunUnavailableError(OpenMeteoRequestError):
    """Raised when the API reports that the requested model run is not available."""


class OpenMeteoParseError(OpenMeteoError):
    """Raised when a successful response cannot be parsed as expected."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes are assumed to already be UTC (the API returns UTC ISO
    strings when ``timezone=UTC``).
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_run(run: datetime) -> str:
    """Format a run initialisation datetime as the API expects (UTC, no seconds)."""

    return ensure_utc(run).strftime("%Y-%m-%dT%H:%M")


def _parse_api_datetime(text: str) -> datetime:
    """Parse an API timestamp into an aware UTC datetime."""

    return ensure_utc(datetime.fromisoformat(text))


def build_single_run_params(
    latitude: float,
    longitude: float,
    run: datetime,
    *,
    hourly: Sequence[str] | str = DEFAULT_HOURLY_VARIABLES,
    model: str = DEFAULT_MODEL,
    forecast_hours: int | None = None,
    timezone_name: str = "UTC",
    **extra: Any,
) -> dict[str, Any]:
    """Build the query parameters for a Single Runs API request.

    This mirrors the documented request format and is kept separate from the HTTP
    layer so it can be unit-tested without network access.
    """

    if isinstance(hourly, str):
        hourly_param = hourly
    else:
        hourly_param = ",".join(hourly)

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": hourly_param,
        "models": model,
        "timezone": timezone_name,
        "run": format_run(run),
    }
    if forecast_hours is not None:
        params["forecast_hours"] = int(forecast_hours)
    params.update(extra)
    return params


def _safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _is_run_unavailable(body: Any) -> bool:
    if isinstance(body, Mapping):
        reason = str(body.get("reason", "")).lower()
        if "not available" in reason or "not be available" in reason:
            return True
    return False


def parse_single_run_response(
    payload: Mapping[str, Any],
    *,
    run_init: datetime,
    model: str = DEFAULT_MODEL,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Parse a raw Single Runs API JSON payload into tidy per-hour records.

    Returns a dict with:
    - ``records``: list of per-hour dicts (valid time, init time, lead hours,
      model, coordinates, and one entry per weather variable).
    - ``units``: the ``hourly_units`` mapping returned by the API.
    - ``variables``: the weather variables present in the response.
    - ``latitude`` / ``longitude`` / ``elevation``: grid metadata.
    """

    if not isinstance(payload, Mapping):
        raise OpenMeteoParseError("response payload is not a JSON object")
    if payload.get("error"):
        raise OpenMeteoParseError(f"API returned an error: {payload.get('reason')}")

    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise OpenMeteoParseError("response is missing the 'hourly' object")

    times = hourly.get("time")
    if not times:
        raise OpenMeteoParseError("response is missing 'hourly.time'")

    variables = [key for key in hourly.keys() if key != "time"]
    units = payload.get("hourly_units") or {}
    init_utc = ensure_utc(run_init)

    records: list[dict[str, Any]] = []
    for i, raw_time in enumerate(times):
        valid_utc = _parse_api_datetime(str(raw_time))
        lead_hours = (valid_utc - init_utc).total_seconds() / 3600.0
        record: dict[str, Any] = {
            "valid_time_utc": valid_utc,
            "init_time_utc": init_utc,
            "lead_hours": lead_hours,
            "model": model,
            "latitude": latitude,
            "longitude": longitude,
        }
        for var in variables:
            series = hourly.get(var) or []
            record[var] = series[i] if i < len(series) else None
        records.append(record)

    return {
        "records": records,
        "units": dict(units),
        "variables": variables,
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "elevation": payload.get("elevation"),
    }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class OpenMeteoClient:
    """Thin, retrying HTTP client for the Open-Meteo Single Runs API.

    Parameters
    ----------
    base_url:
        API endpoint. Override only for testing.
    timeout:
        Per-request timeout in seconds.
    retries:
        Number of *additional* attempts after the first failure.
    backoff:
        Base backoff factor in seconds; delay = ``backoff * 2**attempt``.
    session:
        Optional ``requests.Session``-compatible object (mainly for tests).
    sleep_fn:
        Injectable sleep function (mainly for tests).
    """

    def __init__(
        self,
        base_url: str = SINGLE_RUNS_BASE_URL,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.5,
        session: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = session if session is not None else requests.Session()
        self.sleep_fn = sleep_fn

    def get_single_run_raw(
        self,
        latitude: float,
        longitude: float,
        run: datetime,
        *,
        hourly: Sequence[str] | str = DEFAULT_HOURLY_VARIABLES,
        model: str = DEFAULT_MODEL,
        forecast_hours: int | None = None,
        timezone_name: str = "UTC",
    ) -> dict[str, Any]:
        """Perform one Single Runs API request and return the parsed JSON dict."""

        params = build_single_run_params(
            latitude,
            longitude,
            run,
            hourly=hourly,
            model=model,
            forecast_hours=forecast_hours,
            timezone_name=timezone_name,
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    self.base_url, params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:  # network-level failure
                last_error = exc
                if attempt < self.retries:
                    self.sleep_fn(self.backoff * (2**attempt))
                    continue
                raise OpenMeteoRequestError(
                    f"network error contacting Open-Meteo: {exc}", url=self.base_url
                ) from exc

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise OpenMeteoParseError(
                        "200 response was not valid JSON"
                    ) from exc

            body = _safe_json(response)

            if status in _RETRYABLE_STATUS and attempt < self.retries:
                self.sleep_fn(self.backoff * (2**attempt))
                continue

            if status == 400 and _is_run_unavailable(body):
                raise OpenMeteoRunUnavailableError(
                    f"requested model run is not available: run={format_run(run)}",
                    status_code=status,
                    url=self.base_url,
                    body=response.text,
                )

            raise OpenMeteoRequestError(
                f"Open-Meteo request failed with HTTP {status}",
                status_code=status,
                url=self.base_url,
                body=response.text,
            )

        # Should be unreachable, but keep a defensive error.
        raise OpenMeteoRequestError(
            f"request failed after {self.retries + 1} attempts: {last_error}"
        )

    def get_single_run(
        self,
        latitude: float,
        longitude: float,
        run: datetime,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch and parse a single run into tidy records (convenience wrapper)."""

        raw = self.get_single_run_raw(latitude, longitude, run, **kwargs)
        model = kwargs.get("model", DEFAULT_MODEL)
        return parse_single_run_response(
            raw,
            run_init=run,
            model=model,
            latitude=latitude,
            longitude=longitude,
        )

    def run_exists(
        self,
        run: datetime,
        *,
        model: str = DEFAULT_MODEL,
        latitude: float,
        longitude: float,
    ) -> tuple[bool, dict[str, Any]]:
        """Check whether a model run is archived.

        Uses a minimal one-hour, single-variable request. Returns
        ``(exists, detail)`` and never raises for a missing run.
        """

        try:
            self.get_single_run_raw(
                latitude,
                longitude,
                run,
                hourly=(DEFAULT_HOURLY_VARIABLES[0],),
                model=model,
                forecast_hours=1,
            )
            return True, {"run": format_run(run), "exists": True}
        except OpenMeteoRunUnavailableError as exc:
            return False, {
                "run": format_run(run),
                "exists": False,
                "reason": str(exc),
            }