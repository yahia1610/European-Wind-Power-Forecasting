"""Validation helpers for Open-Meteo weather data.

These functions are intentionally framework-light: they operate on the parsed
JSON payload (raw) and on lists of per-hour records (normalized). They never
interpolate or repair data — they only report what is present or missing, so
problems stay visible.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

REQUIRED_VARIABLES: tuple[str, ...] = (
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
)

# Units documented by Open-Meteo for these variables.
EXPECTED_UNITS: dict[str, str] = {
    "wind_speed_100m": "km/h",
    "wind_direction_100m": "°",
    "temperature_2m": "°C",
    "surface_pressure": "hPa",
}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value)


def validate_raw_response(
    payload: Mapping[str, Any],
    *,
    expected_variables: Sequence[str] = REQUIRED_VARIABLES,
    expected_run: str | None = None,
) -> dict[str, Any]:
    """Validate a raw Single Runs API JSON payload.

    Returns a report dict with ``ok``, ``errors``, ``warnings``, ``missing_counts``,
    ``units``, ``n_timesteps``, ``duplicate_timestamps`` and ``expected_run``.
    """

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "warnings": warnings,
        "expected_run": expected_run,
    }

    if not isinstance(payload, Mapping):
        errors.append("payload is not a JSON object")
        return report

    if payload.get("error"):
        errors.append(f"API reported an error: {payload.get('reason')}")
        return report

    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        errors.append("missing 'hourly' object")
        return report

    times = _as_list(hourly.get("time"))
    if not times:
        errors.append("missing or empty 'hourly.time'")
        return report

    units = payload.get("hourly_units") or {}
    report["units"] = dict(units)
    report["n_timesteps"] = int(len(times))

    # Duplicate timestamps.
    duplicates = [t for t, c in Counter(times).items() if c > 1]
    report["duplicate_timestamps"] = duplicates
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate timestamp(s) in response")

    # Variable presence, length, and units.
    missing_counts: dict[str, int] = {}
    for var in expected_variables:
        if var not in hourly:
            errors.append(f"missing required variable: {var}")
            missing_counts[var] = len(times)
            continue
        series = _as_list(hourly.get(var))
        if len(series) != len(times):
            errors.append(
                f"variable {var} length {len(series)} != time length {len(times)}"
            )
        missing_counts[var] = sum(1 for v in series if v is None)
        if var not in units:
            warnings.append(f"no unit reported for variable: {var}")
        else:
            expected = EXPECTED_UNITS.get(var)
            if expected is not None and units.get(var) != expected:
                warnings.append(
                    f"unit mismatch for {var}: expected {expected!r}, "
                    f"got {units.get(var)!r}"
                )

    report["missing_counts"] = missing_counts
    report["ok"] = not errors
    return report


def validate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_variables: Sequence[str] = REQUIRED_VARIABLES,
) -> dict[str, Any]:
    """Validate normalized per-hour records.

    Checks required variables, non-negative lead times, duplicate valid times
    (per init/location), and counts missing values.
    """

    errors: list[str] = []
    warnings: list[str] = []
    records = list(records)
    report: dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "warnings": warnings,
        "n_records": len(records),
    }

    if not records:
        errors.append("no records to validate")
        return report

    missing_counts = {var: 0 for var in expected_variables}
    lead_values: list[float] = []
    keys: Counter = Counter()

    for rec in records:
        for var in expected_variables:
            if var not in rec:
                errors.append(f"record missing field: {var}")
            elif rec.get(var) is None:
                missing_counts[var] += 1

        lead = rec.get("lead_hours")
        if lead is None:
            errors.append("record missing 'lead_hours'")
        else:
            lead_values.append(float(lead))
            if lead < 0:
                errors.append(f"negative lead_hours: {lead}")

        key = (
            rec.get("init_time_utc"),
            rec.get("latitude"),
            rec.get("longitude"),
            rec.get("valid_time_utc"),
        )
        keys[key] += 1

    duplicates = {k: c for k, c in keys.items() if c > 1}
    report["duplicate_record_keys"] = len(duplicates)
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate (init, location, valid_time) keys")

    report["missing_counts"] = missing_counts
    if lead_values:
        report["lead_hours_min"] = min(lead_values)
        report["lead_hours_max"] = max(lead_values)
        report["lead_hours_mean"] = sum(lead_values) / len(lead_values)

    report["ok"] = not errors
    return report


def summarize_dataframe(df: Any, *, variables: Sequence[str] = REQUIRED_VARIABLES) -> dict[str, Any]:
    """Summarize a normalized weather DataFrame (requires pandas)."""

    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]) if hasattr(df, "shape") else None,
        "duplicate_valid_times": int(df.duplicated().sum()) if len(df) else 0,
    }

    if "valid_time_utc" in df.columns and len(df):
        summary["first_valid_time_utc"] = str(df["valid_time_utc"].min())
        summary["last_valid_time_utc"] = str(df["valid_time_utc"].max())

    if "delivery_date" in df.columns:
        summary["delivery_dates"] = sorted(str(x) for x in df["delivery_date"].unique())

    if "location_label" in df.columns:
        summary["locations"] = sorted(str(x) for x in df["location_label"].unique())

    if "init_time_utc" in df.columns:
        summary["init_times_utc"] = sorted(str(x) for x in df["init_time_utc"].unique())

    if "lead_hours" in df.columns and len(df):
        summary["lead_hours_min"] = float(df["lead_hours"].min())
        summary["lead_hours_max"] = float(df["lead_hours"].max())
        summary["lead_hours_mean"] = float(df["lead_hours"].mean())

    summary["missing_counts"] = {
        var: int(df[var].isna().sum()) for var in variables if var in df.columns
    }

    # Duplicate (location, valid_time) pairs are a real integrity issue.
    if {"location_label", "valid_time_utc"}.issubset(df.columns):
        summary["duplicate_location_valid_time"] = int(
            df.duplicated(subset=["location_label", "valid_time_utc"]).sum()
        )

    return summary