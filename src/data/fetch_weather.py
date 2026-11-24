"""Orchestrate the 7-day ECMWF IFS HRES weather pilot.

This module implements the forecast-decision-time logic and the collection
pipeline. It deliberately does **not** interpolate or repair data.

Decision time
-------------
The intended forecast decision time is 18:00 Europe/Berlin on the day before
the delivery date. In UTC this is 17:00 (CET) or 16:00 (CEST).

Run selection
-------------
ECMWF IFS HRES runs at 00, 06, 12 and 18 UTC. A run initialised at ``T`` is not
necessarily published at ``T``; global model output typically becomes available
~4-6 hours later. Because Open-Meteo does not expose *historical* per-run
publication timestamps, we use a **conservative nominal delay** (default 6 h) and
select the latest run whose *estimated* availability is at or before the decision
cutoff. This is a documented methodological assumption, not verified historical
availability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from zoneinfo import ZoneInfo

from . import weather_validation
from .open_meteo_client import (
    DEFAULT_HOURLY_VARIABLES,
    DEFAULT_MODEL,
    RUN_CYCLE_HOURS_UTC,
    OpenMeteoClient,
    format_run,
    parse_single_run_response,
)
from .weather_locations import WEATHER_LOCATIONS, WeatherLocation, as_dicts

UTC = timezone.utc
BERLIN = ZoneInfo("Europe/Berlin")

DECISION_LOCAL_HOUR = 18
DEFAULT_NOMINAL_DELAY_HOURS = 6.0

DEFAULT_PILOT_START = date(2026, 8, 1)
DEFAULT_PILOT_END = date(2026, 8, 7)

PROCESSED_COLUMNS = [
    "delivery_date",
    "location_label",
    "classification",
    "latitude",
    "longitude",
    "model",
    "init_time_utc",
    "valid_time_utc",
    "lead_hours",
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
]


# --------------------------------------------------------------------------- #
# Time / run-selection logic
# --------------------------------------------------------------------------- #
def decision_time_utc(
    delivery_date: date,
    *,
    tz: ZoneInfo = BERLIN,
    decision_local_hour: int = DECISION_LOCAL_HOUR,
) -> datetime:
    """Return the decision time (18:00 local on the preceding day) in UTC."""

    prev = delivery_date - timedelta(days=1)
    decision_local = datetime(
        prev.year, prev.month, prev.day, decision_local_hour, 0, tzinfo=tz
    )
    return decision_local.astimezone(UTC)


def delivery_window_utc(
    delivery_date: date, *, tz: ZoneInfo = BERLIN
) -> tuple[datetime, datetime]:
    """Return the [start, end] UTC timestamps of a local delivery day (24 h)."""

    start_local = datetime(
        delivery_date.year, delivery_date.month, delivery_date.day, 0, 0, tzinfo=tz
    )
    start_utc = start_local.astimezone(UTC)
    end_utc = start_utc + timedelta(hours=24) - timedelta(hours=1)
    return start_utc, end_utc


def candidate_run_inits(
    decision_utc: datetime,
    *,
    cycle_hours: Sequence[int] = RUN_CYCLE_HOURS_UTC,
    lookback_days: int = 3,
) -> list[datetime]:
    """List candidate model initialisation times at or before the decision time."""

    decision_utc = decision_utc.astimezone(UTC)
    inits: list[datetime] = []
    for back in range(lookback_days + 1):
        day = (decision_utc.date() - timedelta(days=back))
        for hour in cycle_hours:
            init = datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC)
            if init <= decision_utc:
                inits.append(init)
    return sorted(set(inits))


@dataclass(frozen=True)
class RunSelection:
    """The chosen model run for one delivery date."""

    decision_time_utc: datetime
    selected_init_utc: datetime
    estimated_available_utc: datetime
    nominal_delay_hours: float
    verified_historical_availability: bool
    candidates_considered: tuple[datetime, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_time_utc": self.decision_time_utc.isoformat(),
            "selected_init_utc": self.selected_init_utc.isoformat(),
            "selected_init_utc_api": format_run(self.selected_init_utc),
            "estimated_available_utc": self.estimated_available_utc.isoformat(),
            "nominal_delay_hours": self.nominal_delay_hours,
            "verified_historical_availability": self.verified_historical_availability,
            "candidates_considered": [c.isoformat() for c in self.candidates_considered],
            "reason": self.reason,
        }


def select_run_init_time(
    decision_utc: datetime,
    *,
    nominal_delay_hours: float = DEFAULT_NOMINAL_DELAY_HOURS,
    cycle_hours: Sequence[int] = RUN_CYCLE_HOURS_UTC,
    verified_availability: Mapping[datetime, datetime] | None = None,
) -> RunSelection:
    """Select the latest run whose availability is at or before the cutoff.

    ``verified_availability`` optionally maps an init time to a *known* actual
    availability time. When a run is not in that mapping, its availability is
    estimated as ``init + nominal_delay_hours``.
    """

    decision_utc = decision_utc.astimezone(UTC)
    verified_availability = dict(verified_availability or {})
    candidates = candidate_run_inits(decision_utc, cycle_hours=cycle_hours)

    available: list[tuple[datetime, datetime]] = []
    for init in candidates:
        if init in verified_availability:
            avail = verified_availability[init]
            is_available = avail <= decision_utc
        else:
            avail = init + timedelta(hours=nominal_delay_hours)
            is_available = avail <= decision_utc
        if is_available:
            available.append((init, avail))

    if not available:
        raise RuntimeError(
            f"No model run available before decision time {decision_utc.isoformat()}"
        )

    selected_init, selected_avail = max(available, key=lambda pair: pair[0])
    verified = selected_init in verified_availability

    if verified:
        reason = (
            "selected the latest run with a verified availability time at or before "
            "the decision cutoff"
        )
    else:
        reason = (
            f"historical per-run publication times are not exposed by Open-Meteo; "
            f"selected the latest run whose estimated availability "
            f"(init + {nominal_delay_hours:g} h) is at or before the decision cutoff. "
            f"This is a conservative methodological assumption, not verified "
            f"historical availability."
        )

    return RunSelection(
        decision_time_utc=decision_utc,
        selected_init_utc=selected_init,
        estimated_available_utc=selected_avail,
        nominal_delay_hours=nominal_delay_hours,
        verified_historical_availability=verified,
        candidates_considered=tuple(candidates),
        reason=reason,
    )


def forecast_hours_needed(
    init_utc: datetime, start_utc: datetime, end_utc: datetime
) -> int:
    """Number of hours from init needed to cover [start, end] (inclusive)."""

    last_lead = (end_utc.astimezone(UTC) - init_utc.astimezone(UTC)).total_seconds() / 3600.0
    return max(1, int(last_lead) + 1)


# --------------------------------------------------------------------------- #
# DataFrame helpers
# --------------------------------------------------------------------------- #
def records_to_dataframe(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert normalized records into a typed, ordered DataFrame."""

    df = pd.DataFrame(list(records))
    if df.empty:
        return pd.DataFrame(columns=PROCESSED_COLUMNS)

    for col in ("init_time_utc", "valid_time_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    ordered = [c for c in PROCESSED_COLUMNS if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _raw_filename(model: str, delivery_date: date, init_utc: datetime, label: str) -> str:
    return (
        f"{model}__delivery{delivery_date.isoformat()}__"
        f"run{init_utc.strftime('%Y%m%dT%H%M')}__{label}.json"
    )


def verify_runs(
    run_inits: Sequence[datetime],
    *,
    client: OpenMeteoClient,
    model: str,
    reference_location: WeatherLocation,
) -> dict[str, dict[str, Any]]:
    """Verify that each requested model run exists (one tiny request per run)."""

    status: dict[str, dict[str, Any]] = {}
    for init in sorted(set(run_inits)):
        exists, detail = client.run_exists(
            init,
            model=model,
            latitude=reference_location.latitude,
            longitude=reference_location.longitude,
        )
        detail = dict(detail)
        detail["checked_at_utc"] = datetime.now(UTC).isoformat()
        status[format_run(init)] = detail
    return status


def collect_pilot(
    delivery_dates: Sequence[date],
    *,
    locations: Sequence[WeatherLocation] = WEATHER_LOCATIONS,
    client: OpenMeteoClient | None = None,
    raw_dir: str | Path = "data/raw/weather",
    processed_dir: str | Path = "data/processed/weather_test",
    model: str = DEFAULT_MODEL,
    hourly: Sequence[str] = DEFAULT_HOURLY_VARIABLES,
    nominal_delay_hours: float = DEFAULT_NOMINAL_DELAY_HOURS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the pilot collection and persist raw + normalized outputs.

    Returns ``(manifest, dataframe)``.
    """

    client = client or OpenMeteoClient()
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1) Decide the run for every delivery date.
    selections: dict[date, RunSelection] = {}
    run_rows: list[dict[str, Any]] = []
    for dd in delivery_dates:
        decision = decision_time_utc(dd)
        selection = select_run_init_time(
            decision, nominal_delay_hours=nominal_delay_hours
        )
        selections[dd] = selection
        start, end = delivery_window_utc(dd)
        run_rows.append(
            {
                "delivery_date": dd.isoformat(),
                "decision_time_utc": selection.decision_time_utc.isoformat(),
                "selected_init_utc": selection.selected_init_utc.isoformat(),
                "selected_init_utc_api": format_run(selection.selected_init_utc),
                "estimated_available_utc": selection.estimated_available_utc.isoformat(),
                "nominal_delay_hours": selection.nominal_delay_hours,
                "verified_historical_availability": (
                    selection.verified_historical_availability
                ),
                "forecast_hours_requested": forecast_hours_needed(
                    selection.selected_init_utc, start, end
                ),
                "delivery_window_start_utc": start.isoformat(),
                "delivery_window_end_utc": end.isoformat(),
                "candidates_considered": [
                    c.isoformat() for c in selection.candidates_considered
                ],
                "reason": selection.reason,
            }
        )

    # 2) Verify every requested run exists before doing the full collection.
    unique_inits = sorted({sel.selected_init_utc for sel in selections.values()})
    run_status = verify_runs(
        unique_inits,
        client=client,
        model=model,
        reference_location=locations[0],
    )
    missing = [run for run, info in run_status.items() if not info.get("exists")]
    if missing:
        raise RuntimeError(
            "The following requested model runs are not available: "
            + ", ".join(missing)
        )

    # 3) Collect raw + normalized data.
    records: list[dict[str, Any]] = []
    raw_files: list[str] = []
    for dd in delivery_dates:
        selection = selections[dd]
        start, end = delivery_window_utc(dd)
        fh = forecast_hours_needed(selection.selected_init_utc, start, end)

        for loc in locations:
            raw = client.get_single_run_raw(
                loc.latitude,
                loc.longitude,
                selection.selected_init_utc,
                hourly=hourly,
                model=model,
                forecast_hours=fh,
            )

            fname = _raw_filename(
                model, dd, selection.selected_init_utc, loc.label
            )
            (raw_path / fname).write_text(
                json.dumps(raw, ensure_ascii=False), encoding="utf-8"
            )
            raw_files.append(fname)

            parsed = parse_single_run_response(
                raw,
                run_init=selection.selected_init_utc,
                model=model,
                latitude=loc.latitude,
                longitude=loc.longitude,
            )
            for rec in parsed["records"]:
                if start <= rec["valid_time_utc"] <= end:
                    rec["delivery_date"] = dd
                    rec["location_label"] = loc.label
                    rec["classification"] = loc.classification
                    records.append(rec)

    df = records_to_dataframe(records)
    parquet_file = processed_path / "weather_pilot.parquet"
    csv_file = processed_path / "weather_pilot.csv"
    df.to_parquet(parquet_file, index=False)
    df.to_csv(csv_file, index=False)

    manifest: dict[str, Any] = {
        "api": {
            "name": "Open-Meteo Single Runs API",
            "endpoint": "https://single-runs-api.open-meteo.com/v1/forecast",
            "documentation": "https://open-meteo.com/en/docs/single-runs-api",
        },
        "model": model,
        "model_name": "ECMWF IFS HRES 9km",
        "hourly_variables": list(hourly),
        "nominal_delay_hours_assumption": nominal_delay_hours,
        "decision_time_rule": (
            f"{DECISION_LOCAL_HOUR}:00 Europe/Berlin on the day before delivery "
            "(17:00 UTC CET / 16:00 UTC CEST)"
        ),
        "availability_note": (
            "Open-Meteo does not expose historical per-run publication timestamps. "
            "The nominal-delay availability is a documented assumption, not verified "
            "historical availability."
        ),
        "delivery_dates": [dd.isoformat() for dd in delivery_dates],
        "locations": as_dicts(),
        "n_locations": len(locations),
        "runs": run_rows,
        "run_existence_check": run_status,
        "raw_files": raw_files,
        "raw_dir": str(raw_path),
        "processed_files": [parquet_file.name, csv_file.name],
        "n_records": int(len(df)),
        "validation": weather_validation.summarize_dataframe(df),
        "license": "CC BY 4.0",
        "attribution": "Weather data: Open-Meteo.com (CC BY 4.0), ECMWF IFS HRES 9km.",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    (processed_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_delivery_dates(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end``."""

    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect the 7-day ECMWF IFS HRES weather pilot."
    )
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_PILOT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_PILOT_END)
    parser.add_argument("--raw-dir", default="data/raw/weather")
    parser.add_argument("--processed-dir", default="data/processed/weather_test")
    parser.add_argument("--nominal-delay-hours", type=float, default=DEFAULT_NOMINAL_DELAY_HOURS)
    args = parser.parse_args(argv)

    dates = parse_delivery_dates(args.start, args.end)
    print(f"Delivery dates: {dates[0]} .. {dates[-1]} ({len(dates)} days)")
    for dd in dates:
        decision = decision_time_utc(dd)
        sel = select_run_init_time(
            decision, nominal_delay_hours=args.nominal_delay_hours
        )
        print(
            f"  {dd} | decision {decision.isoformat()} | "
            f"init {sel.selected_init_utc.isoformat()} "
            f"(verified={sel.verified_historical_availability})"
        )

    manifest, df = collect_pilot(
        dates,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        nominal_delay_hours=args.nominal_delay_hours,
    )

    print("\nCollection summary:")
    print(f"  locations: {manifest['n_locations']}")
    print(f"  records:   {manifest['n_records']}")
    print(f"  raw files: {len(manifest['raw_files'])}")
    print(f"  missing:   {manifest['validation']['missing_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())