"""Data preparation for the DE-LU hourly electricity snapshot.

This module loads the immutable raw snapshot, validates its time index, derives
the modelling target and benchmark columns, adds strictly-historical lag
features, and writes a cleaned dataset plus a JSON quality report.

Design principles
-----------------
- The raw Parquet is **never modified**.
- All timestamps are handled in **UTC**.
- Missing target values are **preserved as NaN** (no arbitrary imputation).
- Negative day-ahead prices are **kept** (they are valid market observations).
- Lag / rolling features are **masked against the forecast decision time** so no
  future information leaks into historical features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UTC = timezone.utc
BERLIN = ZoneInfo("Europe/Berlin")

# Decision time: 18:00 Europe/Berlin on the day before delivery.
DECISION_LOCAL_HOUR = 18

# Price aggregation change documented by the upstream source manifest:
# hourly product before this instant, mean of four quarter-hour prices after.
PRICE_METHODOLOGY_CHANGE_UTC = pd.Timestamp("2025-10-01 00:00:00", tz=UTC)

# Chronological split boundaries by *delivery date* (start-inclusive,
# end-exclusive). Splitting on delivery_date keeps every delivery day intact.
SPLIT_BOUNDARIES = {
    "train": (date(2019, 1, 1), date(2023, 1, 1)),
    "validation": (date(2023, 1, 1), date(2025, 1, 1)),
    "test": (date(2025, 1, 1), date(2100, 1, 1)),
}

REQUIRED_SOURCE_COLUMNS = [
    "timestamp_utc",
    "delivery_date",
    "price_eur_mwh",
    "load_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "wind_onshore_actual_mw",
    "wind_offshore_actual_mw",
]


class DataValidationError(ValueError):
    """Raised when the raw snapshot fails a structural integrity check."""


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def sha256_of_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file (for provenance)."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Load the raw snapshot and set a UTC ``timestamp_utc`` index.

    The file is opened read-only; it is never written back.
    """

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"snapshot missing required columns: {missing}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    df.index.name = "timestamp_utc"

    # Convert delivery_date to a real date for downstream use.
    df["delivery_date"] = pd.to_datetime(df["delivery_date"]).dt.date
    return df


def validate_hourly_index(
    df: pd.DataFrame, *, name: str = "snapshot"
) -> dict[str, Any]:
    """Validate that the index is unique, monotonic, hourly and gap-free."""

    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(f"{name}: index is not a DatetimeIndex")
    if index.tz is None:
        raise DataValidationError(f"{name}: index is not timezone-aware (expected UTC)")
    if str(index.tz) != "UTC":
        raise DataValidationError(f"{name}: index timezone is {index.tz}, expected UTC")

    duplicates = int(index.duplicated().sum())
    monotonic = bool(index.is_monotonic_increasing)

    expected = pd.date_range(index.min(), index.max(), freq="h")
    n_expected = len(expected)
    n_actual = len(index)
    gaps = n_expected - n_actual

    unique_diffs = pd.Series(index).diff().dropna().dt.total_seconds().unique()
    hourly = len(unique_diffs) > 0 and set(unique_diffs) == {3600.0}

    return {
        "name": name,
        "n_rows": int(n_actual),
        "n_expected": int(n_expected),
        "duplicate_timestamps": duplicates,
        "monotonic_increasing": monotonic,
        "hourly_frequency": bool(hourly),
        "missing_hours": int(gaps),
        "first_timestamp_utc": index.min().isoformat(),
        "last_timestamp_utc": index.max().isoformat(),
    }


# --------------------------------------------------------------------------- #
# Derived columns
# --------------------------------------------------------------------------- #
def _decision_time_for_date(delivery_date: date) -> datetime:
    """18:00 Europe/Berlin on the day before ``delivery_date``, in UTC."""

    prev = delivery_date - timedelta(days=1)
    local = datetime(prev.year, prev.month, prev.day, DECISION_LOCAL_HOUR, 0, tzinfo=BERLIN)
    return local.astimezone(UTC)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the target, benchmark, error, and metadata columns."""

    df = df.copy()

    df["wind_total_actual_mw"] = (
        df["wind_onshore_actual_mw"] + df["wind_offshore_actual_mw"]
    )
    df["wind_total_forecast_mw"] = (
        df["wind_onshore_forecast_mw"] + df["wind_offshore_forecast_mw"]
    )
    # Forecast error (actual - forecast). Positive => under-forecast.
    df["wind_forecast_error_mw"] = (
        df["wind_total_actual_mw"] - df["wind_total_forecast_mw"]
    )

    df["is_target_missing"] = df["wind_total_actual_mw"].isna()
    df["price_is_quarter_mean"] = df.index >= PRICE_METHODOLOGY_CHANGE_UTC

    df["decision_time_utc"] = df["delivery_date"].map(_decision_time_for_date)
    df["decision_time_utc"] = pd.to_datetime(df["decision_time_utc"], utc=True)

    return df


def _masked_shift(
    values: pd.Series, k: int, decision_times: pd.Series
) -> pd.Series:
    """Shift by ``k`` rows and mask values whose source time is not strictly
    before the forecast decision time."""

    shifted_values = values.shift(k)
    source_times = pd.Series(values.index, index=values.index).shift(k)
    available = source_times < decision_times
    return shifted_values.where(available)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly-historical lag and rolling features.

    Two families are produced:

    1. **Hour-specific masked lags** (``..._lag24/48/168``): a value is used
       only when its observation time is strictly before the decision time.
       Late delivery hours (whose lag would fall after the cutoff) are NaN.
    2. **Decision-time features** (``wind_actual_decision_*``): statistics of
       the actual wind series using only observations strictly before the
       decision time, broadcast to every delivery hour of that date. These
       represent what was known when the 18:00 forecast was issued.
    """

    df = df.copy()
    values = df["wind_total_actual_mw"]
    decision = df["decision_time_utc"]

    # 1) Hour-specific masked lags.
    for k in (24, 48, 168):
        df[f"wind_total_actual_mw_lag{k}"] = _masked_shift(values, k, decision)

    # 2) Decision-time features (constant within a delivery date).
    decision_columns = [
        "wind_actual_decision_lag1",
        "wind_actual_decision_lag24",
        "wind_actual_decision_lag48",
        "wind_actual_decision_lag168",
        "wind_actual_decision_roll24_mean",
        "wind_actual_decision_roll24_std",
    ]
    for col in decision_columns:
        df[col] = np.nan

    index = values.index
    for _, group in df.groupby("delivery_date", sort=False):
        dec = group["decision_time_utc"].iloc[0]
        # Number of observations strictly before the decision time.
        pos = int(index.searchsorted(dec, side="left"))

        if pos >= 1:
            df.loc[group.index, "wind_actual_decision_lag1"] = values.iloc[pos - 1]
        for k, col in (
            (24, "wind_actual_decision_lag24"),
            (48, "wind_actual_decision_lag48"),
            (168, "wind_actual_decision_lag168"),
        ):
            if pos >= k:
                df.loc[group.index, col] = values.iloc[pos - k]
        if pos >= 24:
            window = values.iloc[pos - 24 : pos]
            df.loc[group.index, "wind_actual_decision_roll24_mean"] = window.mean()
            df.loc[group.index, "wind_actual_decision_roll24_std"] = window.std()

    return df


def make_chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a chronological ``split`` column based on delivery date.

    Splitting on ``delivery_date`` (not the raw timestamp) keeps each delivery
    day wholly inside one period, which is required for day-ahead validation.
    Rows are never shuffled.
    """

    df = df.copy()
    delivery = pd.to_datetime(df["delivery_date"])
    split = pd.Series(pd.NaT, index=df.index, dtype=object)
    for label, (start, end) in SPLIT_BOUNDARIES.items():
        mask = (delivery >= pd.Timestamp(start)) & (delivery < pd.Timestamp(end))
        split.loc[mask] = label

    unassigned = int(split.isna().sum())
    if unassigned:
        raise DataValidationError(
            f"{unassigned} delivery dates fall outside the split boundaries"
        )
    df["split"] = split
    return df


# --------------------------------------------------------------------------- #
# Data-quality investigation
# --------------------------------------------------------------------------- #
def _max_consecutive_run(mask: pd.Series) -> int:
    """Length of the longest consecutive ``True`` run in a boolean Series."""

    longest = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def investigate_missing_load(df: pd.DataFrame) -> dict[str, Any]:
    """Investigate the missing ``load_forecast_mw`` values."""

    mask = df["load_forecast_mw"].isna()
    n_missing = int(mask.sum())
    report: dict[str, Any] = {
        "missing_load_forecast_rows": n_missing,
        "share_of_rows": float(n_missing / len(df)) if len(df) else 0.0,
    }
    if n_missing:
        missing_index = df.index[mask]
        report["first_missing_utc"] = missing_index.min().isoformat()
        report["last_missing_utc"] = missing_index.max().isoformat()
        report["distinct_missing_dates"] = int(
            pd.Series(missing_index.date).nunique()
        )
        report["max_consecutive_missing_hours"] = int(
            _max_consecutive_run(mask.to_numpy())
        )
        by_year = pd.Series(missing_index.year).value_counts().sort_index()
        report["missing_by_year"] = {int(k): int(v) for k, v in by_year.items()}
    return report


def investigate_missing_target(df: pd.DataFrame) -> dict[str, Any]:
    """Describe rows with a missing target (total actual wind)."""

    mask = df["wind_total_actual_mw"].isna()
    n_missing = int(mask.sum())
    report: dict[str, Any] = {
        "missing_target_rows": n_missing,
        "missing_target_timestamps_utc": [],
    }
    if n_missing:
        report["missing_target_timestamps_utc"] = [
            ts.isoformat() for ts in df.index[mask]
        ]
    return report


def investigate_prices(df: pd.DataFrame) -> dict[str, Any]:
    """Describe prices, including the valid negative observations."""

    prices = df["price_eur_mwh"]
    return {
        "n_negative_prices": int((prices < 0).sum()),
        "share_negative_prices": float((prices < 0).mean()) if len(prices) else 0.0,
        "min_price_eur_mwh": float(prices.min()),
        "max_price_eur_mwh": float(prices.max()),
        "n_missing_prices": int(prices.isna().sum()),
        "price_methodology_change_utc": PRICE_METHODOLOGY_CHANGE_UTC.isoformat(),
        "rows_before_change": int((~df["price_is_quarter_mean"]).sum()),
        "rows_after_change": int(df["price_is_quarter_mean"].sum()),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    df: pd.DataFrame
    report: dict[str, Any]


def build_processed_dataset(
    raw_path: str | Path = "data/raw/snapshot.parquet",
    out_path: str | Path = "data/processed/de_lu_hourly.parquet",
    report_path: str | Path = "data/processed/preprocessing_report.json",
) -> PreprocessResult:
    """Run the full preprocessing pipeline and persist outputs."""

    raw_path = Path(raw_path)
    out_path = Path(out_path)
    report_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_snapshot(raw_path)
    index_report = validate_hourly_index(df)

    missing_before = {
        "load_forecast_mw": int(df["load_forecast_mw"].isna().sum()),
        "wind_onshore_actual_mw": int(df["wind_onshore_actual_mw"].isna().sum()),
        "wind_offshore_actual_mw": int(df["wind_offshore_actual_mw"].isna().sum()),
        "wind_onshore_forecast_mw": int(df["wind_onshore_forecast_mw"].isna().sum()),
        "wind_offshore_forecast_mw": int(df["wind_offshore_forecast_mw"].isna().sum()),
        "price_eur_mwh": int(df["price_eur_mwh"].isna().sum()),
    }

    df = add_derived_columns(df)
    df = add_lag_features(df)
    df = make_chronological_split(df)

    split_counts = {str(k): int(v) for k, v in df["split"].value_counts().items()}

    lag_columns = [
        "wind_total_actual_mw_lag24",
        "wind_total_actual_mw_lag48",
        "wind_total_actual_mw_lag168",
        "wind_actual_decision_lag1",
        "wind_actual_decision_lag24",
        "wind_actual_decision_lag48",
        "wind_actual_decision_lag168",
        "wind_actual_decision_roll24_mean",
        "wind_actual_decision_roll24_std",
    ]
    lag_nulls = {col: int(df[col].isna().sum()) for col in lag_columns}

    report: dict[str, Any] = {
        "source": {
            "path": str(raw_path),
            "sha256": sha256_of_file(raw_path),
            "modified": False,
        },
        "output": {"path": str(out_path)},
        "index_validation": index_report,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_before": missing_before,
        "missing_target": investigate_missing_target(df),
        "missing_load_investigation": investigate_missing_load(df),
        "prices": investigate_prices(df),
        "split_counts": split_counts,
        "lag_null_counts": lag_nulls,
        "transformations": [
            "set timestamp_utc as a UTC DatetimeIndex",
            "converted delivery_date to date",
            "wind_total_actual_mw = onshore_actual + offshore_actual",
            "wind_total_forecast_mw = onshore_forecast + offshore_forecast",
            "wind_forecast_error_mw = wind_total_actual_mw - wind_total_forecast_mw",
            "is_target_missing flags rows without a total actual wind value",
            "price_is_quarter_mean flags the 2025-10-01 aggregation change",
            "decision_time_utc = 18:00 Europe/Berlin on the previous day (UTC)",
            "lag features masked against decision_time_utc (no leakage)",
            "decision-time features computed from observations strictly before 18:00 "
            "Europe/Berlin and broadcast to each delivery date",
            "chronological train/validation/test split assigned (no shuffling)",
        ],
        "notes": [
            "Missing target values are preserved as NaN; no imputation was applied.",
            "Negative prices are retained as valid market observations.",
            "Lag features are NaN during the initial warm-up window by construction.",
        ],
    }

    df.to_parquet(out_path, index=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return PreprocessResult(df=df, report=report)


def main() -> int:
    result = build_processed_dataset()
    report = result.report
    print("Preprocessing complete.")
    print(f"  rows: {report['n_rows']} | columns: {report['n_columns']}")
    print(f"  output: {report['output']['path']}")
    print(f"  index: {report['index_validation']}")
    print(f"  missing load forecast: {report['missing_load_investigation']}")
    print(f"  missing target: {report['missing_target']}")
    print(f"  prices: {report['prices']}")
    print(f"  splits: {report['split_counts']}")
    print(f"  lag nulls: {report['lag_null_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())