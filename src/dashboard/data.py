"""Data loading and helpers for the Streamlit dashboard.

The dashboard reads only compact, committed artifacts under ``reports/``:

- ``dashboard_results.parquet`` — held-out test predictions (actual, Ridge,
  LightGBM, TSO) plus horizon/season.
- ``dashboard_metadata.json`` — coverage, splits, metrics, data quality,
  methodology, attribution, and the weather-pilot summary.

It never reads the raw/processed datasets or model binaries, so it works on a
fresh GitHub clone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.metrics import regression_metrics  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "dashboard_results.parquet"
METADATA_PATH = REPORTS_DIR / "dashboard_metadata.json"

REQUIRED_COLUMNS = [
    "timestamp",
    "actual_wind_mw",
    "ridge_pred_mw",
    "lightgbm_pred_mw",
    "tso_forecast_mw",
]
PREDICTION_COLUMNS = ["ridge_pred_mw", "lightgbm_pred_mw", "tso_forecast_mw"]


class DashboardDataError(RuntimeError):
    """Raised when a required dashboard artifact is missing or invalid."""


def load_results(path: str | Path = RESULTS_PATH) -> pd.DataFrame:
    """Load and validate the compact test-results dataset."""

    path = Path(path)
    if not path.exists():
        raise DashboardDataError(
            f"Dashboard results not found at {path}. "
            "Run `python -m src.models.train` to regenerate it."
        )

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise DashboardDataError(f"Could not read {path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DashboardDataError(f"Dashboard results missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df.reset_index(drop=True)

    if df.empty:
        raise DashboardDataError("Dashboard results are empty")

    return df


def load_metadata(path: str | Path = METADATA_PATH) -> dict[str, Any]:
    """Load the dashboard metadata JSON."""

    path = Path(path)
    if not path.exists():
        raise DashboardDataError(
            f"Dashboard metadata not found at {path}. "
            "Run `python -m src.models.train` to regenerate it."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DashboardDataError(f"Could not parse {path}: {exc}") from exc


def filter_by_date(
    df: pd.DataFrame, start: Any | None = None, end: Any | None = None
) -> pd.DataFrame:
    """Filter by an inclusive UTC date range.

    ``start``/``end`` may be strings, dates, or timestamps. ``end`` is inclusive
    (the whole end day is included).
    """

    result = df
    if start is not None:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        result = result[result["timestamp"] >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        result = result[result["timestamp"] < end_ts + pd.Timedelta(days=1)]
    return result


def predictions_aligned(df: pd.DataFrame) -> bool:
    """True when actual and every prediction column are present and aligned."""

    if df.empty:
        return False
    if df[REQUIRED_COLUMNS].isna().any().any():
        return False
    return len(df) == df["timestamp"].nunique()


def compute_metrics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute MAE/RMSE/bias/mean-normalized MAE for each prediction column."""

    metrics: dict[str, dict[str, float]] = {}
    for col in PREDICTION_COLUMNS:
        metrics[col] = regression_metrics(df["actual_wind_mw"], df[col])
    return metrics


def model_table(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Build the model-comparison table (test metrics + persistence)."""

    computed = compute_metrics(df)
    rows = []
    for model, key in (
        ("Ridge", "ridge_pred_mw"),
        ("LightGBM", "lightgbm_pred_mw"),
        ("TSO official", "tso_forecast_mw"),
    ):
        m = computed[key]
        rows.append(
            {
                "model": model,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": m["bias"],
                "mean_normalized_mae": m["mean_normalized_mae"],
            }
        )

    # Persistence metrics come from the committed metadata.
    persistence = next(
        (
            r
            for r in metadata.get("models_test_metrics", [])
            if r.get("model") == "persistence"
        ),
        None,
    )
    if persistence is not None:
        rows.append(
            {
                "model": "Persistence",
                "mae": persistence["mae"],
                "rmse": persistence["rmse"],
                "bias": persistence["bias"],
                "mean_normalized_mae": persistence["mean_normalized_mae"],
            }
        )

    table = pd.DataFrame(rows)
    persistence_mae = table.loc[table["model"] == "Persistence", "mae"]
    if len(persistence_mae):
        base = float(persistence_mae.iloc[0])
        table["improvement_vs_persistence_pct"] = (
            (base - table["mae"]) / base * 100.0
        )
    else:
        table["improvement_vs_persistence_pct"] = float("nan")
    return table


def error_by_hour(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    error = df[pred_col] - df["actual_wind_mw"]
    frame = pd.DataFrame({"hour": df["timestamp"].dt.hour, "error": error})
    out = frame.groupby("hour")["error"].agg(mean_error="mean", mae=lambda s: s.abs().mean())
    return out


def error_by_season(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    error = df[pred_col] - df["actual_wind_mw"]
    season = df.get("season")
    if season is None:
        season = df["timestamp"].dt.month.map(
            {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
             5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
             9: "Autumn", 10: "Autumn", 11: "Autumn"}
        )
    frame = pd.DataFrame({"season": season, "error": error})
    out = frame.groupby("season")["error"].agg(mean_error="mean", mae=lambda s: s.abs().mean())
    return out.reindex(["Winter", "Spring", "Summer", "Autumn"])