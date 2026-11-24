"""Feature engineering for the day-ahead wind forecasting models.

Only features that are genuinely available at the 18:00 Europe/Berlin decision
time are constructed. Contemporaneous actual generation, the TSO forecast, the
day-ahead price, the load forecast, and the 7-day weather pilot are **not** used
as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "wind_total_actual_mw"
DECISION_COLUMN = "decision_time_utc"

# Features available at the forecast decision time.
FEATURE_COLUMNS: list[str] = [
    # Calendar / cyclical encodings
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "season_winter",
    "season_spring",
    "season_summer",
    "season_autumn",
    # Forecast horizon (hours from decision to valid time)
    "forecast_horizon_hours",
    # Decision-time wind history (H+1 safe, constant within a delivery day)
    "wind_actual_decision_lag1",
    "wind_actual_decision_lag24",
    "wind_actual_decision_lag48",
    "wind_actual_decision_lag168",
    "wind_actual_decision_roll24_mean",
    "wind_actual_decision_roll24_std",
    # Hour-specific seasonal lags (always before the cutoff for a delivery day)
    "wind_total_actual_mw_lag48",
    "wind_total_actual_mw_lag168",
]

FORBIDDEN_FEATURES: list[str] = [
    "wind_total_forecast_mw",
    "price_eur_mwh",
    "load_forecast_mw",
    "wind_total_actual_mw",
]

_SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def _cyclical(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclical, and forecast-horizon features."""

    df = df.copy()
    index = df.index

    hour = index.hour.to_numpy()
    dow = index.day_of_week.to_numpy()
    month = index.month.to_numpy()

    df["hour_sin"], df["hour_cos"] = _cyclical(hour, 24)
    df["dow_sin"], df["dow_cos"] = _cyclical(dow, 7)
    df["month_sin"], df["month_cos"] = _cyclical(month, 12)
    df["is_weekend"] = (dow >= 5).astype(int)

    season = pd.Series(month, index=index).map(_SEASON_BY_MONTH)
    for name in ("winter", "spring", "summer", "autumn"):
        df[f"season_{name}"] = (season == name).astype(int)

    decision = pd.to_datetime(df[DECISION_COLUMN], utc=True)
    df["forecast_horizon_hours"] = (
        pd.Series(index, index=index) - decision
    ).dt.total_seconds() / 3600.0

    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, eligible_mask)``.

    Rows with a missing target or any missing feature (warm-up) are marked
    ineligible and excluded from training/evaluation.
    """

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing engineered features: {missing}")

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(float)
    eligible = y.notna() & X.notna().all(axis=1)
    return X, y, eligible