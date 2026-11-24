"""Naive forecasting baselines that respect the D-1 18:00 information cutoff.

All baselines use only observations that are available at the forecast decision
time under the H+1 publication rule.
"""

from __future__ import annotations

import pandas as pd

from .availability import ACTUAL_PUBLICATION_DELAY_HOURS

DECISION_COLUMN = "decision_time_utc"
TARGET_COLUMN = "wind_total_actual_mw"


def _source_times(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index, index=df.index)


def persistence_forecast(df: pd.DataFrame) -> pd.Series:
    """Most recent available actual generation at the decision time.

    Uses the last interval whose H+1 publication is at or before the decision
    time (i.e. ``decision_time - 1h``), broadcast to the delivery day.
    """

    decision = pd.to_datetime(df[DECISION_COLUMN], utc=True)
    source = decision - pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)

    values = df[TARGET_COLUMN]
    # Position of the last observation strictly before the decision cutoff.
    positions = df.index.searchsorted(source, side="right") - 1
    positions = pd.Series(positions, index=df.index)

    result = pd.Series(index=df.index, dtype=float)
    valid = positions >= 0
    result.loc[valid] = values.to_numpy()[positions.to_numpy()[valid.to_numpy()]]
    return result


def seasonal_naive_forecast(df: pd.DataFrame, lag_hours: int) -> pd.Series:
    """Seasonal naive forecast using ``lag_hours`` (48 or 168).

    A value is used only when its H+1 publication is at or before the decision
    time. For lags >= 48 h this is always satisfied for a delivery day.
    """

    if lag_hours <= 0:
        raise ValueError("lag_hours must be positive")

    values = df[TARGET_COLUMN]
    shifted = values.shift(lag_hours)
    source = _source_times(df).shift(lag_hours)
    decision = pd.to_datetime(df[DECISION_COLUMN], utc=True)
    available = (source + pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)) <= decision
    return shifted.where(available)


def build_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with all baseline predictions."""

    return pd.DataFrame(
        {
            "persistence": persistence_forecast(df),
            "seasonal_naive_48h": seasonal_naive_forecast(df, 48),
            "seasonal_naive_168h": seasonal_naive_forecast(df, 168),
        },
        index=df.index,
    )