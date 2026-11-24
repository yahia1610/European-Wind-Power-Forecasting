"""Forecast-availability audit for the DE-LU day-ahead problem.

Forecast decision time: 18:00 Europe/Berlin on D-1.

Timestamp convention
--------------------
``timestamp_utc`` is the **interval start** (the beginning of the hour). This is
verified against ``local_hour`` / ``delivery_date`` in the snapshot.

Actual-data publication
-----------------------
Actual generation for an interval starting at ``t`` is published with an
**H+1** deadline, i.e. it is available at ``t + 1 hour`` at the latest. An
observation is therefore usable at decision time ``d`` only if
``t + 1h <= d``. Because the data is hourly, this is equivalent to ``t < d``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

UTC = timezone.utc
TIMESTAMP_CONVENTION = "interval_start"
ACTUAL_PUBLICATION_DELAY_HOURS = 1  # H+1


def verify_timestamp_convention(df: pd.DataFrame) -> dict[str, Any]:
    """Verify that ``timestamp_utc`` is the interval start."""

    if "utc_offset_minutes" not in df.columns or "local_hour" not in df.columns:
        return {"verified": False, "reason": "required columns missing"}

    local_dt = pd.Series(df.index, index=df.index) + pd.to_timedelta(
        df["utc_offset_minutes"], unit="m"
    )
    hour_ok = bool((local_dt.dt.hour == df["local_hour"]).all())
    delivery_ok = bool(
        (local_dt.dt.date == pd.to_datetime(df["delivery_date"]).dt.date).all()
    )
    return {
        "convention": TIMESTAMP_CONVENTION,
        "local_hour_matches": hour_ok,
        "delivery_date_matches": delivery_ok,
        "verified": bool(hour_ok and delivery_ok),
    }


def actual_available_at(
    interval_start: pd.Series | datetime, decision_time: pd.Series | datetime
) -> pd.Series | bool:
    """Return whether an actual observation is available by the decision time.

    Uses the H+1 rule: ``interval_start + 1h <= decision_time``. Returns a
    scalar ``bool`` for scalar inputs and a boolean Series for Series inputs.
    """

    start_is_series = isinstance(interval_start, pd.Series)
    decision_is_series = isinstance(decision_time, pd.Series)

    if not start_is_series and not decision_is_series:
        return bool(
            (interval_start + timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS))
            <= decision_time
        )

    start = interval_start if start_is_series else pd.Series([interval_start])
    decision = decision_time if decision_is_series else pd.Series([decision_time])
    return (start + pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)) <= decision


def latest_available_actual_start(decision_time: datetime) -> datetime:
    """Latest interval start whose H+1 publication is at or before decision."""

    return decision_time - timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)


def audit_availability(df: pd.DataFrame) -> dict[str, Any]:
    """Audit feature availability against the D-1 18:00 decision cutoff."""

    convention = verify_timestamp_convention(df)
    decision = pd.to_datetime(df["decision_time_utc"], utc=True)

    # The decision-time lag-1 feature must reference an H+1-available interval.
    lag1_source = decision - pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)
    lag1_available = bool(
        ((lag1_source + pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)) <= decision).all()
    )

    # Hour-specific lag features must also satisfy H+1 availability.
    lag48_source = pd.Series(df.index, index=df.index) - pd.Timedelta(hours=48)
    lag168_source = pd.Series(df.index, index=df.index) - pd.Timedelta(hours=168)
    lag48_available = bool(
        ((lag48_source + pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)) <= decision).all()
    )
    lag168_available = bool(
        ((lag168_source + pd.Timedelta(hours=ACTUAL_PUBLICATION_DELAY_HOURS)) <= decision).all()
    )

    return {
        "timestamp_convention": convention,
        "decision_time_rule": "18:00 Europe/Berlin on D-1",
        "actual_publication_delay_hours": ACTUAL_PUBLICATION_DELAY_HOURS,
        "decision_lag1_respects_h1": lag1_available,
        "lag48_respects_h1": lag48_available,
        "lag168_respects_h1": lag168_available,
        "excluded_features": [
            "wind_total_actual_mw (target, contemporaneous)",
            "wind_total_forecast_mw (kept as independent benchmark only)",
            "price_eur_mwh (publication time not verified in snapshot)",
            "load_forecast_mw (publication time not verified in snapshot)",
            "weather pilot (7 days only, not a long-term feature)",
        ],
        "included_features": [
            "calendar / cyclical encodings",
            "forecast horizon (valid_time - decision_time)",
            "decision-time wind lags and rolling stats (H+1 safe)",
            "hour-specific 48 h and 168 h wind lags (H+1 safe)",
        ],
    }