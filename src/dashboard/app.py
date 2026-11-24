"""Streamlit dashboard for the European Wind Power Forecasting project.

Run from the project root:

    streamlit run src/dashboard/app.py

The dashboard reads only committed artifacts under ``reports/`` and never
re-trains models or reads the raw/processed datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data import (  # noqa: E402
    DashboardDataError,
    PREDICTION_COLUMNS,
    compute_metrics,
    error_by_hour,
    error_by_season,
    filter_by_date,
    load_metadata,
    load_results,
    model_table,
    predictions_aligned,
)

st.set_page_config(
    page_title="European Wind Power Forecasting",
    page_icon="🌬️",
    layout="wide",
)

PRED_LABELS = {
    "ridge_pred_mw": "Ridge",
    "lightgbm_pred_mw": "LightGBM",
    "tso_forecast_mw": "TSO official",
}
COLORS = {
    "actual_wind_mw": "#111827",
    "ridge_pred_mw": "#2563eb",
    "lightgbm_pred_mw": "#16a34a",
    "tso_forecast_mw": "#f59e0b",
}


def load_dashboard_data():
    try:
        df = load_results()
        meta = load_metadata()
        return df, meta, None
    except DashboardDataError as exc:
        return None, None, str(exc)


def section_overview(meta: dict) -> None:
    st.title("European Wind Power Forecasting & Market Analytics")
    st.caption("Hourly DE-LU wind generation forecasting for day-ahead trading")

    coverage = meta.get("coverage", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Market", meta.get("market", "DE-LU"))
    col2.metric("Historical coverage", f"{coverage.get('start', '?')[:10]} → {coverage.get('end', '?')[:10]}")
    col3.metric("Forecast decision", "18:00 Europe/Berlin (D-1)")

    st.markdown("### Business objective")
    st.write(meta.get("business_objective", ""))

    st.markdown("### How it works")
    st.write(
        "The project forecasts the next 24 hours of German-Luxembourg wind "
        "generation using only information available at the 18:00 Europe/Berlin "
        "decision time on the previous day. Actual generation follows an H+1 "
        "publication rule, so every historical feature is masked against that "
        "cutoff to prevent look-ahead bias."
    )

    st.markdown("### Model snapshot")
    st.write(
        "Ridge and LightGBM are trained on calendar, cyclical, forecast-horizon, "
        "and H+1-safe wind-history features. The official TSO day-ahead wind "
        "forecast is kept as an **independent benchmark**, not as a model input."
    )


def section_performance(df: pd.DataFrame, meta: dict) -> None:
    st.header("Model Performance")
    st.caption("Held-out test set (2025-01-01 → 2026-09-06)")

    table = model_table(df, meta).set_index("model")
    persistence = table.loc["Persistence"] if "Persistence" in table.index else None

    cols = st.columns(4)
    cols[0].metric("Ridge MAE (MW)", f"{table.loc['Ridge', 'mae']:,.1f}")
    cols[1].metric("LightGBM MAE (MW)", f"{table.loc['LightGBM', 'mae']:,.1f}")
    if persistence is not None:
        cols[2].metric("Persistence MAE (MW)", f"{persistence['mae']:,.1f}")
    cols[3].metric("TSO MAE (MW)", f"{table.loc['TSO official', 'mae']:,.1f}")

    st.markdown("### Detailed metrics")
    st.dataframe(table.round(3), use_container_width=True)

    st.info(
        "**The official TSO day-ahead forecast is the best-performing benchmark "
        "in the current experiment.** The machine-learning models improve on the "
        "persistence baseline but do not outperform the TSO forecast."
    )

    fig = go.Figure()
    for model in table.index:
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[table.loc[model, "mae"]],
                name=model,
                marker_color=COLORS.get(
                    {"Ridge": "ridge_pred_mw", "LightGBM": "lightgbm_pred_mw",
                     "TSO official": "tso_forecast_mw"}.get(model, ""),
                    "#6b7280",
                ),
                text=[f"{table.loc[model, 'mae']:,.0f}"],
                textposition="outside",
            )
        )
    fig.update_layout(
        title="Test MAE by model (MW)",
        yaxis_title="MAE (MW)",
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def section_actual_vs_predicted(df: pd.DataFrame) -> None:
    st.header("Actual vs Predicted")
    st.caption("Interactive held-out test series. Dates are interpreted in UTC.")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=120).days)
    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=default_start, min_value=min_date, max_value=max_date)
    end = col2.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    filtered = filter_by_date(df, start, end)
    if filtered.empty:
        st.warning("No observations in the selected range.")
        return

    st.caption(f"{len(filtered):,} hourly observations in range")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["actual_wind_mw"],
                             name="Actual", line=dict(color=COLORS["actual_wind_mw"], width=1.4)))
    for col in PREDICTION_COLUMNS:
        fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered[col],
                                 name=PRED_LABELS[col], line=dict(color=COLORS[col], width=1.2)))
    fig.update_layout(
        title="Actual vs predicted wind generation (MW)",
        yaxis_title="MW",
        xaxis_title="Time (UTC)",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Zoomed view of the last 14 days of the selection.
    zoom = filtered.tail(24 * 14)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=zoom["timestamp"], y=zoom["actual_wind_mw"],
                              name="Actual", line=dict(color=COLORS["actual_wind_mw"])))
    for col in PREDICTION_COLUMNS:
        fig2.add_trace(go.Scatter(x=zoom["timestamp"], y=zoom[col],
                                  name=PRED_LABELS[col], line=dict(color=COLORS[col])))
    fig2.update_layout(title="Zoom — last 14 days of selection",
                       yaxis_title="MW", height=420,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)


def section_error_analysis(df: pd.DataFrame) -> None:
    st.header("Error Analysis")
    st.caption("Errors are prediction minus actual (positive = over-prediction).")

    model = st.selectbox("Model", ["Ridge", "LightGBM", "TSO official"])
    pred_col = {"Ridge": "ridge_pred_mw", "LightGBM": "lightgbm_pred_mw",
                "TSO official": "tso_forecast_mw"}[model]

    col1, col2 = st.columns(2)
    by_hour = error_by_hour(df, pred_col)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_hour.index, y=by_hour["mae"], name="MAE",
                         marker_color=COLORS[pred_col]))
    fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour["mean_error"], name="Mean error",
                             mode="lines+markers", line=dict(color="#111827")))
    fig.update_layout(title=f"Error by hour of day — {model}", xaxis_title="Hour (UTC)",
                      yaxis_title="MW", height=420)
    col1.plotly_chart(fig, use_container_width=True)

    by_season = error_by_season(df, pred_col)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=by_season.index, y=by_season["mae"], name="MAE",
                          marker_color=COLORS[pred_col]))
    fig2.add_trace(go.Scatter(x=by_season.index, y=by_season["mean_error"], name="Mean error",
                              mode="lines+markers", line=dict(color="#111827")))
    fig2.update_layout(title=f"Error by season — {model}", yaxis_title="MW", height=420)
    col2.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Error distribution")
    fig3 = go.Figure()
    for col, label in (("ridge_pred_mw", "Ridge"), ("lightgbm_pred_mw", "LightGBM"),
                       ("tso_forecast_mw", "TSO official")):
        err = (df[col] - df["actual_wind_mw"]).dropna()
        fig3.add_trace(go.Histogram(x=err, name=label, opacity=0.55,
                                    marker_color=COLORS[col], nbinsx=80))
    fig3.update_layout(barmode="overlay", title="Forecast error distribution",
                       xaxis_title="error (MW)", yaxis_title="count", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Model comparison")
    table = model_table(df, {}).set_index("model")
    fig4 = go.Figure()
    for metric, color in (("mae", "#2563eb"), ("rmse", "#16a34a")):
        fig4.add_trace(go.Bar(x=table.index, y=table[metric], name=metric.upper(),
                              marker_color=color))
    fig4.update_layout(barmode="group", title="MAE and RMSE by model (MW)",
                       yaxis_title="MW", height=420)
    st.plotly_chart(fig4, use_container_width=True)


def section_data_quality(meta: dict) -> None:
    st.header("Data Quality & Methodology")

    coverage = meta.get("coverage", {})
    quality = meta.get("data_quality", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{coverage.get('rows', 0):,}")
    col2.metric("Timezone", quality.get("timezone", "UTC"))
    col3.metric("Frequency", "Hourly" if quality.get("hourly_frequency") else "Irregular")

    st.markdown("### Missing values")
    missing = quality.get("missing_values", {})
    if missing:
        st.dataframe(pd.DataFrame({"column": list(missing), "missing": list(missing.values())}).set_index("column"),
                     use_container_width=True)
    st.caption(f"Duplicate timestamps: {quality.get('duplicate_timestamps', 0)} · "
               f"Negative prices retained: {quality.get('negative_prices', 0):,}")

    st.markdown("### Methodology")
    st.write(f"- **Decision time:** {meta.get('decision_time', '')}")
    st.write(f"- **Timestamp convention:** {meta.get('timestamp_convention', '')}")
    st.write(f"- **Actual publication:** {meta.get('actual_publication', '')}")
    st.write(f"- **DST handling:** {meta.get('dst_handling', '')}")

    st.markdown("### Known limitations")
    for item in meta.get("limitations", []):
        st.write(f"- {item}")

    st.markdown("### Weather pilot status")
    pilot = meta.get("weather_pilot", {})
    if pilot.get("available"):
        st.write(
            f"- 7-day Open-Meteo pilot: {pilot.get('n_records', 0):,} records, "
            f"{pilot.get('n_locations', 0)} locations, model {pilot.get('model', '')}."
        )
        st.write(f"- **Not used for historical model training.** {pilot.get('note', '')}")
    else:
        st.write("- Weather pilot summary not available.")


def section_weather_pilot(meta: dict) -> None:
    st.header("Weather Pilot (Pipeline Demonstration)")
    pilot = meta.get("weather_pilot", {})
    if not pilot.get("available"):
        st.warning("Weather pilot summary is not available in the committed metadata.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{pilot.get('n_records', 0):,}")
    col2.metric("Locations", pilot.get("n_locations", 0))
    col3.metric("Missing values", pilot.get("missing_values", 0))
    col4.metric("Lead time (mean h)", f"{pilot.get('lead_hours', {}).get('mean', 0):.1f}")

    st.write(f"**Source:** {pilot.get('source', '')} · **Model:** {pilot.get('model', '')}")
    dates = pilot.get("delivery_dates", [])
    if dates:
        st.write(f"**Delivery dates:** {dates[0]} → {dates[-1]}")

    daily = pd.DataFrame(pilot.get("daily_mean_wind_speed", []))
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        pivot = daily.pivot(index="date", columns="location_label", values="wind_speed_100m")
        fig = go.Figure()
        for loc in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[loc], name=loc, mode="lines+markers"))
        fig.update_layout(title="Daily mean wind speed at 100 m by location (km/h)",
                          xaxis_title="Date", yaxis_title="km/h", height=460)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**This pilot validates the weather-collection pipeline only.** It is not "
        "used as a long-term historical training feature. "
        + pilot.get("attribution", "")
    )


def main() -> None:
    df, meta, error = load_dashboard_data()

    if error is not None:
        st.error(f"Dashboard data could not be loaded: {error}")
        st.markdown(
            "To regenerate the committed artifacts, run from the project root:\n\n"
            "```bash\npython -m src.models.train\n```"
        )
        return

    if not predictions_aligned(df):
        st.error("Dashboard results are misaligned or contain missing predictions.")
        return

    with st.sidebar:
        st.markdown("## Navigation")
        section = st.radio(
            "Section",
            [
                "Project Overview",
                "Model Performance",
                "Actual vs Predicted",
                "Error Analysis",
                "Data Quality & Methodology",
                "Weather Pilot",
            ],
        )
        st.markdown("---")
        st.caption(meta.get("attribution", ""))
        st.caption("License: CC BY 4.0")

    if section == "Project Overview":
        section_overview(meta)
    elif section == "Model Performance":
        section_performance(df, meta)
    elif section == "Actual vs Predicted":
        section_actual_vs_predicted(df)
    elif section == "Error Analysis":
        section_error_analysis(df)
    elif section == "Data Quality & Methodology":
        section_data_quality(meta)
    elif section == "Weather Pilot":
        section_weather_pilot(meta)


if __name__ == "__main__":
    main()