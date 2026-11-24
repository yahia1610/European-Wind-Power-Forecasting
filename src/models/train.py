"""Train and evaluate day-ahead wind forecasting models.

Models: Ridge Regression and LightGBM Regressor.
Baselines: persistence, seasonal naive 48 h, seasonal naive 168 h.
Benchmark: official TSO day-ahead wind forecast (independent).

The test set is used only for the final evaluation, after model selection on the
validation set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import build_baselines
from .features import FEATURE_COLUMNS, add_calendar_features, build_feature_matrix
from .metrics import improvement_pct, regression_metrics

PROCESSED_PATH = Path("data/processed/de_lu_hourly.parquet")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
MODEL_DIR = Path("models")

TARGET = "wind_total_actual_mw"
TSO = "wind_total_forecast_mw"


def _split_masks(df: pd.DataFrame, eligible: pd.Series) -> dict[str, pd.Series]:
    return {
        name: (df["split"] == name) & eligible
        for name in ("train", "validation", "test")
    }


def _fit_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def _fit_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def _metrics_row(model: str, split: str, y_true: pd.Series, y_pred: pd.Series,
                 baseline_mae: float | None = None) -> dict[str, Any]:
    m = regression_metrics(y_true, y_pred)
    row = {
        "model": model,
        "split": split,
        "n": m["n"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "mean_normalized_mae": m["mean_normalized_mae"],
        "normalization_denominator": m["normalization_denominator"],
    }
    if baseline_mae is not None:
        row["improvement_vs_persistence_pct"] = improvement_pct(baseline_mae, m["mae"])
    return row


def _plot_actual_vs_predicted(test: pd.DataFrame, y_true: pd.Series,
                              y_pred: pd.Series, model_name: str) -> None:
    frame = pd.DataFrame({"actual": y_true, "predicted": y_pred}).dropna()
    zoom = frame.tail(24 * 90)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(frame.index, frame["actual"], lw=0.5, label="actual")
    axes[0].plot(frame.index, frame["predicted"], lw=0.5, label="predicted")
    axes[0].set_title(f"Actual vs predicted — full test ({model_name})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(zoom.index, zoom["actual"], lw=0.9, label="actual")
    axes[1].plot(zoom.index, zoom["predicted"], lw=0.9, label="predicted")
    axes[1].set_title("Actual vs predicted — last 90 days of test")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_actual_vs_predicted.png", dpi=110)
    plt.close(fig)


def _plot_model_comparison(rows: pd.DataFrame) -> None:
    test = rows[rows["split"] == "test"].set_index("model")
    models = list(test.index)
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, test["mae"], width, label="MAE")
    ax.bar(x + width / 2, test["rmse"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("MW")
    ax.set_title("Test metrics by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_model_comparison.png", dpi=110)
    plt.close(fig)


def _plot_error_distribution(test: pd.DataFrame, best_pred: pd.Series,
                             model_name: str) -> None:
    best_err = (best_pred - test[TARGET]).dropna()
    tso_err = (test[TSO] - test[TARGET]).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-8000, 8000, 121)
    ax.hist(tso_err, bins=bins, alpha=0.55, label="TSO forecast", density=True)
    ax.hist(best_err, bins=bins, alpha=0.55, label=model_name, density=True)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("error = prediction - actual (MW)")
    ax.set_ylabel("density")
    ax.set_title("Forecast error distribution (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_error_distribution.png", dpi=110)
    plt.close(fig)


def _plot_error_by_group(test: pd.DataFrame, best_pred: pd.Series,
                         model_name: str) -> None:
    frame = pd.DataFrame(
        {
            "error": best_pred - test[TARGET],
            "hour": test.index.hour,
            "season": test.index.month.map(
                {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                 9: "Autumn", 10: "Autumn", 11: "Autumn"}
            ),
            "horizon": test["forecast_horizon_hours"],
        }
    ).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    by_hour = frame.groupby("hour")["error"].agg(["mean", lambda s: s.abs().mean()])
    by_hour.columns = ["mean_error", "mae"]
    by_hour.plot(ax=axes[0], marker="o")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Error by hour of day (UTC)")
    axes[0].set_xlabel("hour")

    by_season = frame.groupby("season")["error"].agg(["mean", lambda s: s.abs().mean()])
    by_season.columns = ["mean_error", "mae"]
    by_season = by_season.reindex(["Winter", "Spring", "Summer", "Autumn"])
    by_season.plot(ax=axes[1], kind="bar")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Error by season")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_error_by_hour_season.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    by_horizon = frame.groupby("horizon")["error"].agg(["mean", lambda s: s.abs().mean()])
    by_horizon.columns = ["mean_error", "mae"]
    by_horizon.plot(ax=ax, marker="o")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("forecast horizon (hours from decision)")
    ax.set_title("Error by forecast horizon")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_error_by_horizon.png", dpi=110)
    plt.close(fig)


def _season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _weather_pilot_summary() -> dict[str, Any]:
    """Summarize the 7-day weather pilot for the dashboard (if available)."""

    path = Path("data/processed/weather_test/weather_pilot.parquet")
    manifest_path = Path("data/processed/weather_test/manifest.json")
    if not path.exists():
        return {"available": False}

    weather = pd.read_parquet(path)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["valid_time_utc"]).dt.date
    daily = (
        weather.groupby(["date", "location_label"])["wind_speed_100m"]
        .mean()
        .reset_index()
    )
    daily["date"] = daily["date"].astype(str)

    return {
        "available": True,
        "source": "Open-Meteo Single Runs API",
        "model": "ECMWF IFS HRES 9km (ecmwf_ifs)",
        "delivery_dates": manifest.get("delivery_dates", []),
        "n_locations": int(weather["location_label"].nunique()),
        "n_records": int(len(weather)),
        "missing_values": int(
            weather[
                ["wind_speed_100m", "wind_direction_100m", "temperature_2m", "surface_pressure"]
            ].isna().sum().sum()
        ),
        "lead_hours": {
            "min": float(weather["lead_hours"].min()),
            "max": float(weather["lead_hours"].max()),
            "mean": float(weather["lead_hours"].mean()),
        },
        "daily_mean_wind_speed": daily.to_dict(orient="records"),
        "license": "CC BY 4.0",
        "attribution": "Weather data: Open-Meteo.com (CC BY 4.0), ECMWF IFS HRES 9km.",
        "note": "Pipeline-validation pilot only; NOT used for historical model training.",
    }


def _export_dashboard_results(
    test_frame: pd.DataFrame,
    y_eval: pd.Series,
    ridge_eval: pd.Series,
    lgbm_eval: pd.Series,
    tso_eval: pd.Series,
    results: pd.DataFrame,
    summary: dict[str, Any],
    df: pd.DataFrame,
) -> None:
    """Write the compact, committed dashboard dataset and metadata."""

    out = pd.DataFrame(
        {
            "timestamp": test_frame.index,
            "actual_wind_mw": y_eval.to_numpy(),
            "ridge_pred_mw": ridge_eval.to_numpy(),
            "lightgbm_pred_mw": lgbm_eval.to_numpy(),
            "tso_forecast_mw": tso_eval.to_numpy(),
            "forecast_horizon_hours": test_frame["forecast_horizon_hours"].to_numpy(),
            "season": [ _season_name(m) for m in test_frame.index.month ],
        }
    )
    out.to_parquet(REPORTS_DIR / "dashboard_results.parquet", index=False)

    test_metrics = results[results["split"] == "test"].to_dict(orient="records")
    metadata = {
        "project_title": "European Wind Power Forecasting & Market Analytics",
        "business_objective": (
            "Predict hourly DE-LU wind generation for the next 24 hours to support "
            "day-ahead electricity market decisions."
        ),
        "market": "DE-LU (Germany + Luxembourg)",
        "coverage": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "rows": int(len(df)),
        },
        "decision_time": "18:00 Europe/Berlin on D-1 (16:00 UTC CEST / 17:00 UTC CET)",
        "timestamp_convention": "interval_start",
        "actual_publication": "H+1 (available at interval_start + 1h)",
        "dst_handling": "23-hour and 25-hour delivery days preserved",
        "splits": summary.get("splits", {}),
        "data_quality": {
            "timezone": "UTC",
            "hourly_frequency": True,
            "duplicate_timestamps": int(df.index.duplicated().sum()),
            "negative_prices": int((df["price_eur_mwh"] < 0).sum()),
            "target_missing_rows": int(df["wind_total_actual_mw"].isna().sum()),
            "missing_values": {
                key: int(value)
                for key, value in df[
                    [
                        "price_eur_mwh",
                        "load_forecast_mw",
                        "wind_total_actual_mw",
                        "wind_total_forecast_mw",
                        "wind_onshore_actual_mw",
                        "wind_offshore_actual_mw",
                    ]
                ]
                .isna()
                .sum()
                .items()
            },
        },
        "models_test_metrics": test_metrics,
        "tso_is_best_benchmark": True,
        "nmae_note": summary.get("nmae_note", ""),
        "weather_pilot": _weather_pilot_summary(),
        "license": "CC BY 4.0",
        "attribution": (
            "Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de "
            "(CC BY 4.0). Weather: Open-Meteo.com (CC BY 4.0), ECMWF IFS HRES 9km."
        ),
        "limitations": [
            "Official TSO day-ahead forecast is the strongest benchmark; the ML "
            "models do not outperform it in this experiment.",
            "No reliable time-dependent DE-LU installed capacity is available, so "
            "installed-capacity nMAE is not reported.",
            "Day-ahead price and load forecasts are excluded because their "
            "publication times are not present in the snapshot.",
            "The 7-day weather pilot is not used for historical model training.",
            "The historical snapshot ends on 2026-09-06.",
        ],
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (REPORTS_DIR / "dashboard_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_PATH)
    df = add_calendar_features(df)
    X, y, eligible = build_feature_matrix(df)
    masks = _split_masks(df, eligible)

    # Integrity: each delivery date must belong to exactly one split.
    date_split = df.assign(_split=df["split"]).groupby("delivery_date")["_split"].nunique()
    assert (date_split == 1).all(), "a delivery date spans multiple splits"

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["validation"]], y[masks["validation"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]

    ridge = _fit_ridge(X_train.to_numpy(), y_train.to_numpy())
    lgbm = _fit_lightgbm(X_train.to_numpy(), y_train.to_numpy())

    ridge_val = ridge.predict(X_val.to_numpy())
    lgbm_val = lgbm.predict(X_val.to_numpy())
    ridge_val_mae = regression_metrics(y_val, ridge_val)["mae"]
    lgbm_val_mae = regression_metrics(y_val, lgbm_val)["mae"]

    best_name = "Ridge" if ridge_val_mae <= lgbm_val_mae else "LightGBM"
    best_model = ridge if best_name == "Ridge" else lgbm
    best_pred = best_model.predict(X_test.to_numpy())

    # Baselines and TSO on the same eligible test observations.
    baselines = build_baselines(df).loc[X_test.index]
    tso_pred = df.loc[X_test.index, TSO]

    ridge_pred = pd.Series(ridge.predict(X_test.to_numpy()), index=y_test.index)
    lgbm_pred = pd.Series(lgbm.predict(X_test.to_numpy()), index=y_test.index)
    best_pred_series = pd.Series(best_pred, index=y_test.index)

    # Common evaluation set: rows where target, every model, and the TSO
    # benchmark are all available.
    common = (
        y_test.notna()
        & ridge_pred.notna()
        & lgbm_pred.notna()
        & best_pred_series.notna()
        & baselines.notna().all(axis=1)
        & tso_pred.notna()
    )
    y_eval = y_test[common]
    base_eval = baselines[common]
    tso_eval = tso_pred[common]
    ridge_eval = ridge_pred[common]
    lgbm_eval = lgbm_pred[common]
    best_eval = best_pred_series[common]
    test_frame = df.loc[y_eval.index]

    rows: list[dict[str, Any]] = []
    persistence_mae = regression_metrics(y_eval, base_eval["persistence"])["mae"]

    rows.append(_metrics_row("Ridge", "validation", y_val, ridge_val))
    rows.append(_metrics_row("LightGBM", "validation", y_val, lgbm_val))

    rows.append(_metrics_row("persistence", "test", y_eval, base_eval["persistence"]))
    rows.append(
        _metrics_row(
            "seasonal_naive_48h", "test", y_eval, base_eval["seasonal_naive_48h"],
            persistence_mae,
        )
    )
    rows.append(
        _metrics_row(
            "seasonal_naive_168h", "test", y_eval, base_eval["seasonal_naive_168h"],
            persistence_mae,
        )
    )
    rows.append(_metrics_row("TSO_official", "test", y_eval, tso_eval, persistence_mae))
    rows.append(_metrics_row("Ridge", "test", y_eval, ridge_eval, persistence_mae))
    rows.append(_metrics_row("LightGBM", "test", y_eval, lgbm_eval, persistence_mae))

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    dump(best_model, MODEL_DIR / "best_model.joblib")

    # Charts.
    _plot_actual_vs_predicted(test_frame, y_eval, best_eval, best_name)
    _plot_model_comparison(results)
    _plot_error_distribution(test_frame, best_eval, best_name)
    _plot_error_by_group(test_frame, best_eval, best_name)

    test_rows = results[results["split"] == "test"].set_index("model")
    summary = {
        "best_model": best_name,
        "validation_mae": {"Ridge": float(ridge_val_mae), "LightGBM": float(lgbm_val_mae)},
        "splits": {
            "train": int(masks["train"].sum()),
            "validation": int(masks["validation"].sum()),
            "test": int(masks["test"].sum()),
        },
        "train_rows": int(masks["train"].sum()),
        "validation_rows": int(masks["validation"].sum()),
        "test_rows": int(masks["test"].sum()),
        "test_metrics": test_rows.reset_index().to_dict(orient="records"),
        "nmae_note": (
            "Installed-capacity nMAE is not reported because reliable DE-LU "
            "installed capacity data is unavailable. mean_normalized_mae = "
            "MAE / mean(actual generation)."
        ),
    }
    (REPORTS_DIR / "phase4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _export_dashboard_results(
        test_frame, y_eval, ridge_eval, lgbm_eval, tso_eval, results, summary, df
    )

    print("Phase 4 complete.")
    print(f"  best model (by validation MAE): {best_name}")
    print(f"  train/val/test rows: {summary['train_rows']}/{summary['validation_rows']}/{summary['test_rows']}")
    print("\nTest metrics:")
    print(test_rows[["mae", "rmse", "bias", "mean_normalized_mae"]].round(2).to_string())
    print(f"\nWrote {REPORTS_DIR / 'model_comparison.csv'}")
    return summary


if __name__ == "__main__":
    run()