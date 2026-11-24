# Data Dictionary — DE-LU Day-Ahead Snapshot

This document describes the initial historical dataset used by the
**European Wind Power Forecasting & Market Analytics** project.

> **Do not edit the raw snapshot.** It is an immutable, checksum-verified copy.
> All cleaning and feature engineering happens in later phases on copies under
> `data/processed/` and `data/features/`.

---

## 1. Source & Provenance

| Item | Value |
|---|---|
| Dataset file | `data/raw/snapshot.parquet` |
| Upstream repository | https://github.com/hrsi56/delu-day-ahead-forecast |
| Upstream file | `data/snapshot.parquet` (commit `ed755291…` on `main`) |
| Snapshot pulled (UTC) | 2026-09-08 13:34:33 |
| Delivery-date cutoff | **2026-09-06** (last complete Europe/Berlin delivery day) |
| SHA-256 | `7dd2dc73407706ca6bd3c1ad51d201ac0de35eec5ca129ce320cca639f697f00` |
| License | **CC BY 4.0** |
| Attribution | *Data: ENTSO-E Transparency Platform; Bundesnetzagentur \| SMARD.de — CC BY 4.0.* |
| Ingestion route | SMARD fallback-primary (ENTSO-E External API unavailable at pull time) |

Provenance files preserved next to the data (also git-ignored):

- `data/raw/SOURCE_README.md` — upstream `data/README.md`
- `data/raw/source_manifest.json` — upstream machine-readable manifest
- `data/raw/snapshot.sha256` — upstream published checksum

---

## 2. Coverage

| Property | Value |
|---|---|
| Rows | **67,343** |
| Columns | **22** |
| First timestamp (UTC) | `2018-12-31 23:00:00+00:00` (= 2019-01-01 00:00 Europe/Berlin) |
| Last timestamp (UTC) | `2026-09-06 21:00:00+00:00` (= 2026-09-06 23:00 Europe/Berlin) |
| Frequency | Hourly, UTC-indexed |
| Duplicate timestamps | **0** |
| Missing hours (gaps) | **0** |
| Monotonic increasing | Yes |

> **Limitation:** the snapshot **ends on 2026-09-06**. Any forecast or backtest
> that needs more recent data must wait for a newer snapshot or the live APIs
> (Phase 2). Do not treat 2026-09-06 as "today".

---

## 3. Columns

All generation and load columns are hourly-average **MW**. Prices are **EUR/MWh**.
Timestamps are **UTC**.

| Column | Type | Unit | Description | Nulls |
|---|---|---|---|---|
| `timestamp_utc` | datetime64[ms, UTC] | — | Hour start in UTC (index-like key column) | 0 |
| `delivery_date` | object (date) | — | Europe/Berlin delivery date of the hour | 0 |
| `local_hour` | int8 | hour | Hour of day in Europe/Berlin local time (0–23) | 0 |
| `utc_offset_minutes` | int16 | minutes | UTC offset: 60 (CET) or 120 (CEST); encodes DST | 0 |
| `price_eur_mwh` | float64 | EUR/MWh | Day-ahead price. Hourly product before 2025-10-01; mean of 4 quarter-hour prices after (filter 4169) | 0 |
| `load_forecast_mw` | float64 | MW | Day-ahead total load forecast, hourly average (filter 411) | 194 |
| `wind_onshore_forecast_mw` | float64 | MW | TSO day-ahead **onshore wind** forecast (filter 123) — *benchmark* | 3 |
| `solar_forecast_mw` | float64 | MW | Day-ahead solar forecast (filter 125) | 3 |
| `wind_offshore_forecast_mw` | float64 | MW | TSO day-ahead **offshore wind** forecast (filter 3791) — *benchmark* | 0 |
| `wind_onshore_actual_mw` | float64 | MW | Actual onshore wind generation (filter 4067) — *target component* | 1 |
| `solar_actual_mw` | float64 | MW | Actual solar generation (filter 4068) | 1 |
| `wind_offshore_actual_mw` | float64 | MW | Actual offshore wind generation (filter 1225) — *target component* | 0 |
| `price_source_interval_count` | int8 | count | Non-null source quarters used for price (1 or 4) | 0 |
| `load_forecast_mw_source_quarter_count` | int8 | count | Non-null source quarters (0–4); 0 ⇒ value fail-closed to null | 0 |
| `wind_onshore_forecast_mw_source_quarter_count` | int8 | count | Same, for onshore wind forecast | 0 |
| `solar_forecast_mw_source_quarter_count` | int8 | count | Same, for solar forecast | 0 |
| `wind_offshore_forecast_mw_source_quarter_count` | int8 | count | Same, for offshore wind forecast | 0 |
| `wind_onshore_actual_mw_source_quarter_count` | int8 | count | Same, for onshore wind actual | 0 |
| `solar_actual_mw_source_quarter_count` | int8 | count | Same, for solar actual | 0 |
| `wind_offshore_actual_mw_source_quarter_count` | int8 | count | Same, for offshore wind actual | 0 |
| `vre_forecast_mw` | float64 | MW | Sum of onshore + offshore + solar **forecast** | 3 |
| `vre_actual_mw` | float64 | MW | Sum of onshore + offshore + solar **actual** | 1 |

### Derived target (not stored in the file)

The project's modelling target is total wind generation:

```
wind_total_actual_mw = wind_onshore_actual_mw + wind_offshore_actual_mw
```

It is **not** a column in the snapshot; compute it during Phase 3. It has one
null hour (both wind actuals must be present).

---

## 4. Missing Observations

The snapshot is complete on the time axis (no gaps, no duplicates) but some
source values are null. Per the upstream manifest, partial source hours are
**fail-closed to null** (never partially summed).

| Column | Null rows | Notes |
|---|---|---|
| `load_forecast_mw` | 194 | Fully missing source bins (SMARD A65 path) |
| `wind_onshore_forecast_mw` | 3 | Fully missing forecast bins |
| `solar_forecast_mw` | 3 | Fully missing forecast bins |
| `vre_forecast_mw` | 3 | Consequence of the two forecast nulls above |
| `wind_onshore_actual_mw` | 1 | One partial source hour (fail-closed) |
| `solar_actual_mw` | 1 | Same hour as the onshore actual null |
| `vre_actual_mw` | 1 | Consequence of the actual nulls above |
| `price_eur_mwh` | 0 | Complete |
| `wind_offshore_*` | 0 | Complete |

The single null wind-actual hour is **2025-07-09 18:00:00 UTC**.

These nulls are expected and documented; handle them explicitly in Phase 3
(e.g., drop or impute only within training windows, never across the
forecast/valid boundary).

---

## 5. Usage Rules

1. **Immutability** — never modify `data/raw/snapshot.parquet`; verify its
   SHA-256 if in doubt.
2. **Not committed** — everything under `data/raw/` is git-ignored.
3. **No leakage** — `wind_*_forecast_mw` and `price_eur_mwh` are the values
   that were available day-ahead; use them only as they would have been known
   at issue time.
4. **Attribution** — any chart, report, or published result derived from this
   data must credit *ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de
   (CC BY 4.0)*.

---

## 6. Weather Pilot Data (Phase 2 — 7-day pilot)

This section documents the initial weather collection pilot. It is a **pipeline
validation exercise**, not the final weather dataset and not a final spatial
aggregation method.

### Source

| Item | Value |
|---|---|
| API | **Open-Meteo Single Runs API** |
| Endpoint | `https://single-runs-api.open-meteo.com/v1/forecast` |
| Documentation | https://open-meteo.com/en/docs/single-runs-api |
| Model | **ECMWF IFS HRES 9km** (`ecmwf_ifs`), 9 km O1280 grid, hourly, 15-day horizon |
| Licence | **CC BY 4.0** |
| Attribution | *Weather data: Open-Meteo.com (CC BY 4.0), ECMWF IFS HRES 9km.* |
| Request parameter | `run` = model initialisation time (UTC, ISO-8601, no seconds) |

### Pilot scope

| Item | Value |
|---|---|
| Delivery dates | **2026-08-01 → 2026-08-07** (7 days) |
| Forecast decision time | 18:00 Europe/Berlin on the preceding day = **16:00 UTC (CEST)** |
| Selected model run | **06:00 UTC on the preceding day** (conservative; see limitation) |
| Forecast lead times | **16–39 h** (mean 27.5 h) |
| Locations | **6** |
| Records | **1,008** (7 days × 6 locations × 24 h) |
| Missing values | **0** across all four variables |
| Raw responses | `data/raw/weather/` (42 JSON files, git-ignored) |
| Normalized data | `data/processed/weather_test/weather_pilot.parquet` / `.csv` (git-ignored) |
| Manifest | `data/processed/weather_test/manifest.json` (git-ignored) |

### Geographic grid points

These are illustrative, geographically distributed points used to validate the
pipeline. They are **not capacity-weighted**.

| Label | Description | Latitude | Longitude | Class |
|---|---|---|---|---|
| `north_germany` | Northern Germany (Schleswig-Holstein, near Kiel) | 54.30 | 10.10 | land |
| `northwest_germany` | Northwestern Germany (East Frisia, near Emden) | 53.50 | 7.30 | land |
| `northeast_germany` | Northeastern Germany (Mecklenburg-Vorpommern) | 53.90 | 12.20 | land |
| `central_germany` | Central Germany (Hessen / Thuringia) | 51.00 | 10.50 | land |
| `southern_germany` | Southern Germany (Bavaria / Baden-Wuerttemberg) | 48.50 | 11.50 | land |
| `north_sea_offshore` | North Sea offshore (German Bight) | 54.20 | 7.50 | offshore |

### Variables and units

All times are stored in **UTC**. Units are exactly as documented by Open-Meteo.

| Column | Unit | Description |
|---|---|---|
| `wind_speed_100m` | **km/h** | Wind speed at 100 m above ground |
| `wind_direction_100m` | **°** (degrees) | Meteorological wind direction (the direction wind blows *from*, clockwise from north) |
| `temperature_2m` | **°C** | Air temperature at 2 m |
| `surface_pressure` | **hPa** | Surface atmospheric pressure |

### Normalized schema (`weather_pilot.parquet`)

`delivery_date`, `location_label`, `classification`, `latitude`, `longitude`,
`model`, `init_time_utc`, `valid_time_utc`, `lead_hours`,
`wind_speed_100m`, `wind_direction_100m`, `temperature_2m`, `surface_pressure`.

- `init_time_utc` — model run initialisation (UTC).
- `valid_time_utc` — the hour the forecast refers to (UTC).
- `lead_hours` — `(valid_time_utc - init_time_utc)` in hours.

### Historical availability limitation (important)

Open-Meteo's metadata API only reports availability for the **latest** model run;
it does **not** expose historical per-run publication timestamps. Therefore we
**cannot verify** that a specific archived run was actually published before the
18:00 Europe/Berlin decision time.

To stay conservative, the pilot selects the latest run whose **estimated**
availability (initialisation + a **6-hour nominal delay**) is at or before the
decision cutoff. For a 16:00 UTC cutoff this selects the **06:00 UTC** run
(estimated available 12:00 UTC). This is a **documented methodological
assumption, not verified historical availability**. The 12:00 UTC run
(estimated available ~18:00 UTC) was deliberately not used.

### Missing data

The pilot has **no missing values** and **no duplicate timestamps**; no
interpolation was applied. Any future missing weather should be reported, not
silently filled.

---

## 7. Processed Modelling Dataset (Phase 3)

| Item | Value |
|---|---|
| File | `data/processed/de_lu_hourly.parquet` (git-ignored) |
| Rows × columns | **67,343 × 37** |
| Index | `timestamp_utc` (UTC), monotonic, hourly, 0 gaps, 0 duplicates |
| Source | `data/raw/snapshot.parquet` — **never modified** (SHA-256 `7dd2dc7…f697f00`) |
| Quality report | `data/processed/preprocessing_report.json` |
| Build script | `src/data/preprocess.py` |

### Derived columns

| Column | Definition |
|---|---|
| `wind_total_actual_mw` | `wind_onshore_actual_mw + wind_offshore_actual_mw` |
| `wind_total_forecast_mw` | `wind_onshore_forecast_mw + wind_offshore_forecast_mw` (official TSO) |
| `wind_forecast_error_mw` | `wind_total_actual_mw - wind_total_forecast_mw` (positive = under-forecast) |
| `is_target_missing` | `True` where `wind_total_actual_mw` is NaN |
| `price_is_quarter_mean` | `True` on/after the 2025-10-01 aggregation change |
| `decision_time_utc` | 18:00 Europe/Berlin on the day before delivery, in UTC |
| `wind_total_actual_mw_lag24/48/168` | Hour-specific lags, masked against `decision_time_utc` |
| `wind_actual_decision_lag1/24/48/168` | Actual wind strictly before the decision, broadcast per delivery date |
| `wind_actual_decision_roll24_mean/std` | Rolling 24 h stats as of the decision time |
| `split` | `train` / `validation` / `test` (chronological, by delivery date) |

### Missing data

- **Target:** 1 row (`2025-07-09 18:00 UTC`) is NaN and is **preserved** (no imputation).
- **`load_forecast_mw`:** 194 missing rows (0.29%). First `2022-02-21 23:00 UTC`,
  last `2024-10-26 22:00 UTC`; 16 distinct dates; longest run 48 h. By year:
  2022 = 144, 2023 = 49, 2024 = 1. Not imputed; documented.
- **Wind forecast:** 3 missing rows.

### Prices

- **2,501 negative prices (3.7%)**, min `-500.00`, max `936.28` EUR/MWh. These
  are **valid market observations and are retained**.
- **Methodology change:** before `2025-10-01` the price is an hourly product;
  from `2025-10-01` it is the mean of four quarter-hour prices. Flagged by
  `price_is_quarter_mean` (59,161 rows before, 8,182 after).

### Chronological splits

Split by **delivery date** so each delivery day stays intact (never shuffled):

| Split | Delivery dates | Rows |
|---|---|---|
| `train` | 2019-01-01 → 2022-12-31 | 35,064 |
| `validation` | 2023-01-01 → 2024-12-31 | 17,544 |
| `test` | 2025-01-01 → 2026-09-06 | 14,735 |

### Leakage prevention

At the 18:00 Europe/Berlin decision time, only observations **strictly before**
the cutoff are used. Lag features are masked so a value is used only when its
observation time precedes `decision_time_utc`; decision-time rolling features
use only data before the cutoff and are broadcast to the delivery day.

**Available at decision time:** calendar features, the official TSO day-ahead
wind forecast, the day-ahead price and load forecast, a weather forecast from a
run initialised before the cutoff, and past actual wind generation.
**Not available:** contemporaneous/future actual generation (the target) and the
realised day-ahead price.