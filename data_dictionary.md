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