# European Wind Power Forecasting & Market Analytics

> Hourly wind power generation forecasting for the **DE-LU** (Germany + Luxembourg)
> bidding zone, built end-to-end as a reproducible Data Science project.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Business Problem

Electricity traders at an energy company need accurate short-term predictions of
wind power generation to make better decisions in the day-ahead electricity market.
This project builds a Machine Learning system that:

1. Predicts **hourly wind power generation** for the DE-LU zone for the next 24 hours.
2. Compares predictions against **actual** generation.
3. Benchmarks our models against the **official ENTSO-E TSO day-ahead wind forecast**.
4. Analyzes how **wind forecast errors relate to day-ahead electricity prices**.

---

## Tech Stack

| Area            | Tools                                             |
|-----------------|---------------------------------------------------|
| Language        | Python 3.11 / 3.12                                 |
| Data handling   | pandas, NumPy                                     |
| Visualization   | Matplotlib, Seaborn, Plotly                       |
| Classical ML    | scikit-learn                                      |
| Gradient Boost  | XGBoost, LightGBM                                 |
| Data sources    | ENTSO-E Transparency Platform API, Open-Meteo API |
| Dashboard       | Streamlit                                         |
| Version control | Git, GitHub                                       |

---

## Project Structure

```
European-Wind-Power-Forecasting/
├── .env.example            # Template for secrets (copy to .env)
├── .gitignore
├── requirements.txt        # Python dependencies
├── requirements-lock.txt   # Exact pinned versions (pip freeze)
├── data_dictionary.md      # Dataset schema, coverage & quality
├── README.md
├── data/
│   ├── raw/                # Raw API responses (not committed)
│   ├── processed/          # Cleaned, merged datasets
│   └── features/           # Feature-engineered datasets
├── notebooks/              # Exploratory notebooks (EDA, experiments)
├── src/
│   ├── config.py           # Central configuration (paths, constants)
│   ├── data/               # Data collection & loading
│   ├── features/           # Feature engineering
│   ├── models/             # Training & evaluation
│   ├── backtest/           # Market backtesting
│   └── dashboard/          # Streamlit app
├── models/                 # Serialized trained models
├── reports/figures/        # Generated plots and reports
└── tests/                  # Unit tests
```

---

## Project Phases

| Phase | Description                                                        | Status |
|-------|--------------------------------------------------------------------|--------|
| 1     | Project setup and environment configuration                        | ✅     |
| 2     | Data collection (ENTSO-E + Open-Meteo APIs)                        | ⏳     |
| 3     | Data cleaning, preprocessing, and integration                      | ⏳     |
| 4     | Exploratory Data Analysis (EDA)                                    | ⏳     |
| 5     | Feature engineering                                                | ⏳     |
| 6     | Machine Learning models (baselines, Ridge, Random Forest, XGBoost) | ⏳     |
| 7     | Walk-forward validation and evaluation (MAE, RMSE, nMAE)           | ⏳     |
| 8     | Comparison with official TSO day-ahead forecast                    | ⏳     |
| 9     | Exploratory market backtesting (day-ahead prices, costs)           | ⏳     |
| 10    | Streamlit dashboard and professional documentation                 | ⏳     |

---

## Getting Started

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure secrets
```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
```
Then edit `.env` and add your ENTSO-E API key.

### 4. Run tests
```bash
pytest
```

---

## Data Sources

### Initial historical snapshot (current)

The project starts from a committed, checksum-verified historical snapshot so
work can begin without waiting for API authentication.

| Item | Value |
|---|---|
| File | `data/raw/snapshot.parquet` (git-ignored, not committed) |
| Source repo | https://github.com/hrsi56/delu-day-ahead-forecast |
| Coverage | 2019-01-01 → **2026-09-06** (Europe/Berlin delivery days), hourly UTC |
| Rows × columns | 67,343 × 22 |
| SHA-256 | `7dd2dc73407706ca6bd3c1ad51d201ac0de35eec5ca129ce320cca639f697f00` |
| License | **CC BY 4.0** |
| Attribution | *Data: ENTSO-E Transparency Platform; Bundesnetzagentur \| SMARD.de — CC BY 4.0.* |

> The snapshot **ends on 2026-09-06**; it is a historical training source, not
> a live feed. Full column/quality details are in
> [`data_dictionary.md`](data_dictionary.md).

### Weather pilot (Phase 2, in progress)

A 7-day pilot validates the weather-collection pipeline before scaling up.

| Item | Value |
|---|---|
| Source | **Open-Meteo Single Runs API** (`single-runs-api.open-meteo.com/v1/forecast`) |
| Model | **ECMWF IFS HRES 9km** (`ecmwf_ifs`) |
| Variables | `wind_speed_100m` (km/h), `wind_direction_100m` (°), `temperature_2m` (°C), `surface_pressure` (hPa) |
| Pilot delivery dates | 2026-08-01 → 2026-08-07 |
| Decision time | 18:00 Europe/Berlin on the preceding day (16:00 UTC in CEST) |
| Selected run | 06:00 UTC on the preceding day (conservative; availability not verifiable) |
| Lead times | 16–39 h (mean 27.5 h) |
| Locations | 6 (N, NW, NE, central, S Germany + North Sea offshore) |
| Records | 1,008 (7 × 6 × 24), 0 missing |
| License | **CC BY 4.0** |
| Attribution | *Weather data: Open-Meteo.com (CC BY 4.0), ECMWF IFS HRES 9km.* |

> The selected run's historical publication time **cannot be verified**; the
> 06:00 UTC choice is a documented conservative assumption. See
> [`data_dictionary.md`](data_dictionary.md) §6 for the full rationale.

### Upcoming / complementary sources

- **ENTSO-E Transparency Platform** — live actual wind generation, day-ahead
  prices, and the official TSO day-ahead wind forecast. Requires a free API key
  (configured in `.env` during Phase 2).
- **Open-Meteo** — historical numerical weather prediction (wind speed at 100 m,
  direction, temperature, surface pressure). Free, no API key required. The
  pilot above is the first step toward the full historical weather dataset.

---

## Dataset License & Attribution

The historical snapshot is derived from ENTSO-E Transparency Platform and
Bundesnetzagentur | SMARD.de data and is redistributed under **CC BY 4.0**.
Attribution is required for any reuse:

> Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de — CC BY 4.0.

The project's own source code is licensed under the MIT License (see
[LICENSE](LICENSE)).

---

## Data Preparation & EDA

The historical electricity snapshot is the primary modelling dataset.

| Step | Output |
|---|---|
| Preprocessing (`src/data/preprocess.py`) | `data/processed/de_lu_hourly.parquet` + `preprocessing_report.json` |
| EDA (`notebooks/01_eda.ipynb`, executed) | `reports/figures/` (9 charts) |
| Chronological splits | train 2019–2022 (35,064) · validation 2023–2024 (17,544) · test 2025–2026-09-06 (14,735) |

Key EDA findings:

- Strong seasonality: winter mean ≈ 20,900 MW vs summer ≈ 9,800 MW.
- Onshore ≈ 80.6% of wind generation, offshore ≈ 19.4%.
- Official TSO forecast vs actual: MAE ≈ 1,301 MW, RMSE ≈ 1,827 MW, nMAE ≈ 0.088.
- Wind vs day-ahead price correlation ≈ −0.26 (strongest in winter, −0.40).
- Daily wind ACF: lag-1 ≈ 0.68, lag-7 ≈ 0.23, lag-365 ≈ 0.21.

The processed dataset and figures are git-ignored; only code and docs are
committed.

---

## Methodology Notes

- **Time zones:** All timestamps are stored in **UTC** internally and converted to
  Europe/Berlin only for reporting. This avoids daylight-saving-time (DST) bugs.
- **No data leakage:** Features only use information available **at forecast issue
  time** (e.g., day-ahead weather forecasts, calendar features, past generation).
- **Validation:** **Walk-forward (expanding window)** validation that respects the
  time order of the data.
- **Metrics:** MAE, RMSE, and normalized MAE (nMAE = MAE / mean generation).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).