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

- **ENTSO-E Transparency Platform** — actual wind generation, day-ahead prices,
  and the official TSO day-ahead wind forecast. Requires a free API key.
- **Open-Meteo** — historical numerical weather prediction (wind speed at 100 m,
  direction, temperature, surface pressure). Free, no API key required.

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