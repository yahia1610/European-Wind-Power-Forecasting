# DE-LU Day-Ahead Snapshot — Source Documentation

This folder preserves the **provenance** of the initial historical dataset used
by the European Wind Power Forecasting project. The dataset itself is **not**
committed here (or anywhere in Git); only its documentation is tracked.

- Canonical dataset description: [`../../../data_dictionary.md`](../../../data_dictionary.md)

---

## Source

| Item | Value |
|---|---|
| Dataset | DE-LU hourly day-ahead snapshot (`snapshot.parquet`) |
| Upstream repository | https://github.com/hrsi56/delu-day-ahead-forecast |
| Upstream data file | `data/snapshot.parquet` (commit `ed755291…` on `main`) |
| Snapshot pulled (UTC) | 2026-09-08 13:34:33 |
| Coverage | 2019-01-01 → **2026-09-06** (Europe/Berlin delivery days), hourly UTC |
| Rows × columns | 67,343 × 22 |
| Ingestion route | SMARD fallback-primary (ENTSO-E External API unavailable at pull time) |

## License & Attribution

**License: CC BY 4.0**

> Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de — CC BY 4.0.

Attribution is required for any reuse or redistribution of derived data or
figures.

## Integrity

SHA-256 of the dataset file (`data/raw/snapshot.parquet`):

```
7dd2dc73407706ca6bd3c1ad51d201ac0de35eec5ca129ce320cca639f697f00  snapshot.parquet
```

Verify locally with:

```powershell
Get-FileHash -Algorithm SHA256 data\raw\snapshot.parquet
```

## Preserved Files

| File | Description |
|---|---|
| `SOURCE_README.md` | Verbatim copy of the upstream `data/README.md` |
| `source_manifest.json` | Upstream machine-readable manifest (coverage, nulls, mapping) |
| `snapshot.sha256` | Upstream published checksum |
| `README.md` | This document |

## Known Limitation

The snapshot **ends on 2026-09-06**. It is a historical training source, not a
live feed; more recent data must come from the live APIs in Phase 2.