# Predicting demand at new bike-sharing stations by learning from stations that are still maturing

Experiment code for the article "Predicting demand at new bike-sharing stations by learning from
stations that are still maturing" (ECOBICI expansion, Mexico City, 2022–2024). 

## Requirements

Python 3.10 or later.

```
pip install -r requirements.txt
```

PyTorch is only needed for step 6.

## Data

The experiments read processed data built from public sources: ECOBICI open trip data and GBFS
feed, the Mexico City open data portal, INEGI (DENUE, November 2022, and 2020 Census by AGEB),
OpenStreetMap, the station list of the previous ECOBICI system and the GBFS history collected by
M. Halford. The processed data are not distributed with this repository and are available from
the corresponding author on request. Set the environment variable `ECOBICI_DATA` to the folder
that holds them (default: `./data`):

| File | Content | Columns used |
| --- | --- | --- |
| `procesado/estaciones.csv` | Station catalogue | `estacion` (id), `grupo` (A, B1, B2), `apertura` (opening date), `alcaldia` (borough), `lat`, `lon` |
| `procesado/panel_periodos.parquet` | 30-day periods since opening | `estacion`, `grupo`, `apertura`, `periodo`, `inicio` (start date), `valido` (at least 15 open days), `r_salidas` (relative demand index r), `frac_vacia`, `frac_llena` (share of GBFS observations empty or full) |
| `procesado/panel_diario.parquet` | Daily panel since opening | `estacion`, `grupo`, `fecha`, `salidas` (departures), `cerrada` (closure day), `dia_desde_apertura`, `frac_vacia` |
| `procesado/variables_estacion.csv` | Predictors at opening | `estacion`, `grupo`, `apertura`, `x`, `y` (metres, EPSG:6372) and the numeric predictors of Table 3 of the article |
| `interim/viajes/*.parquet` | Cleaned monthly trips | `bici` (bicycle), `est_retiro`, `ts_retiro` (origin station and time), `est_arribo`, `ts_arribo` (destination station and time) |

Column names follow the processed data, which are in Spanish. Results are written to
`data/results` and reports to `data/reports`.

## Running

```
python run.py              # steps 1 to 8
python run.py --from 3     # from step 3 on
```

Each step can also be run on its own (`python 02_model_comparison.py`).

## Steps

| Step | Script | Article |
| --- | --- | --- |
| 1 | `01_maturation.py` | Mature demand Y1, saturation curves, age and network effects (Maturation of new stations) |
| 2 | `02_model_comparison.py` | Learners and baselines with mature-only and oracle labels, SHAP importance (Pre-opening prediction) |
| 3 | `03_ramp_aware_labels.py` | Mature-only vs ramp-aware labels, quality of provisional labels (Table 5) |
| 4 | `04_early_update.py` | Common maturation shape, early update with retrospective and prospective shapes, stock-out elasticities (Table 6) |
| 5 | `05_ablation.py` | Predictor ablation and target over periods 13–18 with mature-only and oracle labels |
| 6 | `06_graphsage.py` | GraphSAGE variants and ensemble with M3 |
| 7 | `07_censoring.py` | Inventories from trip chains and censoring-corrected targets |
| 8 | `08_ramp_sensitivity.py` | Ablation, 13–18 target and censoring correction with ramp-aware labels |

Shared code: `models.py` (data loading, models, rolling-origin validation and metrics), `ramp.py`
(chain-ladder labels available at each decision date), `censoring.py` (index recomputed with
corrected departures) and `trips.py` (DuckDB views over the trip files).

Labels used in the outputs: `mature` = mature-only labels, `oracle` = oracle labels, `ramp` =
ramp-aware labels; `M0_rule` = transit rule, `M1_voronoi` = Voronoi interpolation, `M2_ridge` =
ridge regression, `M3_lgbm` = gradient boosting (M3), `M4_sage` and `M4_sage_lt` = GraphSAGE
without and with neighbour labels.

Random seeds are fixed (LightGBM and GraphSAGE seeds 0–4, bootstrap seed 2026). Other versions of
LightGBM or PyTorch may change the last decimals.
