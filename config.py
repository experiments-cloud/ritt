"""Paths and parameters shared by the experiment scripts.

The experiments read the processed ECOBICI data (station catalogue, daily and 30-day panels,
station predictors and cleaned monthly trip files). Set the environment variable ECOBICI_DATA to
the folder that holds them; by default it is ./data next to the scripts.
"""
from pathlib import Path
import os

DATA = Path(os.environ.get("ECOBICI_DATA", Path(__file__).resolve().parent / "data"))

# Inputs: processed data (see README for the expected columns)
PROCESSED = DATA / "procesado"
STATIONS = PROCESSED / "estaciones.csv"
DAILY_PANEL = PROCESSED / "panel_diario.parquet"
PERIOD_PANEL = PROCESSED / "panel_periodos.parquet"
PREDICTORS = PROCESSED / "variables_estacion.csv"
TRIPS = DATA / "interim" / "viajes"            # cleaned monthly trip files (Parquet)

# Outputs
RESULTS = DATA / "results"
REPORTS = DATA / "reports"
STATION_CURVES = RESULTS / "station_curves.csv"

# Relative demand index (as built by the data preparation)
REF_MIN_AGE_DAYS = 90      # minimum age of a group A station to enter the daily reference
REF_MIN_STATIONS = 30      # with fewer mature A stations, all open A stations are used
PERIOD_DAYS = 30
MIN_VALID_DAYS = 15        # open days for a period to count

# Start of the new system and last day of the previous one (used to assign trips to a system)
NEW_SYSTEM_START = "2022-08-13"
OLD_SYSTEM_END = "2022-10-29"


def ensure_dirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
