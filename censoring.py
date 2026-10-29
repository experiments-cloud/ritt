"""Relative index recomputed with censoring-corrected departures (07_censoring.py, 08_ramp_sensitivity.py).

kappa = corrected / observed departures by station and calendar month (07_censoring.py), bounded
to [1, 5]. It is applied to daily departures, and the group A reference, the 30-day periods and
their validity are rebuilt with the rules of the relative demand index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MIN_VALID_DAYS, PERIOD_DAYS, REF_MIN_AGE_DAYS, REF_MIN_STATIONS

KAPPA_MAX = 5.0


def corrected_periods(p: pd.DataFrame, kap: pd.DataFrame) -> pd.DataFrame:
    """Period panel (estacion, grupo, periodo, days, r_salidas, valido) with the corrected index."""
    p = p.copy()
    p["month"] = p["fecha"].dt.to_period("M").dt.to_timestamp()
    k = kap.assign(k=(kap["corrected_departures"] / kap["service_departures"]).where(kap["service_departures"] > 0))
    p = p.merge(k[["estacion", "month", "k"]], on=["estacion", "month"], how="left")
    p["dep_c"] = p["salidas"] * p["k"].fillna(1.0).clip(1.0, KAPPA_MAX)
    open_ = ~p["cerrada"]
    mature = p["grupo"].eq("A") & open_ & (p["dia_desde_apertura"] >= REF_MIN_AGE_DAYS)
    all_a = p["grupo"].eq("A") & open_
    rm = p[mature].groupby("fecha")["dep_c"].agg(["mean", "size"])
    ra = p[all_a].groupby("fecha")["dep_c"].mean()
    ref = np.where(rm["size"].reindex(ra.index).fillna(0) >= REF_MIN_STATIONS, rm["mean"].reindex(ra.index), ra)
    p = p.merge(pd.DataFrame({"fecha": ra.index, "ref_c": ref}), on="fecha", how="left")
    v = p[open_ & p["ref_c"].gt(0)].copy()
    v["periodo"] = v["dia_desde_apertura"] // PERIOD_DAYS + 1
    per = v.groupby(["estacion", "grupo", "periodo"], as_index=False).agg(
        days=("fecha", "size"), s=("dep_c", "sum"), ref=("ref_c", "sum"))
    per["r_salidas"] = per["s"] / per["ref"]
    per["valido"] = per["days"] >= MIN_VALID_DAYS
    return per.drop(columns=["s", "ref"])


def y1(per: pd.DataFrame) -> pd.DataFrame:
    """Y1 = mean of r over periods 7-12 (at least 4 valid); NaN otherwise."""
    out = []
    for (st, g), s in per[per["valido"]].groupby(["estacion", "grupo"]):
        w = s.set_index("periodo")["r_salidas"].reindex(range(7, 13))
        out.append({"estacion": st, "grupo": g, "Y1": w.mean() if w.notna().sum() >= 4 else np.nan})
    return pd.DataFrame(out)
