"""Step 7. Correction of censored demand with trip chains.

The GBFS history starts in 2024-08, too late for periods 7-12 of almost every expansion station,
so the inventory of each station is rebuilt from the trips themselves:

  Chain of a bicycle: it arrives at station s at t0 and its next trip departs at t1.
  - If the next trip departs from s, the bicycle was docked at s during [t0, t1]  (certain presence).
  - If it departs from another station, rebalancing moved it: it stayed at s for an unknown time
    after t0 and at the departure station for an unknown time before t1          (possible presence).
  Intervals are capped at 24 h (a bicycle docked for longer may be out of service).

  Minimum inventory = certain presences only        -> UPPER bound of the time empty (larger correction).
  Maximum inventory = certain plus possible ones    -> LOWER bound of the time empty (smaller correction).

By station, calendar month and hour of day (05:00 to 24:00), corrected demand = departures / share
of time with bicycles (at least 0.2, i.e. a maximum factor of 5). The ratio corrected / observed
(kappa) by station and month is applied to daily departures, and the group A reference, the index
r and Y1 are recomputed (censoring.py).

Validation: share of the day empty by station-day against the GBFS where both are available.
Evaluation: the models of step 2 with the corrected Y1 (both bounds), mature and oracle labels.

Output: data/results/kappa_censoring.parquet, censoring_curves.csv, censoring_metrics.csv
        and data/reports/censoring.md

    python 07_censoring.py
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import censoring
import config
import models
from trips import create_views

SERVICE_HOURS = range(5, 24)
CAP_H = 24
MIN_AVAILABILITY = 1 / censoring.KAPPA_MAX


def inventory(con, bound: str):
    """Table empty_<bound>(est, h, sec): seconds empty by station and clock hour."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE chain AS
        SELECT est_arribo AS est, ts_arribo AS t0,
               lead(est_retiro) OVER w AS est_next, lead(ts_retiro) OVER w AS t1
        FROM trips_new
        WHERE bici IS NOT NULL AND ts_arribo IS NOT NULL AND est_arribo IS NOT NULL
        WINDOW w AS (PARTITION BY bici ORDER BY ts_retiro)
    """)
    certain = f"""SELECT est, t0 AS a, least(t1, t0 + INTERVAL {CAP_H} HOUR) AS b
                  FROM chain WHERE est = est_next AND t1 > t0"""
    possible = f"""SELECT est, t0 AS a, least(t1, t0 + INTERVAL {CAP_H} HOUR) AS b
                   FROM chain WHERE est_next IS NOT NULL AND est <> est_next AND t1 > t0
                   UNION ALL
                   SELECT est_next AS est, greatest(t0, t1 - INTERVAL {CAP_H} HOUR) AS a, t1 AS b
                   FROM chain WHERE est_next IS NOT NULL AND est <> est_next AND t1 > t0"""
    source = certain if bound == "upper" else f"{certain} UNION ALL {possible}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE empty_{bound} AS
        WITH iv AS ({source}),
        ev AS (SELECT est, a AS ts, 1 AS d FROM iv UNION ALL SELECT est, b AS ts, -1 AS d FROM iv),
        inv AS (SELECT est, ts, sum(d) OVER (PARTITION BY est ORDER BY ts, d DESC
                                            ROWS UNBOUNDED PRECEDING) AS n,
                       lead(ts) OVER (PARTITION BY est ORDER BY ts, d DESC) AS ts_next
                FROM ev),
        zero AS (SELECT est, ts AS a, ts_next AS b FROM inv WHERE n <= 0 AND ts_next > ts)
        SELECT est, h, sum(epoch(least(b, h + INTERVAL 1 HOUR) - greatest(a, h))) AS sec
        FROM zero, LATERAL (SELECT unnest(generate_series(date_trunc('hour', a), b, INTERVAL 1 HOUR)) AS h)
        WHERE h < b
        GROUP BY est, h
    """)


def kappa(con, bound: str, p: pd.DataFrame) -> pd.DataFrame:
    """Correction factor by station and calendar month (service hours, open days)."""
    hours = ",".join(str(h) for h in SERVICE_HOURS)
    hourly = con.execute(f"""
        WITH dep AS (SELECT est_retiro AS est, date_trunc('hour', ts_retiro) AS h, count(*) AS n
                     FROM trips_new WHERE est_retiro IS NOT NULL GROUP BY 1, 2),
        base AS (SELECT coalesce(s.est, v.est) AS est, coalesce(s.h, v.h) AS h,
                        coalesce(s.n, 0) AS n, least(coalesce(v.sec, 0), 3600) / 3600.0 AS he
                 FROM dep s FULL OUTER JOIN empty_{bound} v ON v.est = s.est AND v.h = s.h)
        SELECT est AS estacion, h::DATE AS fecha, hour(h) AS hd, n, he FROM base
        WHERE hour(h) IN ({hours})
    """).df()
    hourly["fecha"] = pd.to_datetime(hourly["fecha"])
    open_days = p.loc[~p["cerrada"], ["estacion", "fecha"]]
    hourly = hourly.merge(open_days, on=["estacion", "fecha"])          # open days since opening only
    hourly["month"] = hourly["fecha"].dt.to_period("M").dt.to_timestamp()
    g = hourly.groupby(["estacion", "month", "hd"], as_index=False).agg(n=("n", "sum"), he=("he", "sum"))
    days = open_days.assign(month=open_days["fecha"].dt.to_period("M").dt.to_timestamp()) \
        .groupby(["estacion", "month"]).size().rename("days").reset_index()
    g = g.merge(days, on=["estacion", "month"])
    g["availability"] = (1 - g["he"] / g["days"]).clip(lower=MIN_AVAILABILITY)
    g["n_c"] = g["n"] / g["availability"]
    k = g.groupby(["estacion", "month"]).agg(service_departures=("n", "sum"), corrected_departures=("n_c", "sum"),
                                             he=("he", "sum"), days=("days", "first")).reset_index()
    k["empty_share_service"] = k["he"] / (k["days"] * len(SERVICE_HOURS))
    return k.drop(columns=["he", "days"]).assign(bound=bound)


def daily_empty_share(con, bound: str) -> pd.DataFrame:
    return con.execute(f"""SELECT est AS estacion, h::DATE AS fecha, sum(least(sec, 3600)) / 86400.0 AS share
                           FROM empty_{bound} GROUP BY 1, 2""").df()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", default=None, help="DuckDB memory limit, e.g. 12GB")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--min-test", type=int, default=5)
    args = ap.parse_args()
    config.ensure_dirs()
    if not any(config.TRIPS.glob("*.parquet")):
        print(f"No trip files in {config.TRIPS}: the censoring correction needs the cleaned trips "
              "(available from the authors on request). Step skipped.")
        return
    cat = pd.read_csv(config.STATIONS, dtype={"estacion": str}, parse_dates=["apertura"])
    cat = cat[cat["grupo"].isin(["A", "B1", "B2"]) & cat["apertura"].notna()]
    con = duckdb.connect()
    if args.memory:
        con.execute(f"SET memory_limit = '{args.memory}'")
    tmp = config.DATA / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    create_views(con, extra_pairs=list(cat["estacion"]))
    p = pd.read_parquet(config.DAILY_PANEL)
    p["estacion"] = p["estacion"].astype(str)
    p["fecha"] = pd.to_datetime(p["fecha"])

    kaps, curves, valid = [], [], []
    for bound in ("lower", "upper"):
        print(f"Inventory from trip chains, {bound} bound of the time empty...", flush=True)
        inventory(con, bound)
        k = kappa(con, bound, p)
        kaps.append(k)
        c = censoring.y1(censoring.corrected_periods(p, k)).rename(columns={"Y1": f"Y1_c_{bound}"})
        curves.append(c.set_index(["estacion", "grupo"]))
        f = daily_empty_share(con, bound)
        f["fecha"] = pd.to_datetime(f["fecha"])
        if "frac_vacia" in p:
            m = p.loc[p["frac_vacia"].notna() & ~p["cerrada"], ["estacion", "fecha", "grupo", "frac_vacia"]] \
                .merge(f, on=["estacion", "fecha"], how="left").fillna({"share": 0.0})
            if len(m) > 100:
                valid.append({"bound": bound, "station_days": len(m),
                              "GBFS_mean": m["frac_vacia"].mean(), "chains_mean": m["share"].mean(),
                              "pearson_day": m["frac_vacia"].corr(m["share"]),
                              "spearman_station": stats.spearmanr(
                                  m.groupby("estacion")["frac_vacia"].mean(),
                                  m.groupby("estacion")["share"].mean()).statistic})
    kap = pd.concat(kaps, ignore_index=True)
    kap.to_parquet(config.RESULTS / "kappa_censoring.parquet", index=False)
    cur = models.read_curves()[["estacion", "grupo", "Y1"]]
    cur = cur.merge(pd.concat(curves, axis=1).reset_index(), on=["estacion", "grupo"], how="left")
    cur.to_csv(config.RESULTS / "censoring_curves.csv", index=False)

    r = ["# Correction of censored demand with trip chains", "",
         "Lower bound: inventory with certain and possible presences (smallest correction). Upper bound: certain "
         "presences only (largest correction). Service hours 05:00-24:00.", ""]
    if valid:
        r += ["## 1. Validation against GBFS (share of the day with the station empty)",
              pd.DataFrame(valid).set_index("bound").to_markdown(floatfmt=".3f"),
              "", "If the GBFS value lies between the two bounds, the correction interval is credible.", ""]
    k = kap.merge(cat[["estacion", "grupo"]], on="estacion")
    k["kappa"] = k["corrected_departures"] / k["service_departures"]
    t = k.groupby(["bound", "grupo"]).agg(median_empty_share=("empty_share_service", "median"),
                                          median_kappa=("kappa", "median"),
                                          kappa_p90=("kappa", lambda x: x.quantile(0.9))).unstack(0)
    r += ["## 2. Time empty in service hours and correction factor (station-month)",
          t.to_markdown(floatfmt=".3f"), ""]
    rows = []
    for g, s in cur.groupby("grupo"):
        for bound in ("lower", "upper"):
            ss = s.dropna(subset=["Y1", f"Y1_c_{bound}"])
            if len(ss) >= 5:
                rows.append({"grupo": g, "bound": bound, "n": len(ss), "median_Y1": ss["Y1"].median(),
                             "median_corrected_Y1": ss[f"Y1_c_{bound}"].median(),
                             "spearman_Y1_vs_corrected": stats.spearmanr(ss["Y1"], ss[f"Y1_c_{bound}"]).statistic})
    r += ["## 3. Observed vs corrected Y1",
          "The corrected Y1 is still relative to the corrected group A: if every station loses demand alike, Y1 "
          "does not change; it changes when a station is empty more often than the average.", "",
          pd.DataFrame(rows).set_index(["grupo", "bound"]).to_markdown(floatfmt=".3f"), ""]

    mets = []
    for target in ("Y1", "Y1_c_lower", "Y1_c_upper"):
        d = models.load(target, curves=cur)
        preds = pd.concat([models.rolling_origin(d, prot, args.min_train, args.min_test)[0]
                           for prot in models.PROTOCOLS], ignore_index=True)
        if preds.empty:
            continue
        m = models.metrics(preds, args.bootstrap, np.random.default_rng(2026)).assign(target=target)
        med = d.loc[d["grupo"].isin(["B1", "B2"]), "Y1"].median()
        m["relative_MAE"] = m["MAE"] / med
        mets.append(m)
        print(f"  models with {target}: done", flush=True)
    met = pd.concat(mets, ignore_index=True)
    met.to_csv(config.RESULTS / "censoring_metrics.csv", index=False)
    s = met[met["grupo"] == "B1+B2"].pivot_table(index=["protocol", "model"], columns="target",
                                                 values=["relative_MAE", "spearman", "capture20_of_max"])
    r += ["## 4. Models with the corrected target (B1+B2)",
          "relative_MAE = MAE / median Y1 of the expansion, to compare targets on different scales.", "",
          s.round(3).to_markdown(), ""]
    models.write_report("censoring.md", r)


if __name__ == "__main__":
    main()
