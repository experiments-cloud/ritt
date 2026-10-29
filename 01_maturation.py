"""Step 1. Maturation of new stations.

Sample size by group and observed periods; growth of r between periods 1-2 and 10-12 (one-sided
Wilcoxon); saturation curve r(m) = L(1 - exp(-m/tau)) fitted to periods 1 to 18 of each station,
with T80 = tau ln 5; age vs network effect with station fixed effects. Computes, for every
station, Y1 (mean of r over periods 7 to 12) and Y1_13_18 (periods 13 to 18), which are the
targets of the other experiments.

Output: data/results/station_curves.csv and data/reports/maturation.md

    python 01_maturation.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

import config
from models import write_report

MAX_FIT_PERIOD = 18
MIN_FIT_POINTS = 12
AGE_BINS = [(1, 1, "01"), (2, 2, "02"), (3, 3, "03"), (4, 6, "04-06"), (7, 12, "07-12"),
            (13, 18, "13-18"), (19, 24, "19-24")]


def saturation(m, L, tau):
    return L * (1 - np.exp(-m / tau))


def fit_curve(g: pd.DataFrame) -> dict:
    g = g[g["valido"] & (g["periodo"] <= MAX_FIT_PERIOD)]
    out = {"n_fit_periods": len(g), "L": np.nan, "tau": np.nan, "T80_months": np.nan, "r2": np.nan}
    if len(g) < MIN_FIT_POINTS:
        return out
    m, r = g["periodo"].to_numpy(float), g["r_salidas"].to_numpy(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (L, tau), _ = curve_fit(saturation, m, r, p0=[max(r.max(), 0.05), 2.0],
                                    bounds=([0, 0.05], [50, 60]), maxfev=10000)
    except (RuntimeError, ValueError):
        return out
    pred = saturation(m, L, tau)
    ss_res, ss_tot = ((r - pred) ** 2).sum(), ((r - r.mean()) ** 2).sum()
    out.update(L=L, tau=tau, T80_months=1.609 * tau * 30 / 30.44, r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return out


def table(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(floatfmt=".3g", **kw)


def open_neighbours(per: pd.DataFrame, cat: pd.DataFrame, radius_m: float = 1000) -> pd.Series:
    """Number of stations (any group) open within radius_m at the start of each period."""
    c = cat.set_index("estacion")
    lat0 = np.radians(c["lat"].mean())
    x, y = c["lon"].to_numpy() * 111_320 * np.cos(lat0), c["lat"].to_numpy() * 110_540
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(dist, np.inf)
    pos = {e: i for i, e in enumerate(c.index)}
    opening = c["apertura"].to_numpy()
    out = pd.Series(np.nan, index=per.index)
    for st, g in per.groupby("estacion"):
        if st not in pos:
            continue
        near = np.sort(opening[dist[pos[st]] <= radius_m])
        out.loc[g.index] = np.searchsorted(near, g["inicio"].to_numpy(), side="right")
    return out


def age_vs_network(per: pd.DataFrame, cat: pd.DataFrame) -> list[str]:
    """Is late growth of r the station's own maturation or a network effect?
    Model a: log r ~ age + station fixed effect.
    Model b: the same + log(1 + stations open within 1 km), which varies with other openings."""
    import statsmodels.formula.api as smf
    d = per[per["valido"] & (per["periodo"] <= 24) & per["grupo"].isin(["B1", "B2"]) & (per["r_salidas"] > 0)].copy()
    d["neighbours_1km"] = open_neighbours(d, cat)
    d["age"] = None
    for a, b, label in AGE_BINS:
        d.loc[d["periodo"].between(a, b), "age"] = label
    d["log_r"] = np.log(d["r_salidas"])
    d["log_neighbours"] = np.log1p(d["neighbours_1km"])
    formula = "log_r ~ C(age, Treatment('07-12')) + C(estacion)"
    rows, net = [], {}
    for group in ("B1", "B2"):
        s = d[d["grupo"] == group].dropna(subset=["log_neighbours"])
        if s["estacion"].nunique() < 10:
            continue
        clusters = s["estacion"].astype("category").cat.codes
        for name, f in (("a: age only", formula), ("b: age + network", formula + " + log_neighbours")):
            m = smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": clusters})
            row = {"grupo": group, "model": name}
            for _, _, label in AGE_BINS:
                if label == "07-12":
                    continue
                k = f"C(age, Treatment('07-12'))[T.{label}]"
                row[label] = 100 * (np.exp(m.params[k]) - 1) if k in m.params else np.nan
            rows.append(row)
            if "log_neighbours" in m.params:
                net[group] = (m.params["log_neighbours"], m.bse["log_neighbours"], m.pvalues["log_neighbours"])
    if not rows:
        return ["Not enough stations for this analysis."]
    t = pd.DataFrame(rows).set_index(["grupo", "model"])
    r = ["## 5. Own maturation vs network effect",
         "Percentage difference of r in each age bin relative to periods 7-12 (station fixed effects, errors "
         "clustered by station). If bins 13-24 move towards 0 when the network term is added, late growth comes "
         "from neighbouring stations that opened later rather than from age.", "",
         table(t), ""]
    for g, (b, se, p) in net.items():
        r.append(f"- {g}: elasticity of r to (1 + open stations within 1 km) = {b:.3f} (SE {se:.3f}, p = {p:.3g}).")
    return r + [""]


def main():
    config.ensure_dirs()
    cat = pd.read_csv(config.STATIONS, dtype={"estacion": str}, parse_dates=["apertura"])
    cat = cat[cat["grupo"].isin(["A", "B1", "B2"]) & cat["apertura"].notna()]
    per = pd.read_parquet(config.PERIOD_PANEL)
    per["estacion"] = per["estacion"].astype(str)

    rows = []
    for (st, group), g in per.groupby(["estacion", "grupo"]):
        val = g[g["valido"]].set_index("periodo")["r_salidas"]
        row = {"estacion": st, "grupo": group, "n_valid_periods": len(val),
               "r_start": val.reindex([1, 2]).mean(), "r_10_12": val.reindex([10, 11, 12]).mean(),
               "Y1": val.reindex(range(7, 13)).mean() if val.reindex(range(7, 13)).notna().sum() >= 4 else np.nan,
               "Y1_13_18": val.reindex(range(13, 19)).mean() if val.reindex(range(13, 19)).notna().sum() >= 4 else np.nan,
               "mean_empty_share": g["frac_vacia"].mean() if "frac_vacia" in g else np.nan,
               "mean_full_share": g["frac_llena"].mean() if "frac_llena" in g else np.nan}
        row.update(fit_curve(g))
        rows.append(row)
    res = pd.DataFrame(rows).merge(cat[["estacion", "alcaldia", "apertura", "lat", "lon"]], on="estacion")
    res.to_csv(config.STATION_CURVES, index=False)

    r = ["# Maturation of new stations", ""]
    exp = res[res["grupo"].isin(["B1", "B2"])]

    size = res.groupby("grupo").agg(stations=("estacion", "size"),
                                    with_12_periods=("n_valid_periods", lambda s: (s >= 12).sum()),
                                    with_18_periods=("n_valid_periods", lambda s: (s >= 18).sum()),
                                    with_24_periods=("n_valid_periods", lambda s: (s >= 24).sum()))
    r += ["## 1. Sample size", table(size), ""]

    r += ["## 2. Growth within stations",
          "Paired comparison of r in periods 1-2 and 10-12 (one-sided Wilcoxon signed-rank test: r grows).", ""]
    tests = []
    for g in ("A", "B1", "B2"):
        s = res[(res["grupo"] == g)].dropna(subset=["r_start", "r_10_12"])
        if len(s) < 5:
            continue
        ratio = (s["r_10_12"] / s["r_start"].replace(0, np.nan)).median()
        p = stats.wilcoxon(s["r_10_12"], s["r_start"], alternative="greater").pvalue
        tests.append({"grupo": g, "n": len(s), "median_r_start": s["r_start"].median(),
                      "median_r_10_12": s["r_10_12"].median(), "median_ratio": ratio,
                      "pct_growing": 100 * (s["r_10_12"] > s["r_start"]).mean(), "p_value": p})
    tt = pd.DataFrame(tests).set_index("grupo") if tests else pd.DataFrame()
    r += [table(tt) if len(tt) else "Not enough data.", ""]

    ok = exp.dropna(subset=["tau"])
    fits = exp.groupby("grupo").agg(stations=("estacion", "size"),
                                    fitted=("tau", lambda s: s.notna().sum()),
                                    median_r2=("r2", "median"),
                                    median_L=("L", "median"),
                                    median_T80_months=("T80_months", "median"),
                                    T80_p25=("T80_months", lambda s: s.quantile(0.25)),
                                    T80_p75=("T80_months", lambda s: s.quantile(0.75)),
                                    median_Y1=("Y1", "median"))
    r += ["## 3. Saturation curve by station (expansion)", table(fits), ""]
    if {"B1", "B2"} <= set(ok["grupo"]):
        mw = stats.mannwhitneyu(ok.loc[ok["grupo"] == "B2", "T80_months"], ok.loc[ok["grupo"] == "B1", "T80_months"],
                                alternative="greater")
        r += [f"T80 larger in B2 than in B1, one-sided Mann-Whitney U: p = {mw.pvalue:.3g}.", ""]
    at_bound = ok[(ok["tau"] >= 59) | (ok["L"] >= 49)]
    r += [f"Fits at the parameter bounds (no saturation within 18 periods): {len(at_bound)}.", ""]

    r += ["## 4. Stock-outs (GBFS, from 2024-08)"]
    if res["mean_empty_share"].notna().any():
        cens = res.groupby("grupo")[["mean_empty_share", "mean_full_share"]].median().mul(100)
        r += ["Median by station of the share of observations with the station empty or full.", "",
              table(cens.rename(columns={"mean_empty_share": "% empty", "mean_full_share": "% full"})), ""]
    else:
        r += ["The period panel has no GBFS columns (frac_vacia, frac_llena).", ""]

    r += age_vs_network(per, cat)
    write_report("maturation.md", r)


if __name__ == "__main__":
    main()
