"""Step 4. Shape of maturation, forecast of the monthly trajectory and early update after
opening, and sensitivity of the index to empty or full stations.

1. Common shape. log r ~ age bin + age bin x B2 + station fixed effect, errors clustered by
   station. The joint Wald test of the interactions tells whether B2 matures with a different
   shape from B1 (not only a different level).
2. Trajectory and early update, with two estimates of the shape g(m):
     - retrospective: median of r_m / Y1 over the expansion stations of the other quarters
       (earlier and later than the station's own);
     - prospective: the shape implied by the development factors available at the start of the
       opening quarter (ramp.shape, expansion triangle), with no later information.
   With each shape:
     - the predicted trajectory is r_m = Y1_hat x g(m), compared with a flat trajectory at Y1_hat;
     - after k observed periods, Y1 is estimated as sum(r_1..k) / sum(g_1..k) and blended in logs
       with the pre-opening prediction of M3. With the retrospective shape the weight is chosen on
       the other quarters; with the prospective shape it is fixed at 0.5, because the final Y1 of
       stations from earlier quarters is not yet known at the decision date.
   The prior comes from M3 trained with mature or oracle labels (step 2) and, when step 3 has
   been run, with ramp-aware labels.
3. Stock-outs. Elasticity of r to the share of time empty or full (station and month fixed
   effects) and correlation of the error of M3 with the mean share of time empty of each station.

Requires steps 1 and 2 (and optionally step 3).
Output: data/results/early_update.csv and data/reports/early_update.md

    python 04_early_update.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config
import models
import ramp

warnings.filterwarnings("ignore")
AGE_BINS = [(1, 1, "01"), (2, 2, "02"), (3, 3, "03"), (4, 6, "04-06"), (7, 12, "07-12"), (13, 18, "13-18"),
            (19, 24, "19-24")]
KS = (1, 2, 3, 6)
WEIGHTS = np.linspace(0, 1, 21)
FIXED_WEIGHT = 0.5
PRIOR_PROTOCOLS = ("mature", "oracle", "ramp")


def table(df, **kw):
    return df.to_markdown(floatfmt=kw.pop("floatfmt", ".3g"), **kw)


def load():
    per = pd.read_parquet(config.PERIOD_PANEL)
    per["estacion"] = per["estacion"].astype(str)
    per_exp = per[per["grupo"].isin(["B1", "B2"])].copy()
    curves = models.read_curves()
    curves = curves[curves["Y1"].notna() & (curves["Y1"] > 0)]
    per_exp = per_exp.merge(curves[["estacion", "Y1"]], on="estacion")
    per_exp["quarter"] = pd.to_datetime(per_exp["apertura"]).dt.to_period("Q").astype(str)
    return per, per_exp, curves


# ---------------- 1. common shape ----------------

def common_shape(per) -> list[str]:
    import statsmodels.formula.api as smf
    d = per[per["valido"] & (per["periodo"] <= 24) & (per["r_salidas"] > 0)].copy()
    d["age"] = None
    for a, b, label in AGE_BINS:
        d.loc[d["periodo"].between(a, b), "age"] = label
    d["log_r"] = np.log(d["r_salidas"])
    d["b2"] = (d["grupo"] == "B2").astype(float)
    f = "log_r ~ C(age, Treatment('07-12')) * b2 + C(estacion)"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["estacion"].astype("category").cat.codes})
    names = list(m.params.index)
    inter = [i for i, n in enumerate(names) if n.endswith(":b2") and "[T." in n]
    R = np.zeros((len(inter), len(names)))
    for row, i in enumerate(inter):
        R[row, i] = 1
    w = m.wald_test(R, scalar=True)
    rows = []
    for _, _, label in AGE_BINS:
        if label == "07-12":
            continue
        k = f"C(age, Treatment('07-12'))[T.{label}]"
        ki = f"{k}:b2"
        b1 = m.params.get(k, np.nan)
        diff = m.params.get(ki, np.nan)
        rows.append({"age bin": label, "B1 (%)": 100 * (np.exp(b1) - 1), "B2 (%)": 100 * (np.exp(b1 + diff) - 1),
                     "B2 - B1 (log)": diff, "p": m.pvalues.get(ki, np.nan)})
    t = pd.DataFrame(rows).set_index("age bin")
    return ["## 1. Shape of maturation in B1 and B2",
            "Percentage difference of r relative to periods 7-12, within each station. The last column tests "
            "whether B2 differs from B1 in that bin.", "", table(t), "",
            f"Joint Wald test of the {len(inter)} interactions: statistic = {float(w.statistic):.2f}, "
            f"p = {float(w.pvalue):.3g}.", ""]


# ---------------- 2. trajectory and early update ----------------

def shape_other_quarters(per_exp) -> dict[str, pd.Series]:
    d = per_exp[per_exp["valido"] & (per_exp["periodo"] <= 12)].copy()
    d["norm"] = d["r_salidas"] / d["Y1"]
    return {q: d[d["quarter"] != q].groupby("periodo")["norm"].median() for q in d["quarter"].unique()}


def shape_prospective(per, quarter_list) -> dict[str, pd.Series]:
    """g(m) from the expansion development factors available at the start of each quarter."""
    d = models.read_predictors()
    opening = d.set_index("estacion")["apertura"]
    cls = {e: ("A" if g == "A" else "B") for e, g in zip(d["estacion"], d["grupo"])}
    out = {}
    for q in quarter_list:
        C = ramp.cumulative(per, opening, pd.Period(q, "Q").start_time, 12)
        out[q] = ramp.shape(ramp.factors(C, cls, 12)["B"])
    return out


def early_update(per, per_exp, priors) -> tuple[list[str], pd.DataFrame]:
    g_other = shape_other_quarters(per_exp)
    g_prosp = shape_prospective(per, sorted(per_exp["quarter"].unique()))
    r_tab = per_exp[per_exp["valido"]].pivot_table(index="estacion", columns="periodo", values="r_salidas")
    rows, trajectory = [], []
    m3 = priors[priors["model"] == "M3_lgbm"]
    for _, f in m3.iterrows():
        st = str(f["estacion"])
        if st not in r_tab.index:
            continue
        q = per_exp.loc[per_exp["estacion"] == st, "quarter"].iloc[0]
        g, gp = g_other.get(q), g_prosp[q]
        if g is None or len(g) < 12:
            continue
        r = r_tab.loc[st]
        base = {"protocol": f["protocol"], "estacion": st, "grupo": f["grupo"], "quarter": q,
                "y": f["y"], "prior": f["pred"]}
        for k in KS:
            obs = r.reindex(range(1, k + 1))
            ok = obs.notna().to_numpy()
            if ok.sum() == 0:
                continue
            rows.append({**base, "k": k, "naive": obs[ok].mean(),
                         "data": obs[ok].mean() / g.reindex(range(1, k + 1))[ok].mean(),
                         "data_prosp": obs[ok].sum() / gp.reindex(range(1, k + 1))[ok].sum()})
        for m in range(1, 7):
            if pd.notna(r.get(m)):
                trajectory.append({"protocol": f["protocol"], "grupo": f["grupo"], "periodo": m, "obs": r[m],
                                   "with_shape": f["pred"] * g[m], "with_shape_prosp": f["pred"] * gp[m],
                                   "flat": f["pred"]})
    a = pd.DataFrame(rows)
    if a.empty:
        return ["## 2. Trajectory and early update", "Not enough data.", ""], a
    # blend weight chosen on the other quarters (within protocol and k)
    a["blend"] = np.nan
    for (prot, k), s in a.groupby(["protocol", "k"]):
        lp, ld, ly = np.log(s["prior"]), np.log(s["data"].clip(lower=1e-3)), s["y"].to_numpy()
        for q in s["quarter"].unique():
            other = (s["quarter"] != q).to_numpy()
            if other.sum() < 10:
                w = 0.5
            else:
                mae = [np.mean(np.abs(np.exp(w * lp[other] + (1 - w) * ld[other]) - ly[other])) for w in WEIGHTS]
                w = WEIGHTS[int(np.argmin(mae))]
            inside = ~other
            a.loc[s.index[inside], "blend"] = np.exp(w * lp[inside] + (1 - w) * ld[inside])
            a.loc[s.index[inside], "prior_weight"] = w
    a["blend_prosp"] = np.exp(FIXED_WEIGHT * np.log(a["prior"]) +
                              (1 - FIXED_WEIGHT) * np.log(a["data_prosp"].clip(lower=1e-3)))
    a.to_csv(config.RESULTS / "early_update.csv", index=False)

    r = ["## 2. Trajectory and early update",
         "MAE of Y1 after observing k periods of 30 days. `prior` = M3 before opening; `naive` = observed mean "
         "of r without correcting the ramp; `data` = observed mean corrected with the shape g(m) of the other "
         "quarters; `blend` = combination of prior and data with the weight chosen on the other quarters. "
         "Columns `_prosp` use the prospective shape (development factors available at the start of the "
         f"opening quarter) and a fixed weight of {FIXED_WEIGHT}.", ""]
    met = []
    for (prot, group, k), s in list(a.groupby(["protocol", "grupo", "k"])) + \
            [((p, "B1+B2", k), s) for (p, k), s in a.groupby(["protocol", "k"])]:
        row = {"protocol": prot, "grupo": group, "k": k, "n": len(s), "prior_weight": s["prior_weight"].mean()}
        for c in ("prior", "naive", "data", "blend", "data_prosp", "blend_prosp"):
            row[c] = np.mean(np.abs(s[c] - s["y"]))
        for c in ("blend", "blend_prosp"):
            row[f"p_{c}_vs_prior"] = stats.wilcoxon(np.abs(s[c] - s["y"]), np.abs(s["prior"] - s["y"])).pvalue \
                if len(s) >= 6 else np.nan
        met.append(row)
    met = pd.DataFrame(met)
    for prot in PRIOR_PROTOCOLS:
        s = met[met["protocol"] == prot]
        if len(s):
            r += [f"### Prior: M3 with {prot} labels",
                  table(s.drop(columns="protocol").set_index(["grupo", "k"]), floatfmt=".3f"), ""]
    c = pd.DataFrame(trajectory)
    if len(c):
        c["err_shape"] = (c["with_shape"] - c["obs"]).abs()
        c["err_shape_prosp"] = (c["with_shape_prosp"] - c["obs"]).abs()
        c["err_flat"] = (c["flat"] - c["obs"]).abs()
        t = c.groupby(["protocol", "grupo"])[["err_shape", "err_shape_prosp", "err_flat"]].mean()
        r += ["### Error of the trajectory over periods 1 to 6",
              "MAE of r by station and period: Y1_hat x g(m) against a flat trajectory at Y1_hat.", "",
              table(t.rename(columns={"err_shape": "shape, other quarters", "err_shape_prosp": "prospective shape",
                                      "err_flat": "flat"}), floatfmt=".3f"), ""]
    return r, met


# ---------------- 3. stock-outs ----------------

def stock_outs(per, priors, curves) -> list[str]:
    import statsmodels.formula.api as smf
    r = ["## 3. Sensitivity to stock-outs (empty or full station)"]
    if not {"frac_vacia", "frac_llena"} <= set(per.columns):
        return r + ["The period panel has no GBFS columns (frac_vacia, frac_llena).", ""]
    d = per[per["valido"] & per["frac_vacia"].notna() & (per["r_salidas"] > 0)].copy()
    if d["estacion"].nunique() < 20:
        return r + ["Not enough GBFS data.", ""]
    d["log_r"] = np.log(d["r_salidas"])
    d["month"] = pd.to_datetime(d["inicio"]).dt.to_period("M").astype(str)
    rows = []
    for g in ("A", "B1", "B2"):
        s = d[d["grupo"] == g]
        if s["estacion"].nunique() < 10:
            continue
        m = smf.ols("log_r ~ frac_vacia + frac_llena + C(estacion) + C(month)", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["estacion"].astype("category").cat.codes})
        rows.append({"grupo": g, "stations": str(s["estacion"].nunique()), "periods": str(len(s)),
                     "effect of +10 pp empty (%)": 100 * (np.exp(0.1 * m.params["frac_vacia"]) - 1),
                     "p empty": m.pvalues["frac_vacia"],
                     "effect of +10 pp full (%)": 100 * (np.exp(0.1 * m.params["frac_llena"]) - 1),
                     "p full": m.pvalues["frac_llena"]})
    r += ["Within each station (station and month fixed effects), change in r when the station spends 10 more "
          "percentage points of time empty or full. Periods with GBFS data only (from 2024-08).", "",
          table(pd.DataFrame(rows).set_index("grupo"), floatfmt=".3g"), ""]
    m3 = priors[(priors["model"] == "M3_lgbm") & priors["protocol"].isin(models.PROTOCOLS)].copy()
    m3["estacion"] = m3["estacion"].astype(str)
    m3 = m3.merge(curves[["estacion", "mean_empty_share"]], on="estacion").dropna(subset=["mean_empty_share"])
    for prot, s in m3.groupby("protocol"):
        if len(s) >= 20:
            rho = stats.spearmanr(np.log(s["y"] / s["pred"]), s["mean_empty_share"])
            r.append(f"- {prot} labels: Spearman correlation between the log error of M3 (observed/predicted) and "
                     f"the mean share of time empty = {rho.statistic:.3f} (p = {rho.pvalue:.3g}, n = {len(s)}). "
                     "A negative value means that M3 overestimates where the station is empty more often.")
    return r + [""]


def main():
    config.ensure_dirs()
    path = config.RESULTS / "model_comparison_predictions.csv"
    if not path.exists():
        raise SystemExit("Missing model_comparison_predictions.csv: run 02_model_comparison.py first")
    priors = pd.read_csv(path, dtype={"estacion": str})
    path_ramp = config.RESULTS / "ramp_predictions.csv"
    if path_ramp.exists():
        pr = pd.read_csv(path_ramp, dtype={"estacion": str})
        priors = pd.concat([priors, pr[pr["protocol"] == "ramp"]], ignore_index=True)
    per, per_exp, curves = load()
    r = ["# Maturation shape and early update", ""]
    r += common_shape(per_exp)
    text, _ = early_update(per, per_exp, priors)
    r += text
    r += stock_outs(per, priors, curves)
    models.write_report("early_update.md", r)


if __name__ == "__main__":
    main()
