"""Labels available at a decision date t.

At t (first day of each opening quarter) the planner sees three kinds of stations:
  mature     their target window (periods a..b, by default 7-12) ended before t;
  maturing   opened before t, with only their first k periods observed;
  candidates the stations that open during the quarter and are to be predicted.

Maturing stations receive a provisional label from chain-ladder development factors:
  C_k = sum of r over periods 1..k (valid and consecutive periods since opening, ended before t).
  Two development triangles: renovated stations (A), whose ramp reflects the relaunch of the
  system, and expansion stations (B1, B2). For each class,
  F_k = sum C_{k+1} / sum C_k over its own pairs observed before t; with fewer than 20 pairs the
  factor pooled over both classes is used; if that is also unavailable, the next period repeats
  the level of the previous one (no further growth).
  Ĉ_j = C_j for j <= k, otherwise Ĉ_{j-1} * F_{j-1}.
  provisional label = (Ĉ_b - Ĉ_{a-1}) / (b - a + 1)      weight = C_k / Ĉ_b (between 0 and 1).
With a = 7 and b = 12: provisional Y1 = (Ĉ_12 - Ĉ_6) / 6.

The same factors imply a maturation shape g(m) that uses only information available before t
(function shape), used by 04_early_update.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config
import models

MIN_PAIRS = 20
PERIOD_DAYS = config.PERIOD_DAYS
MODELS = ("M0_k5", "M0_rule", "M3_lgbm")


def cumulative(per: pd.DataFrame, opening: pd.Series, t: pd.Timestamp, kmax: int) -> dict[str, np.ndarray]:
    """C_1..C_k of each station from its valid, consecutive periods ended before t."""
    out = {}
    for st, g in per.groupby("estacion"):
        op = opening.get(st)
        if op is None or pd.isna(op) or op >= t:
            continue
        g = g.set_index("periodo")
        r = []
        for m in range(1, kmax + 1):
            if op + pd.Timedelta(days=PERIOD_DAYS * m) > t or m not in g.index or not g.at[m, "valido"]:
                break
            r.append(g.at[m, "r_salidas"])
        if r:
            out[st] = np.cumsum(r)
    return out


def factors(C: dict[str, np.ndarray], cls: dict[str, str], kmax: int) -> dict[str, np.ndarray]:
    """Development factors by class ('A', 'B'); NaN = not estimable (fewer than MIN_PAIRS pairs)."""
    def f(ids):
        F = np.full(kmax, np.nan)          # F[k] takes C_k to C_{k+1}
        for k in range(1, kmax):
            pairs = [(C[e][k - 1], C[e][k]) for e in ids if len(C[e]) > k]
            if len(pairs) >= MIN_PAIRS:
                a, b = np.array(pairs).T
                F[k] = b.sum() / a.sum()
        return F
    pooled = f(list(C))
    out = {}
    for c in ("A", "B"):
        own = f([e for e in C if cls.get(e) == c])
        out[c] = np.where(np.isnan(own), pooled, own)
    return out


def project(c: np.ndarray, F: np.ndarray, b: int) -> np.ndarray:
    """Ĉ_1..Ĉ_b: the observed cumulative values c followed by their projection with factors F."""
    k = len(c)
    Ch = np.empty(b)
    Ch[:k] = c
    for j in range(k, b):
        if np.isnan(F[j]):                  # no estimable factor: the last level is repeated
            Ch[j] = Ch[j - 1] + (Ch[j - 1] - (Ch[j - 2] if j >= 2 else 0.0))
        else:
            Ch[j] = Ch[j - 1] * F[j]
    return Ch


def provisional(c: np.ndarray, F: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """Provisional label for window a..b and its weight, from the observed cumulative values c."""
    Ch = project(c, F, b)
    before = Ch[a - 2] if a >= 2 else 0.0
    return (Ch[b - 1] - before) / (b - a + 1), float(np.clip(c[-1] / Ch[b - 1], 0, 1))


def shape(F: np.ndarray, a: int = 7, b: int = 12) -> pd.Series:
    """Maturation shape implied by the factors: expected increment of C in each period m divided by
    the expected mean level over window a..b (g(m) = 1 is the level of Y1)."""
    Ch = project(np.array([1.0]), F, b)
    inc = np.diff(np.r_[0.0, Ch])
    level = (Ch[b - 1] - (Ch[a - 2] if a >= 2 else 0.0)) / (b - a + 1)
    return pd.Series(inc / level, index=range(1, b + 1))


def labels(d, per, t, y_final, a=7, b=12):
    """Labels available at t: mature (weight 1) and provisional (weight = observed development)."""
    earlier = d[d["apertura"] < t]
    mature = earlier[earlier["apertura"] + pd.Timedelta(days=b * PERIOD_DAYS) < t].copy()
    mature = mature[mature["estacion"].map(y_final).notna()]
    mature["Y1"], mature["weight"], mature["k"] = mature["estacion"].map(y_final), 1.0, b
    C = cumulative(per, d.set_index("estacion")["apertura"], t, b)
    cls = {e: ("A" if g == "A" else "B") for e, g in zip(d["estacion"], d["grupo"])}
    F = factors(C, cls, b)
    rows = []
    for st in earlier.loc[~earlier["estacion"].isin(mature["estacion"]), "estacion"]:
        if st in C:
            y, w = provisional(C[st], F[cls[st]], a, b)
            if y > 0:
                rows.append({"estacion": st, "Y1": y, "weight": w, "k": len(C[st])})
    prov = earlier.merge(pd.DataFrame(rows, columns=["estacion", "Y1", "weight", "k"]), on="estacion") \
        if rows else earlier.iloc[:0].assign(Y1=[], weight=[], k=[])
    return mature, prov, F


def read_inputs():
    d = models.read_predictors()
    curves = models.read_curves()
    per = pd.read_parquet(config.PERIOD_PANEL)
    per["estacion"] = per["estacion"].astype(str)
    return d, curves, per


def fit(tr, te, cols):
    """Predictions of the three learners of the factorial comparison."""
    return {"M0_k5": models.m0_knn(tr, te, 5)[0], "M0_rule": models.m0_rule(tr, te)[0],
            "M3_lgbm": models.m3_lgbm(tr, te, cols)[0]}


def test_set(d, q, y_final):
    """Expansion stations that open in q, with their final label."""
    te = d[(d["quarter"] == q) & d["grupo"].isin(["B1", "B2"])].copy()
    te["Y1"] = te["estacion"].map(y_final)
    return te[te["Y1"].notna()]


def compare(pred, n_boot, rng):
    """Ramp-aware vs mature-only labels: same learner and same stations (quarters evaluated in both)."""
    rows = []
    for model in MODELS:
        a = pred[(pred["model"] == model) & (pred["protocol"] == "mature")].set_index("estacion")
        b = pred[(pred["model"] == model) & (pred["protocol"] == "ramp")].set_index("estacion")
        common = a.index.intersection(b.index)
        for group in ("B1", "B2", "B1+B2"):
            idx = common if group == "B1+B2" else common[a.loc[common, "grupo"].eq(group).to_numpy()]
            if len(idx) < 6:
                continue
            ea, eb = (a.loc[idx, "pred"] - a.loc[idx, "y"]).abs(), (b.loc[idx, "pred"] - b.loc[idx, "y"]).abs()
            delta = (eb - ea).to_numpy()
            boot = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
            rows.append({"model": model, "grupo": group, "n": len(idx), "MAE_mature": ea.mean(),
                         "MAE_ramp": eb.mean(), "dMAE": delta.mean(),
                         "dMAE_ci95": f"[{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]",
                         "p_wilcoxon": stats.wilcoxon(eb, ea).pvalue,
                         "bias_mature_pct": 100 * (a.loc[idx, "pred"].mean() / a.loc[idx, "y"].mean() - 1),
                         "bias_ramp_pct": 100 * (b.loc[idx, "pred"].mean() / b.loc[idx, "y"].mean() - 1)})
    return pd.DataFrame(rows)
