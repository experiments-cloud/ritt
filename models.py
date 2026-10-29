"""Data loading, models, rolling-origin validation and metrics shared by the experiments.

Target: Y1, the mean relative demand index r over periods 7 to 12 after opening.

Rolling origin by opening quarter: for each quarter, Y1 is predicted for the expansion stations
(B1, B2) that open in it, training only with stations that opened before. Training sets:
  mature  only stations whose Y1 window ended before the quarter (what a planner actually has);
  oracle  every station opened before the quarter, with its final Y1 even when it is observed
          later (upper reference, unavailable in practice).
03_ramp_aware_labels.py adds a third set, ramp (mature stations plus provisional labels).

Models:
  M0_k3, M0_k5  mean Y1 of the 3 or 5 nearest training stations
  M0_rule       planning rule: log-linear regression on distance to Metro, Metrobus and cycleway
                and cycleway length within 500 m
  M1_voronoi    Voronoi area interpolation (Liu & Oshan, 2022): the cell of the new station takes
                the Y1 of the cells it overlaps, weighted by area (cells clipped to 300 m); without
                overlap it falls back to M0_k3
  M2_ridge      log-linear ridge regression (penalty chosen over 1 km spatial blocks)
  M3_lgbm       LightGBM on log Y1, mean of 5 seeds, fixed hyperparameters

When the training set has a `weight` column, M0_k*, M0_rule and M3_lgbm use it as the weight of
each station (provisional labels).

Input columns keep the names of the processed data (estacion = station id, grupo = group,
apertura = opening date, x and y in metres).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import MultiPoint, Point
from shapely.ops import voronoi_diagram

import config

warnings.filterwarnings("ignore", category=UserWarning)
Y1_WINDOW_DAYS = 12 * config.PERIOD_DAYS      # Y1 is fully observed 360 days after opening
NOT_PREDICTORS = {"estacion", "grupo", "alcaldia", "apertura", "x", "y", "Y1", "weight", "k"}
LOG1P = ("ue_", "empleo_", "pob_", "densidad", "ocupada", "n_", "km_", "dist_")
SEEDS = (0, 1, 2, 3, 4)
BASELINES = ("M0_k3", "M0_k5", "M0_rule", "M1_voronoi")
RULE = ["dist_metro_m", "dist_metrobus_m", "dist_ciclovia_m", "km_ciclovia_500"]
PROTOCOLS = ("mature", "oracle")

# predictor blocks for the ablation
PLANNED_NETWORK = ["dist_estacion_plan_m", "n_est_plan_500", "n_est_plan_1000"]
OPEN_NETWORK = ["dist_estacion_abierta_m", "n_abiertas_1000", "r_vecinas_previa"]
NETWORK = PLANNED_NETWORK + OPEN_NETWORK + ["dist_poligono_original_m"]
CYCLEWAYS = ["km_ciclovia_500", "dist_ciclovia_m"]


# ---------------- data ----------------

def read_predictors() -> pd.DataFrame:
    v = pd.read_csv(config.PREDICTORS, dtype={"estacion": str}, parse_dates=["apertura"])
    v["quarter"] = v["apertura"].dt.to_period("Q")
    return v


def read_curves() -> pd.DataFrame:
    if not config.STATION_CURVES.exists():
        raise SystemExit(f"Missing {config.STATION_CURVES.name}: run 01_maturation.py first")
    return pd.read_csv(config.STATION_CURVES, dtype={"estacion": str})


def load(target: str = "Y1", curves: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predictors of each station with column Y1 taken from `target` (only Y1 > 0)."""
    if curves is None:
        curves = read_curves()
    y = curves[["estacion", target]].rename(columns={target: "Y1"})
    d = read_predictors().merge(y, on="estacion", how="left")
    return d[d["Y1"].notna() & (d["Y1"] > 0)].reset_index(drop=True)


def predictors(d: pd.DataFrame) -> list[str]:
    return [c for c in d.select_dtypes("number").columns if c not in NOT_PREDICTORS]


def variants(cols: list[str]) -> dict[str, list[str]]:
    """Predictor sets of the M3 ablation."""
    without = lambda drop: [c for c in cols if c not in set(drop)]  # noqa: E731
    return {
        "full": cols,
        "no_capacity": without(["capacidad"]),
        "no_planned_network": without(PLANNED_NETWORK),
        "no_cycleways": without(CYCLEWAYS),
        "no_neighbour_index": without(["r_vecinas_previa"]),
        "conservative": without(["capacidad", *PLANNED_NETWORK, *CYCLEWAYS]),
        "environment_only": without(NETWORK),
        "network_only": [c for c in cols if c in set(NETWORK)],
    }


def transform(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if c.startswith(LOG1P):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


def quarters(d: pd.DataFrame) -> list:
    return sorted(d.loc[d["grupo"].isin(["B1", "B2"]), "quarter"].unique())


def test_set(d: pd.DataFrame, q) -> pd.DataFrame:
    return d[(d["quarter"] == q) & d["grupo"].isin(["B1", "B2"])]


def training_set(d: pd.DataFrame, q, protocol: str, window_days: int = Y1_WINDOW_DAYS) -> pd.DataFrame:
    tr = d[d["apertura"] < q.start_time]
    if protocol == "mature":
        tr = tr[tr["apertura"] + pd.Timedelta(days=window_days) < q.start_time]
    return tr


def prediction_table(protocol, q, model, te, pred, fallback=False, **extra) -> pd.DataFrame:
    return pd.DataFrame({"protocol": protocol, "quarter": str(q), "model": model,
                         "estacion": te["estacion"].to_numpy(), "grupo": te["grupo"].to_numpy(),
                         "y": te["Y1"].to_numpy(), "pred": pred, "fallback": fallback, **extra})


# ---------------- models ----------------

def _weights(tr):
    return tr["weight"].to_numpy() if "weight" in tr else None


def m0_knn(tr, te, k):
    xy, y, w = tr[["x", "y"]].to_numpy(), tr["Y1"].to_numpy(), _weights(tr)
    out = []
    for x0, y0 in te[["x", "y"]].to_numpy():
        idx = np.argsort(np.hypot(xy[:, 0] - x0, xy[:, 1] - y0))[:k]
        out.append(y[idx].mean() if w is None else np.average(y[idx], weights=w[idx]))
    return np.array(out), np.zeros(len(te), bool)


def m0_rule(tr, te):
    """log Y1 ~ distance to Metro, Metrobus and cycleway (least squares, weighted when `weight` exists)."""
    cols = [c for c in RULE if c in tr and tr[c].notna().any()]
    if not cols:
        return np.full(len(te), tr["Y1"].median()), np.ones(len(te), bool)
    Xtr, Xte = transform(tr[cols]), transform(te[cols])
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    A = np.c_[np.ones(len(Xtr)), Xtr.to_numpy(float)]
    y, w = np.log(tr["Y1"].to_numpy()), _weights(tr)
    if w is None:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        smearing = np.mean(np.exp(y - A @ beta))
    else:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        smearing = np.average(np.exp(y - A @ beta), weights=w)
    return np.exp(np.c_[np.ones(len(Xte)), Xte.to_numpy(float)] @ beta) * smearing, np.zeros(len(te), bool)


def cells(points_xy, radius=300):
    """Voronoi cells clipped to a circle of `radius` m around each point."""
    pts = [Point(p) for p in points_xy]
    vor = voronoi_diagram(MultiPoint(pts), envelope=MultiPoint(pts).buffer(5000).envelope)
    out = []
    for p in pts:
        cell = next((c for c in vor.geoms if c.contains(p)), None)
        out.append(cell.intersection(p.buffer(radius)) if cell is not None else p.buffer(radius))
    return out


def m1_voronoi(tr, te):
    xy_tr = tr[["x", "y"]].to_numpy()
    old = cells(xy_tr)
    y_tr = tr["Y1"].to_numpy()
    pred, used_fallback = [], []
    fallback, _ = m0_knn(tr, te, 3)
    for i, (x, y) in enumerate(te[["x", "y"]].to_numpy()):
        near = np.where(np.hypot(xy_tr[:, 0] - x, xy_tr[:, 1] - y) <= 2000)[0]
        if len(near) == 0:
            pred.append(fallback[i]), used_fallback.append(True)
            continue
        new = cells(np.vstack([xy_tr[near], [x, y]]))[-1]
        areas = np.array([new.intersection(old[j]).area for j in near])
        if areas.sum() <= 0:
            pred.append(fallback[i]), used_fallback.append(True)
        else:
            pred.append((areas * y_tr[near]).sum() / areas.sum()), used_fallback.append(False)
    return np.array(pred), np.array(used_fallback)


def spatial_blocks(tr, size_m=1000):
    return (np.floor(tr["x"] / size_m).astype(int).astype(str) + "_" +
            np.floor(tr["y"] / size_m).astype(int).astype(str)).to_numpy()


def prepare_X(tr, te, cols):
    cols = [c for c in cols if tr[c].notna().any()]          # no data at all in training: dropped
    Xtr, Xte = transform(tr[cols]), transform(te[cols])
    for c in cols:                       # missing values: training median plus an indicator
        if Xtr[c].isna().any() or Xte[c].isna().any():
            med = Xtr[c].median()
            Xtr[f"{c}_missing"], Xte[f"{c}_missing"] = Xtr[c].isna().astype(float), Xte[c].isna().astype(float)
            Xtr[c], Xte[c] = Xtr[c].fillna(med), Xte[c].fillna(med)
    constant = [c for c in Xtr.columns if Xtr[c].std() == 0]
    return Xtr.drop(columns=constant), Xte.drop(columns=constant)


def m2_ridge(tr, te, cols):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = prepare_X(tr, te, cols)
    y = np.log(tr["Y1"].to_numpy())
    g = spatial_blocks(tr)
    cv = list(GroupKFold(n_splits=min(5, len(set(g)))).split(Xtr, y, g))
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20), cv=cv))
    model.fit(Xtr, y)
    smearing = np.mean(np.exp(y - model.predict(Xtr)))          # Duan's smearing
    return np.exp(model.predict(Xte)) * smearing, np.zeros(len(te), bool)


def m3_lgbm(tr, te, cols, importance: list | None = None, label=""):
    import lightgbm as lgb
    Xtr, Xte = transform(tr[cols]), transform(te[cols])
    y, w = np.log(tr["Y1"].to_numpy()), _weights(tr)
    preds, preds_tr, contrib = [], [], []
    for s in SEEDS:
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=s, verbose=-1)
        m.fit(Xtr, y, sample_weight=w)
        preds.append(m.predict(Xte))
        preds_tr.append(m.predict(Xtr))
        if importance is not None:
            contrib.append(m.predict(Xte, pred_contrib=True)[:, :-1])
    if importance is not None and len(te):
        shap = np.mean(contrib, axis=0)
        for group in ("B1", "B2"):
            mask = (te["grupo"] == group).to_numpy()
            if mask.any():
                for c, val in zip(cols, np.abs(shap[mask]).mean(axis=0)):
                    importance.append({"quarter": label, "grupo": group, "predictor": c, "mean_abs_shap": val})
    resid = np.exp(y - np.mean(preds_tr, axis=0))
    smearing = np.mean(resid) if w is None else np.average(resid, weights=w)      # Duan's smearing
    return np.exp(np.mean(preds, axis=0)) * smearing, np.zeros(len(te), bool)


# ---------------- rolling-origin validation ----------------

def rolling_origin(d, protocol, min_train, min_test, with_importance=False):
    """Predictions of the six models in every quarter that meets the minimum sizes."""
    cols = predictors(d)
    rows, importance, log = [], [], []
    for q in quarters(d):
        te, tr = test_set(d, q), training_set(d, q, protocol)
        entry = {"protocol": protocol, "quarter": str(q), "n_test": len(te), "n_train": len(tr),
                 "n_train_expansion": int(tr["grupo"].isin(["B1", "B2"]).sum())}
        if len(te) < min_test or len(tr) < min_train:
            log.append({**entry, "status": "skipped"})
            continue
        log.append({**entry, "status": "evaluated"})
        imp = importance if with_importance else None
        fits = {
            "M0_k3": lambda: m0_knn(tr, te, 3), "M0_k5": lambda: m0_knn(tr, te, 5),
            "M0_rule": lambda: m0_rule(tr, te),
            "M1_voronoi": lambda: m1_voronoi(tr, te), "M2_ridge": lambda: m2_ridge(tr, te, cols),
            "M3_lgbm": lambda: m3_lgbm(tr, te, cols, imp, str(q)),
        }
        for name, f in fits.items():
            pred, fallback = f()
            rows.append(prediction_table(protocol, q, name, te, pred, fallback))
    pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pred, pd.DataFrame(importance), pd.DataFrame(log)


# ---------------- metrics ----------------

def top20(y, p):
    k = max(1, int(round(0.2 * len(y))))
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k


def capture20(y, p):
    """Demand captured by the predicted top 20 %: share of the total and of the maximum attainable."""
    y = np.asarray(y, float)
    k = max(1, int(round(0.2 * len(y))))
    captured = y[np.argsort(-np.asarray(p))[:k]].sum()
    return 100 * captured / y.sum(), 100 * captured / np.sort(y)[::-1][:k].sum()


def holm(pvals: dict) -> dict:
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (m, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[m] = running
    return adjusted


def metrics(pred: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """Metrics by protocol and group; bootstrap 95 % CI of the MAE and Wilcoxon tests with Holm's
    correction against the best baseline (lowest MAE among the BASELINES present)."""
    rows = []
    for (prot, group), g in list(pred.groupby(["protocol", "grupo"])) + \
            [((p, "B1+B2"), gg) for p, gg in pred.groupby("protocol")]:
        y = g.pivot_table(index="estacion", columns="model", values="y", aggfunc="first").iloc[:, 0]
        piv = g.pivot_table(index="estacion", columns="model", values="pred", aggfunc="first")
        err = piv.sub(y, axis=0).abs()
        base = [b for b in BASELINES if b in err]
        best_base = err[base].mean().idxmin()
        n = len(y)
        boot = rng.integers(0, n, size=(n_boot, n))
        pvals, block = {}, []
        for m in piv.columns:
            row = {"protocol": prot, "grupo": group, "model": m, "n": n,
                   "MAE": err[m].mean(), "RMSE": np.sqrt((err[m] ** 2).mean()),
                   "spearman": stats.spearmanr(y, piv[m]).statistic,
                   "top20": top20(y.to_numpy(), piv[m].to_numpy())}
            row["capture20"], row["capture20_of_max"] = capture20(y, piv[m])
            row["bias_pct"] = 100 * (piv[m].mean() - y.mean()) / y.mean()
            mae_b = err[m].to_numpy()[boot].mean(axis=1)
            row["MAE_ci95"] = f"[{np.quantile(mae_b, 0.025):.3f}, {np.quantile(mae_b, 0.975):.3f}]"
            row["pct_fallback"] = 100 * g.loc[g["model"] == m, "fallback"].mean()
            if m != best_base and n >= 6:
                pvals[m] = stats.wilcoxon(err[m], err[best_base]).pvalue
            row["best_baseline"] = best_base
            block.append(row)
        adjusted = holm(pvals)
        for r in block:
            r["p_holm_vs_best_baseline"] = adjusted.get(r["model"], np.nan)
        rows += block
    return pd.DataFrame(rows)


def write_report(name: str, lines: list[str]):
    path = config.REPORTS / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done: {path}")
