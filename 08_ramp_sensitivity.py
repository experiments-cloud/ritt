"""Step 8. Sensitivity analyses with ramp-aware labels.

Steps 5 (ablation, target 13-18) and 7 (censoring) evaluate the models with mature and oracle
labels. Here they are repeated with the ramp-aware labels of step 3 (ramp.py):

  A. M3 ablation: the variants of step 5, trained on mature plus provisional labels.
  B. Target over periods 13-18: provisional labels with the same rule over window 13-18
     (a station is mature when its 18 periods ended before the decision date).
  C. Censoring: index r and Y1 recomputed with the two bounds of step 7 (kappa_censoring.parquet,
     censoring.py); provisional labels are computed from the corrected index and M0_k5, M0_rule and
     M3 are evaluated with mature-only and ramp-aware labels.

Requires steps 1, 3 and 7.

Output: data/results/sensitivity_*.csv and data/reports/ramp_sensitivity.md

    python 08_ramp_sensitivity.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import censoring
import config
import models
import ramp

TABLE_COLS = ["model", "n", "MAE", "MAE_ci95", "spearman", "capture20_of_max", "bias_pct", "p_holm_vs_best_baseline"]


def with_ramp(mature, prov):
    return pd.concat([mature, prov], ignore_index=True)


# ---------------- A. ablation ----------------

def ablation(d, per, y_final, cols, min_train, min_test):
    sets = models.variants(cols)
    out = []
    for q in models.quarters(d):
        te = ramp.test_set(d, q, y_final)
        mature, prov, _ = ramp.labels(d, per, q.start_time, y_final)
        tr = with_ramp(mature, prov)
        if len(tr) < min_train or len(te) < min_test:
            continue
        for name, c in sets.items():
            if c:
                out.append(models.prediction_table("ramp", q, name, te, models.m3_lgbm(tr, te, c)[0]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def deltas(pred, reference, n_boot, rng):
    """MAE of each variant minus MAE of `reference`, paired by station, with a bootstrap 95 % CI."""
    rows = []
    for group in ("B1", "B2", "B1+B2"):
        g = pred if group == "B1+B2" else pred[pred["grupo"] == group]
        piv = g.pivot_table(index="estacion", columns="model", values="pred", aggfunc="first")
        y = g.pivot_table(index="estacion", columns="model", values="y", aggfunc="first").iloc[:, 0]
        if reference not in piv or len(y) < 6:
            continue
        err = piv.sub(y, axis=0).abs()
        boot = rng.integers(0, len(y), size=(n_boot, len(y)))
        for m in piv.columns:
            dlt = (err[m] - err[reference]).to_numpy()
            b = dlt[boot].mean(axis=1)
            rows.append({"grupo": group, "variant": m, "n": len(y), "MAE": err[m].mean(), "dMAE": dlt.mean(),
                         "dMAE_lo": np.quantile(b, .025), "dMAE_hi": np.quantile(b, .975),
                         "spearman": stats.spearmanr(y, piv[m]).statistic,
                         "bias_pct": 100 * (piv[m].mean() / y.mean() - 1)})
    return pd.DataFrame(rows)


# ---------------- B. target 13-18 ----------------

def target_13_18(d, per, y_final, cols, min_train, min_test):
    conservative = models.variants(cols)["conservative"]
    out = []
    for q in models.quarters(d):
        te = ramp.test_set(d, q, y_final)
        mature, prov, _ = ramp.labels(d, per, q.start_time, y_final, 13, 18)
        for strategy, tr in (("mature", mature), ("ramp", with_ramp(mature, prov))):
            if len(tr) < min_train or len(te) < min_test:
                continue
            for name, p in ramp.fit(tr, te, cols).items():
                out.append(models.prediction_table(strategy, q, name, te, p))
            out.append(models.prediction_table(strategy, q, "M3_conservative", te,
                                               models.m3_lgbm(tr, te, conservative)[0]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------- C. censoring ----------------

def censoring_ramp(d, p, kap_all, cols, min_train, min_test):
    out = []
    for bound in ("lower", "upper"):
        per_c = censoring.corrected_periods(p, kap_all[kap_all["bound"] == bound])
        y_c = censoring.y1(per_c).set_index("estacion")["Y1"].where(lambda s: s > 0).dropna().to_dict()
        for q in models.quarters(d):
            te = ramp.test_set(d, q, y_c)
            mature, prov, _ = ramp.labels(d, per_c, q.start_time, y_c)
            for strategy, tr in (("mature", mature), ("ramp", with_ramp(mature, prov))):
                if len(tr) < min_train or len(te) < min_test:
                    continue
                for name, pr in ramp.fit(tr, te, cols).items():
                    out.append(models.prediction_table(strategy, q, name, te, pr, bound=bound))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------- report ----------------

def model_table(pred, n_boot, rng):
    met = models.metrics(pred, n_boot, rng)
    met["relative_MAE"] = met["MAE"] / met.apply(
        lambda f: pred.loc[(pred["protocol"] == f["protocol"]) &
                           ((pred["grupo"] == f["grupo"]) | (f["grupo"] == "B1+B2")), "y"].median(), axis=1)
    return met


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--only", choices=["A", "B", "C"], help="run a single part")
    args = ap.parse_args()
    config.ensure_dirs()
    rng = np.random.default_rng(2026)

    d, curves, per = ramp.read_inputs()
    y1 = curves.set_index("estacion")["Y1"].where(lambda s: s > 0).dropna().to_dict()
    y1318 = curves.set_index("estacion")["Y1_13_18"].where(lambda s: s > 0).dropna().to_dict()
    cols = models.predictors(d)
    mn = dict(min_train=args.min_train, min_test=args.min_test)

    r = ["# Sensitivity with ramp-aware labels", "",
         f"Same chain-ladder rules as step 3 (A and expansion triangles, at least {ramp.MIN_PAIRS} pairs, flat "
         "level when no factor is available).", ""]

    if args.only in (None, "A"):
        print("A. Ablation with ramp-aware labels...", flush=True)
        pa = ablation(d, per, y1, cols, **mn)
        pa.to_csv(config.RESULTS / "sensitivity_ablation_predictions.csv", index=False)
        ta = deltas(pa, "full", args.bootstrap, rng)
        ta.to_csv(config.RESULTS / "sensitivity_ablation.csv", index=False)
        ta["variant"] = pd.Categorical(ta["variant"], list(models.variants(cols)), ordered=True)
        r += ["## A. M3 ablation with ramp-aware labels (all evaluable quarters)",
              "dMAE = MAE of the variant minus MAE of the full model; paired-bootstrap 95 % CI.", ""]
        for g in ("B1", "B2", "B1+B2"):
            s = ta[ta["grupo"] == g].sort_values("variant")
            if len(s):
                r += [f"### {g} (n = {s['n'].iloc[0]})", s.drop(columns=["grupo", "n"]).to_markdown(index=False, floatfmt=".3f"), ""]
        # conservative model against the baselines of step 3
        p3 = config.RESULTS / "ramp_predictions.csv"
        if p3.exists():
            b = pd.read_csv(p3, dtype={"estacion": str})
            b = b[(b["protocol"] == "ramp") & b["model"].isin(["M0_k5", "M0_rule"])]
            c = pa[pa["model"] == "conservative"].assign(model="M3_conservative")
            common = set(c["estacion"]) & set(b["estacion"])
            if common:
                mix = pd.concat([b[b["estacion"].isin(common)], c[c["estacion"].isin(common)]], ignore_index=True)
                mix["protocol"] = "ramp"
                mt = model_table(mix, args.bootstrap, rng)
                r += ["### Conservative M3 against the baselines (ramp-aware labels)",
                      mt[mt["grupo"] == "B1+B2"][TABLE_COLS].to_markdown(index=False, floatfmt=".3f"), ""]

    if args.only in (None, "B"):
        print("B. Target over periods 13-18...", flush=True)
        pb = target_13_18(d, per, y1318, cols, **mn)
        pb.to_csv(config.RESULTS / "sensitivity_y13_18_predictions.csv", index=False)
        r += ["## B. Target over periods 13-18", ""]
        if pb.empty:
            r += ["No evaluable quarter.", ""]
        else:
            r += [f"Quarters evaluated: {pb.groupby('protocol')['quarter'].nunique().to_dict()}.", ""]
            for strategy in ("mature", "ramp"):
                s = pb[pb["protocol"] == strategy]
                if s["estacion"].nunique() >= 6:
                    mt = model_table(s, args.bootstrap, rng)
                    r += [f"### {strategy} labels", mt[mt["grupo"] == "B1+B2"][TABLE_COLS].to_markdown(index=False, floatfmt=".3f"), ""]
            cmp = ramp.compare(pb[pb["model"].isin(ramp.MODELS)], args.bootstrap, rng)
            if len(cmp):
                r += ["### Ramp-aware vs mature-only labels, same stations", cmp.to_markdown(index=False, floatfmt=".3f"), ""]

    if args.only in (None, "C"):
        kp = config.RESULTS / "kappa_censoring.parquet"
        r += ["## C. Censoring with ramp-aware labels", ""]
        if not kp.exists():
            r += ["Missing kappa_censoring.parquet: run 07_censoring.py first.", ""]
        else:
            print("C. Censoring with ramp-aware labels...", flush=True)
            p = pd.read_parquet(config.DAILY_PANEL)
            p["estacion"] = p["estacion"].astype(str)
            p["fecha"] = pd.to_datetime(p["fecha"])
            pc = censoring_ramp(d, p, pd.read_parquet(kp).astype({"estacion": str}), cols, **mn)
            pc.to_csv(config.RESULTS / "sensitivity_censoring_predictions.csv", index=False)
            r += ["relative_MAE = MAE / median corrected Y1 of the stations evaluated.", ""]
            cols_c = TABLE_COLS[:3] + ["relative_MAE"] + TABLE_COLS[3:]
            for bound in ("lower", "upper"):
                s = pc[pc["bound"] == bound].drop(columns="bound")
                for strategy in ("mature", "ramp"):
                    t = s[s["protocol"] == strategy]
                    if t["estacion"].nunique() >= 6:
                        mt = model_table(t, args.bootstrap, rng)
                        r += [f"### {bound} bound, {strategy} labels, B1+B2",
                              mt[mt["grupo"] == "B1+B2"][cols_c].to_markdown(index=False, floatfmt=".3f"), ""]
                cmp = ramp.compare(s, args.bootstrap, rng)
                if len(cmp):
                    r += [f"### {bound} bound, ramp-aware vs mature-only labels, same stations",
                          cmp.to_markdown(index=False, floatfmt=".3f"), ""]

    models.write_report("ramp_sensitivity.md", r)


if __name__ == "__main__":
    main()
