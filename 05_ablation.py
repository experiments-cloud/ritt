"""Step 5. Predictor ablation and alternative target (model M3).

Repeats the rolling origin of step 2 (same quarters, mature and oracle labels) with LightGBM after
removing blocks of predictors (models.variants):

  - predictors whose value at opening is uncertain: dock capacity (current GBFS feed), planned
    network (final layout) and cycleways (no dates); the "conservative" variant removes all three;
  - whole families: ECOBICI network only or urban environment only;
  - alternative target Y1_13_18 (periods 13 to 18).

Output: data/results/ablation_predictions.csv, ablation_metrics.csv and data/reports/ablation.md

    python 05_ablation.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config
import models
from models import variants

WINDOWS = {"Y1": 12 * config.PERIOD_DAYS, "Y1_13_18": 18 * config.PERIOD_DAYS}


def run(d, target, protocol, sets, min_train, min_test):
    rows = []
    for q in models.quarters(d):
        te, tr = models.test_set(d, q), models.training_set(d, q, protocol, WINDOWS[target])
        if len(te) < min_test or len(tr) < min_train:
            continue
        for name, cols in sets.items():
            if not cols:
                continue
            pred, _ = models.m3_lgbm(tr, te, cols)
            rows.append(pd.DataFrame({"target": target, "protocol": protocol, "quarter": str(q),
                                      "variant": name, "estacion": te["estacion"].to_numpy(),
                                      "grupo": te["grupo"].to_numpy(), "y": te["Y1"].to_numpy(), "pred": pred}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metrics(pred: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    keys = ["target", "protocol"]
    blocks = [((*k, g), s) for k, gg in pred.groupby(keys) for g, s in gg.groupby("grupo")]
    blocks += [((*k, "B1+B2"), gg) for k, gg in pred.groupby(keys)]
    for (target, prot, group), g in blocks:
        y = g.pivot_table(index="estacion", columns="variant", values="y", aggfunc="first").iloc[:, 0]
        p = g.pivot_table(index="estacion", columns="variant", values="pred", aggfunc="first")
        err = p.sub(y, axis=0).abs()
        n = len(y)
        boot = rng.integers(0, n, size=(n_boot, n))
        for v in p.columns:
            delta = (err[v] - err["full"]).to_numpy()
            db = delta[boot].mean(axis=1)
            rows.append({"target": target, "protocol": prot, "grupo": group, "variant": v, "n": n,
                         "MAE": err[v].mean(), "spearman": stats.spearmanr(y, p[v]).statistic,
                         "top20": models.top20(y.to_numpy(), p[v].to_numpy()),
                         "dMAE": delta.mean(), "dMAE_lo": np.quantile(db, 0.025), "dMAE_hi": np.quantile(db, 0.975)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    config.ensure_dirs()

    preds = []
    for target in WINDOWS:
        d = models.load(target)
        sets = variants(models.predictors(d))
        for prot in models.PROTOCOLS:
            p = run(d, target, prot, sets, args.min_train, args.min_test)
            print(f"  {target}, {prot} labels: {p['estacion'].nunique() if len(p) else 0} stations evaluated")
            preds.append(p)
    pred = pd.concat(preds, ignore_index=True)
    if pred.empty:
        raise SystemExit("No quarter met the minimum sizes.")
    pred.to_csv(config.RESULTS / "ablation_predictions.csv", index=False)
    met = metrics(pred, args.bootstrap, np.random.default_rng(2026))
    met.to_csv(config.RESULTS / "ablation_metrics.csv", index=False)

    order = list(variants([]))
    r = ["# Predictor ablation and alternative target (M3)", "",
         "dMAE = MAE of the variant minus MAE of the full model, with a paired-bootstrap 95 % CI. "
         "dMAE > 0: removing the block worsens the prediction.", "",
         "- `conservative`: no dock capacity, no planned network and no cycleways (only information "
         "known with certainty when the station opens).",
         "- `environment_only`: no ECOBICI network predictor. `network_only`: ECOBICI network only.", ""]
    for target in WINDOWS:
        for prot in models.PROTOCOLS:
            for g in ("B1", "B2", "B1+B2"):
                s = met[(met["target"] == target) & (met["protocol"] == prot) & (met["grupo"] == g)].copy()
                if s.empty:
                    continue
                s["dMAE_ci95"] = s.apply(lambda f: f"[{f['dMAE_lo']:+.3f}, {f['dMAE_hi']:+.3f}]", axis=1)
                s = s.set_index("variant").reindex([v for v in order if v in set(s["variant"])])
                r += [f"## {target}, {prot} labels, {g} (n = {int(s['n'].iloc[0])})",
                      s[["MAE", "dMAE", "dMAE_ci95", "spearman", "top20"]].to_markdown(floatfmt=".3f"), ""]
    models.write_report("ablation.md", r)


if __name__ == "__main__":
    main()
