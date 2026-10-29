"""Step 2. Model comparison for the mature demand (Y1) of new stations.

Rolling origin by opening quarter with the `mature` and `oracle` training sets and the models
M0_k3, M0_k5, M0_rule, M1_voronoi, M2_ridge and M3_lgbm (defined in models.py).

Metrics by group (B1 = interpolation, B2 = extrapolation): MAE, RMSE, Spearman, overlap of the top
20 %, demand captured by the predicted top 20 % (share of the total and of the maximum attainable)
and level bias; bootstrap 95 % CI and Wilcoxon tests against the best baseline with Holm's
correction. Predictor importance of M3 (SHAP) under oracle labels.

Output: data/results/model_comparison_predictions.csv, model_comparison_metrics.csv,
        model_comparison_importance.csv and data/reports/model_comparison.md

    python 02_model_comparison.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
import models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    config.ensure_dirs()
    d = models.load()
    print(f"{len(d)} stations with Y1 ({d['grupo'].value_counts().to_dict()}); "
          f"{len(models.predictors(d))} predictors")
    preds, imps, logs = [], [], []
    for prot in models.PROTOCOLS:
        p, i, lg = models.rolling_origin(d, prot, args.min_train, args.min_test,
                                         with_importance=(prot == "oracle"))
        preds.append(p), imps.append(i), logs.append(lg)
        print(f"  {prot}: {lg['status'].eq('evaluated').sum()} quarters evaluated, "
              f"{p['estacion'].nunique() if len(p) else 0} stations")
    pred = pd.concat(preds, ignore_index=True)
    log = pd.concat(logs, ignore_index=True)
    imp = pd.concat(imps, ignore_index=True)
    pred.to_csv(config.RESULTS / "model_comparison_predictions.csv", index=False)
    if pred.empty:
        raise SystemExit("No quarter met the minimum sizes; check --min-train and --min-test.")
    met = models.metrics(pred, args.bootstrap, np.random.default_rng(2026))
    met.to_csv(config.RESULTS / "model_comparison_metrics.csv", index=False)

    r = ["# Model comparison", "",
         "Y1 = mean relative demand index r over periods 7-12. Rolling origin by opening quarter.", "",
         "capture20 = share of total demand held by the predicted top 20 % of stations; capture20_of_max = the "
         "same as a share of the maximum attainable; bias_pct = (mean prediction - mean observation) / mean "
         "observation x 100.", "",
         "## Quarters", log.to_markdown(index=False), ""]
    cols = ["model", "n", "MAE", "MAE_ci95", "RMSE", "spearman", "top20", "capture20", "capture20_of_max",
            "bias_pct", "p_holm_vs_best_baseline", "pct_fallback"]
    for prot in models.PROTOCOLS:
        for g in ("B1", "B2", "B1+B2"):
            s = met[(met["protocol"] == prot) & (met["grupo"] == g)]
            if len(s):
                r += [f"## {prot} labels, {g} (best baseline: {s['best_baseline'].iloc[0]})",
                      s[cols].to_markdown(index=False, floatfmt=".3f"), ""]
    if len(imp):
        imp.to_csv(config.RESULTS / "model_comparison_importance.csv", index=False)
        top = (imp.groupby(["grupo", "predictor"])["mean_abs_shap"].mean().reset_index()
               .sort_values(["grupo", "mean_abs_shap"], ascending=[True, False]).groupby("grupo").head(10))
        r += ["## Most important predictors (M3, mean absolute SHAP, oracle labels)",
              top.to_markdown(index=False, floatfmt=".4f"), ""]
    models.write_report("model_comparison.md", r)


if __name__ == "__main__":
    main()
