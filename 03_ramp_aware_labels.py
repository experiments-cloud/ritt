"""Step 3. Prediction with the information available at each decision date.

Factorial design:
  label strategy        x  learner
  {"mature", "ramp"}       {M0_k5, M0_rule, M3_lgbm}

  mature: training uses only mature stations (the same set as the `mature` labels of step 2).
  ramp:   each maturing station is added with a provisional Y1 from chain-ladder development
          factors (ramp.py) and a weight equal to the share of its development already observed.

Test: the expansion stations (B1, B2) of each quarter with their final Y1. Every quarter with at
least 40 labelled stations and 5 test stations is evaluated. The provisional labels are also
compared with the final Y1.

Output: data/results/ramp_predictions.csv, ramp_metrics.csv, ramp_labels.csv
        and data/reports/ramp_aware_labels.md

    python 03_ramp_aware_labels.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
import models
import ramp


def run(d, per, y_final, min_train, min_test):
    cols = models.predictors(d)
    preds, log, diag = [], [], []
    for q in models.quarters(d):
        te = ramp.test_set(d, q, y_final)
        mature, prov, _ = ramp.labels(d, per, q.start_time, y_final)
        for _, f in prov.iterrows():
            diag.append({"quarter": str(q), "estacion": f["estacion"], "grupo": f["grupo"], "k": f["k"],
                         "weight": f["weight"], "provisional": f["Y1"], "final": y_final.get(f["estacion"], np.nan)})
        sets = {"mature": mature, "ramp": pd.concat([mature, prov], ignore_index=True)}
        for strategy, tr in sets.items():
            ok = len(tr) >= min_train and len(te) >= min_test
            log.append({"quarter": str(q), "strategy": strategy, "n_test": len(te),
                        "n_mature": len(mature), "n_provisional": len(prov) if strategy == "ramp" else 0,
                        "n_provisional_expansion": int(prov["grupo"].isin(["B1", "B2"]).sum()) if strategy == "ramp" else 0,
                        "status": "evaluated" if ok else "skipped"})
            if not ok:
                continue
            for name, p in ramp.fit(tr, te, cols).items():
                preds.append(models.prediction_table(strategy, q, name, te, p))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(log), pd.DataFrame(diag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    config.ensure_dirs()
    d, curves, per = ramp.read_inputs()
    y_final = curves.set_index("estacion")["Y1"].where(lambda s: s > 0).dropna().to_dict()
    pred, log, diag = run(d, per, y_final, args.min_train, args.min_test)
    pred.to_csv(config.RESULTS / "ramp_predictions.csv", index=False)
    diag.to_csv(config.RESULTS / "ramp_labels.csv", index=False)
    rng = np.random.default_rng(2026)
    met = models.metrics(pred, args.bootstrap, rng)
    met.to_csv(config.RESULTS / "ramp_metrics.csv", index=False)
    cmp = ramp.compare(pred, args.bootstrap, rng)

    r = ["# Mature-only vs ramp-aware labels at the decision date", "",
         "Factorial design: label strategy {mature, ramp} x learner {M0_k5, M0_rule, M3_lgbm}. Ramp = mature "
         "stations plus provisional labels from chain-ladder factors (separate triangles for A and the expansion; "
         f"at least {ramp.MIN_PAIRS} pairs per factor), weighted by the observed development.", "",
         "## 1. Quarters and available labels", log.to_markdown(index=False), ""]
    if len(diag):
        dg = diag.dropna(subset=["final"]).copy()
        dg["k_bin"] = pd.cut(dg["k"], [0, 1, 2, 3, 6, 11], labels=["1", "2", "3", "4-6", "7-11"])
        dg["err"] = (dg["provisional"] - dg["final"]).abs()
        dg["bias"] = dg["provisional"] / dg["final"] - 1
        t = dg.groupby("k_bin", observed=True).agg(n=("err", "size"), MAE=("err", "mean"),
                                                   median_bias_pct=("bias", lambda x: 100 * x.median()),
                                                   mean_weight=("weight", "mean"))
        r += ["## 2. Provisional labels against the final Y1",
              "By number of periods observed at the decision date (all stations and quarters).", "",
              t.to_markdown(floatfmt=".3f"), ""]
    r += ["## 3. Ramp-aware vs mature-only labels (same learner, same stations)",
          "dMAE < 0 with a CI that excludes 0: the provisional labels improve the prediction.", "",
          cmp.to_markdown(index=False, floatfmt=".3f"), ""]
    cols = ["model", "n", "MAE", "MAE_ci95", "spearman", "top20", "capture20_of_max", "bias_pct",
            "p_holm_vs_best_baseline"]
    for strategy in ("mature", "ramp"):
        for g in ("B1", "B2", "B1+B2"):
            s = met[(met["protocol"] == strategy) & (met["grupo"] == g)]
            if len(s):
                r += [f"## 4. {strategy} labels, {g} (all evaluable quarters)",
                      s[cols].to_markdown(index=False, floatfmt=".3f"), ""]
    models.write_report("ramp_aware_labels.md", r)


if __name__ == "__main__":
    main()
