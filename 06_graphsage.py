"""Step 6. Graph neural network (GraphSAGE) over the station network.

Same rolling origin and label sets as step 2; the comparison with M3
(model_comparison_predictions.csv) uses a paired bootstrap.

Graph of each quarter: training stations plus the stations to predict; an edge joins stations
within 1 km (at most 16 neighbours), with weight exp(-d / 500 m). Variants:

  M4_sage      two-layer GraphSAGE on the predictors of each station (no neighbour labels).
  M4_sage_lt   the same plus the label trick: the known log Y1 of neighbouring training stations
               enters as an input. At each epoch half of the training labels are hidden at random
               and the loss is computed on the hidden ones; new stations never see their own label.
  M3+M4        mean in logs of M3 and M4_sage_lt (ensemble).

Each variant: 5 seeds, early stopping on 15 % of the training stations held out by 1 km blocks,
Duan's smearing.

Requires PyTorch (CPU is enough) and step 2.
Output: data/results/graphsage_predictions.csv, graphsage_metrics.csv and data/reports/graphsage.md

    python 06_graphsage.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config
import models

RADIUS_M, MAX_NEIGHBOURS, SCALE_M = 1000, 16, 500


def import_torch():
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is missing: pip install torch  (the CPU build is enough)")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


def graph(xy: np.ndarray, torch):
    """Row-normalised sparse adjacency matrix (without self-loops)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    rows, cols, weights = [], [], []
    dist, idx = tree.query(xy, k=min(MAX_NEIGHBOURS + 1, len(xy)), distance_upper_bound=RADIUS_M)
    for i in range(len(xy)):
        for d, j in zip(np.atleast_1d(dist[i]), np.atleast_1d(idx[i])):
            if j == i or not np.isfinite(d):
                continue
            rows.append(i), cols.append(j), weights.append(np.exp(-d / SCALE_M))
    rows, cols, weights = np.array(rows, int), np.array(cols, int), np.array(weights, float)
    total = np.bincount(rows, weights=weights, minlength=len(xy)) if len(rows) else np.zeros(len(xy))
    if len(rows):
        weights = weights / total[rows]
    A = torch.sparse_coo_tensor(np.vstack([rows, cols]), weights, (len(xy), len(xy)), dtype=torch.float32)
    has_neighbours = torch.tensor(total > 0, dtype=torch.float32).unsqueeze(1)
    return A.coalesce(), has_neighbours


def build_model(torch, n_in, hidden=64, dropout=0.2):
    nn = torch.nn

    class SAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1_self, self.l1_nb = nn.Linear(n_in, hidden), nn.Linear(n_in, hidden, bias=False)
            self.l2_self, self.l2_nb = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
            self.out = nn.Linear(hidden, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, A):
            h = torch.relu(self.l1_self(x) + self.l1_nb(torch.sparse.mm(A, x)))
            h = self.drop(h)
            h = torch.relu(self.l2_self(h) + self.l2_nb(torch.sparse.mm(A, h)))
            return self.out(self.drop(h)).squeeze(1)

    return SAGE()


def fit_predict(tr, te, cols, label_trick: bool, seeds, torch):
    Xtr, Xte = models.prepare_X(tr, te, cols)
    X = pd.concat([Xtr, Xte], ignore_index=True).to_numpy(float)
    mu, sd = Xtr.to_numpy(float).mean(0), Xtr.to_numpy(float).std(0) + 1e-9
    X = np.clip((X - mu) / sd, -5, 5)     # no extreme extrapolation outside the training range
    n_tr, n = len(tr), len(tr) + len(te)
    xy = np.vstack([tr[["x", "y"]].to_numpy(), te[["x", "y"]].to_numpy()])
    A, _ = graph(xy, torch)
    y = np.log(tr["Y1"].to_numpy())
    y_mu, y_sd = y.mean(), y.std() + 1e-9
    yz = torch.tensor((y - y_mu) / y_sd, dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    blocks = models.spatial_blocks(tr)
    preds_te, preds_tr = [], []
    for s in seeds:
        torch.manual_seed(s)
        rng = np.random.default_rng(s)
        unique = np.unique(blocks)
        val_blocks = set(rng.choice(unique, size=max(1, int(0.15 * len(unique))), replace=False))
        is_val = np.isin(blocks, list(val_blocks))
        idx_fit, idx_val = np.where(~is_val)[0], np.where(is_val)[0]
        n_in = X.shape[1] + (2 if label_trick else 0)
        model = build_model(torch, n_in)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

        def inputs(visible: np.ndarray):
            if not label_trick:
                return Xt
            label = torch.zeros(n)
            mask = torch.zeros(n)
            label[visible] = yz[visible]
            mask[visible] = 1.0
            # each station sees the weighted mean of the visible labels of its neighbours
            num = torch.sparse.mm(A, (label * mask).unsqueeze(1)).squeeze(1)
            den = torch.sparse.mm(A, mask.unsqueeze(1)).squeeze(1)
            mean = torch.where(den > 0, num / den.clamp(min=1e-9), torch.zeros_like(num))
            return torch.cat([Xt, mean.unsqueeze(1), (den > 0).float().unsqueeze(1)], dim=1)

        best, best_state, patience = np.inf, None, 0
        for _ in range(600):
            model.train()
            if label_trick:
                hidden = rng.random(len(idx_fit)) < 0.5
                visible, target = idx_fit[~hidden], idx_fit[hidden]
            else:
                visible, target = idx_fit, idx_fit
            opt.zero_grad()
            out = model(inputs(visible), A)
            loss = torch.nn.functional.smooth_l1_loss(out[target], yz[target])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                out = model(inputs(idx_fit), A)
                vl = torch.nn.functional.l1_loss(out[idx_val], yz[idx_val]).item() if len(idx_val) else loss.item()
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 50:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            all_tr = np.arange(n_tr)
            out = model(inputs(all_tr), A).numpy() * y_sd + y_mu
            preds_te.append(out[n_tr:])
            # training predictions without each station's own label, for Duan's smearing
            out_tr = np.empty(n_tr)
            if label_trick:
                for block in np.array_split(rng.permutation(n_tr), 5):
                    vis = np.setdiff1d(all_tr, block)
                    out_tr[block] = (model(inputs(vis), A).numpy() * y_sd + y_mu)[block]
            else:
                out_tr = out[:n_tr]
            preds_tr.append(out_tr)
    lo, hi = y.min() - 0.5, y.max() + 0.5        # bounded to the training range
    log_te = np.clip(np.nan_to_num(np.mean(preds_te, axis=0), nan=y_mu), lo, hi)
    log_tr = np.clip(np.nan_to_num(np.mean(preds_tr, axis=0), nan=y_mu), lo, hi)
    smearing = np.mean(np.exp(y - log_tr))
    return log_te, smearing


def run(d, protocol, min_train, min_test, seeds, torch):
    cols = models.predictors(d)
    rows = []
    for q in models.quarters(d):
        te = models.test_set(d, q).reset_index(drop=True)
        tr = models.training_set(d, q, protocol).reset_index(drop=True)
        if len(te) < min_test or len(tr) < min_train:
            continue
        print(f"    {protocol} {q}: training {len(tr)}, test {len(te)}", flush=True)
        for name, lt in (("M4_sage", False), ("M4_sage_lt", True)):
            log_p, sm = fit_predict(tr, te, cols, lt, seeds, torch)
            rows.append(models.prediction_table(protocol, q, name, te, np.exp(log_p) * sm))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def against_m3(pred: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    blocks = [((p, g), s) for (p, g), s in pred.groupby(["protocol", "grupo"])]
    blocks += [((p, "B1+B2"), s) for p, s in pred.groupby("protocol")]
    for (prot, group), s in blocks:
        piv = s.pivot_table(index="estacion", columns="model", values="pred", aggfunc="first")
        y = s.pivot_table(index="estacion", columns="model", values="y", aggfunc="first").iloc[:, 0]
        if "M3_lgbm" not in piv:
            continue
        err = piv.sub(y, axis=0).abs()
        n = len(y)
        boot = rng.integers(0, n, size=(n_boot, n))
        for m in [c for c in piv.columns if c.startswith("M4") or c == "M3+M4"]:
            delta = (err[m] - err["M3_lgbm"]).to_numpy()
            db = delta[boot].mean(axis=1)
            rows.append({"protocol": prot, "grupo": group, "model": m, "n": n, "MAE": err[m].mean(),
                         "MAE_M3": err["M3_lgbm"].mean(), "dMAE_vs_M3": delta.mean(),
                         "dMAE_ci95": f"[{np.quantile(db, 0.025):+.3f}, {np.quantile(db, 0.975):+.3f}]",
                         "p_wilcoxon": stats.wilcoxon(err[m], err["M3_lgbm"]).pvalue if n >= 6 else np.nan,
                         "spearman": stats.spearmanr(y, piv[m]).statistic,
                         "spearman_M3": stats.spearmanr(y, piv["M3_lgbm"]).statistic})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    torch = import_torch()
    config.ensure_dirs()
    path = config.RESULTS / "model_comparison_predictions.csv"
    if not path.exists():
        raise SystemExit("Missing model_comparison_predictions.csv: run 02_model_comparison.py first")
    d = models.load()
    seeds = tuple(range(args.seeds))
    m4 = pd.concat([run(d, p, args.min_train, args.min_test, seeds, torch) for p in models.PROTOCOLS],
                   ignore_index=True)
    if m4.empty:
        raise SystemExit("No quarter met the minimum sizes.")
    base = pd.read_csv(path, dtype={"estacion": str})
    m4["estacion"] = m4["estacion"].astype(str)
    # ensemble of M3 and M4_sage_lt (mean in logs)
    a = base[base["model"] == "M3_lgbm"].set_index(["protocol", "estacion"])["pred"]
    b = m4[m4["model"] == "M4_sage_lt"].set_index(["protocol", "estacion"])
    ens = b.copy()
    ens["pred"] = np.exp((np.log(b["pred"]) + np.log(a.reindex(b.index))) / 2)
    ens["model"] = "M3+M4"
    m4 = pd.concat([m4, ens.reset_index()], ignore_index=True)
    m4.to_csv(config.RESULTS / "graphsage_predictions.csv", index=False)
    everything = pd.concat([base, m4], ignore_index=True)
    rng = np.random.default_rng(2026)
    met = models.metrics(everything, args.bootstrap, rng)
    met.to_csv(config.RESULTS / "graphsage_metrics.csv", index=False)
    cmp = against_m3(everything, args.bootstrap, rng)

    r = ["# GraphSAGE models", "",
         "Same rolling origin as step 2. dMAE_vs_M3 < 0 with a CI that excludes 0 means that the variant "
         "improves on M3.", "",
         "## Direct comparison with M3", cmp.to_markdown(index=False, floatfmt=".3f"), ""]
    cols = ["model", "n", "MAE", "MAE_ci95", "RMSE", "spearman", "top20", "capture20_of_max", "bias_pct",
            "p_holm_vs_best_baseline"]
    for prot in models.PROTOCOLS:
        s = met[(met["protocol"] == prot) & (met["grupo"] == "B1+B2")]
        if len(s):
            r += [f"## All models, {prot} labels, B1+B2", s[cols].to_markdown(index=False, floatfmt=".3f"), ""]
    models.write_report("graphsage.md", r)


if __name__ == "__main__":
    main()
