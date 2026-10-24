"""Walk-forward backtest (spec sections 65-67). Hyper-parameters are tuned ONLY on the validation window;
the test window (2019-20 onward) is never used for any tuning. Models are refit at the start of every calendar month
using strictly earlier matches. Run: python -m app.pipeline.backtest"""
from __future__ import annotations
import json, time, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from ..data import build_dataset
from ..core.elo import Elo, OrderedLogit
from ..core.dixon_coles import DixonColes, score_matrix, outcome_probs
from ..core import ensemble as ens, metrics as M
from ..features.engine import run_state, FEATURE_VERSION

VAL_START, TEST_START, WARM_END = pd.Timestamp("2014-08-01"), pd.Timestamp("2019-08-01"), pd.Timestamp("2011-08-01")
GB_PARAMS = dict(max_depth=3, learning_rate=0.04, max_iter=200, l2_regularization=2.0, min_samples_leaf=40, random_state=0)
OUT = Path("../../artifacts"); OUT.mkdir(exist_ok=True, parents=True)

def outcome(df): return np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))

def elo_x(P, params, teams):
    e = Elo(**params); xs = np.zeros(len(P))
    for date, g in P.groupby("date", sort=True):
        for i, r in g.iterrows():
            e.new_season(r.season, teams[r.season]) if e.season != r.season else None
            xs[P.index.get_loc(i)] = (e.rating(r.home) - e.rating(r.away) + e.h) / 400.0
        for i, r in g.iterrows(): e.update(r.home, r.away, int(r.hg), int(r.ag))
    return xs

def tune_elo(P, teams, y):
    grid = [dict(k=k, home_adv=h, regress=r, newcomer=n) for k, h, r, n in itertools.product((14, 20, 28), (50, 75), (0.2, 0.35), (1400, 1450))]
    tr = (P.date < VAL_START).values; va = ((P.date >= VAL_START) & (P.date < TEST_START)).values; best = None
    for g in grid:
        x = elo_x(P, g, teams); m = OrderedLogit().fit(x[tr & (P.date >= WARM_END).values], y[tr & (P.date >= WARM_END).values])
        ll = M.logloss(m.proba(x[va]), y[va])
        if best is None or ll < best[0]: best = (ll, g)
    return best[1], best[0]

def months(P, start):
    d = P[P.date >= start].date; return sorted({(x.year, x.month) for x in d})

def tune_dc(P, y):
    best = None
    for hl, rg in itertools.product((180, 365, 540, 900), (1.0, 3.0, 8.0)):
        pr, yy = [], []
        for (yr, mo) in months(P[P.date < TEST_START], VAL_START):
            b0 = pd.Timestamp(yr, mo, 1); tr = P[P.date < b0]; blk = P[(P.date >= b0) & (P.date < b0 + pd.offsets.MonthBegin(1))]
            if blk.empty: continue
            m = DixonColes(hl, rg).fit(tr.home, tr.away, tr.hg, tr.ag, (b0 - tr.date).dt.days)
            for r in blk.itertuples(): pr.append(outcome_probs(score_matrix(*m.lambdas(r.home, r.away), m.rho)))
            yy += list(y[P.index.get_indexer(blk.index)])
        ll = M.logloss(np.array(pr), np.array(yy))
        if best is None or ll < best[0]: best = (ll, hl, rg)
    return best

def main():
    t0 = time.time(); df, res, rep = build_dataset(data_dir="../../data/raw")
    P = df[df.played].copy(); teams = df.groupby("season").apply(lambda g: sorted(set(g.home) | set(g.away))).to_dict(); y = outcome(P)
    print("rows", len(P), flush=True)
    elo_p, elo_ll = tune_elo(P, teams, y); print("elo", elo_p, round(elo_ll, 4), f"{time.time()-t0:.0f}s", flush=True)
    dc_ll, hl, rg = tune_dc(P, y); print("dc", hl, rg, round(dc_ll, 4), f"{time.time()-t0:.0f}s", flush=True)
    X, st, _ = run_state(df, elo_p); X = X.loc[P.index]
    cols = ["clim", "elo", "pois", "dc", "gb"]; pred = {c: np.zeros((len(P), 3)) for c in cols}; lam = np.zeros((len(P), 2)); done = np.zeros(len(P), bool)
    for (yr, mo) in months(P, VAL_START):
        b0 = pd.Timestamp(yr, mo, 1); b1 = b0 + pd.offsets.MonthBegin(1)
        trm = (P.date < b0).values; blm = ((P.date >= b0) & (P.date < b1)).values
        if not blm.any(): continue
        tr = P[trm]; blk = P[blm]; ytr = y[trm]
        pred["clim"][blm] = np.bincount(ytr, minlength=3) / len(ytr)
        w_ok = trm & (P.date >= WARM_END).values
        om = OrderedLogit().fit(X.loc[P.index[w_ok], "elo_x"].values, y[w_ok]); pred["elo"][blm] = om.proba(X.loc[blk.index, "elo_x"].values)
        m = DixonColes(hl, rg).fit(tr.home, tr.away, tr.hg, tr.ag, (b0 - tr.date).dt.days)
        for j, r in zip(np.where(blm)[0], blk.itertuples()):
            l = m.lambdas(r.home, r.away); lam[j] = l
            pred["dc"][j] = outcome_probs(score_matrix(*l, m.rho)); pred["pois"][j] = outcome_probs(score_matrix(*l, 0.0))
        sw = 0.5 ** ((b0 - P.date[w_ok]).dt.days.values / 1500.0)
        gb = HistGradientBoostingClassifier(**GB_PARAMS).fit(X.loc[P.index[w_ok]], y[w_ok], sample_weight=sw)
        pred["gb"][blm] = gb.predict_proba(X.loc[blk.index]); done |= blm
    print("walk-forward done", f"{time.time()-t0:.0f}s", flush=True)
    d = P[done]; yd = y[done]; PR = {c: v[done] for c, v in pred.items()}; lamd = lam[done]
    val = (d.date < TEST_START).values; test = ~val
    w = ens.fit_weights([PR["elo"][val], PR["dc"][val], PR["gb"][val]], yd[val])
    pool_all = ens.pool([PR["elo"], PR["dc"], PR["gb"]], w); cal = ens.Calibrator().fit(pool_all[val], yd[val]); PR["ensemble_raw"] = pool_all; PR["ensemble"] = cal.apply(pool_all)
    dis = ens.disagreement([PR["elo"], PR["dc"], PR["gb"]])
    results = {"test": {c: M.summary(v[test], yd[test]) for c, v in PR.items()}, "validation": {c: M.summary(v[val], yd[val]) for c, v in PR.items()}}
    seasons = {}
    for s, g in d.groupby("season"):
        mk = (d.season == s).values
        seasons[s] = {c: M.summary(PR[c][mk], yd[mk]) for c in ("clim", "elo", "dc", "gb", "ensemble")}
    ll_e = M.per_match_logloss(PR["ensemble"][test], yd[test])
    boot = {c: M.bootstrap_diff(ll_e, M.per_match_logloss(PR[c][test], yd[test])) for c in ("clim", "elo", "pois", "dc", "gb")}
    O = M.onehot(yd[test]); rel = {k: M.reliability(PR["ensemble"][test][:, i], O[:, i]) for i, k in enumerate(("home", "draw", "away"))}
    from scipy.stats import spearmanr
    rho_dis = spearmanr(dis[test], ll_e); terc = pd.qcut(dis[test], 3, labels=False)
    dis_tbl = [dict(tercile=int(t), mean_disagreement=float(dis[test][terc == t].mean()), log_loss=float(ll_e[terc == t].mean()), n=int((terc == t).sum())) for t in range(3)]
    out = dict(feature_version=FEATURE_VERSION, windows=dict(validation=[str(VAL_START.date()), str(TEST_START.date())], test_from=str(TEST_START.date())),
               tuned=dict(elo=elo_p, dc=dict(half_life_days=hl, ridge=rg), gb=GB_PARAMS), weights=dict(zip(("elo", "dc", "gb"), map(float, w))),
               calibrator=cal.to_dict(), metrics=results, seasons=seasons, bootstrap_vs=boot, reliability=rel,
               disagreement=dict(spearman_vs_logloss=float(rho_dis.correlation), p=float(rho_dis.pvalue), terciles=dis_tbl),
               n_predictions=int(done.sum()), data_report=rep["seasons"], runtime_s=round(time.time() - t0))
    (OUT / "backtest.json").write_text(json.dumps(out, indent=1))
    bt = d[["match_id", "season", "date", "home", "away", "hg", "ag"]].copy(); bt["y"] = yd; bt["lam_h"], bt["lam_a"] = lamd[:, 0], lamd[:, 1]
    for c in ("clim", "elo", "pois", "dc", "gb", "ensemble_raw", "ensemble"):
        for i, k in enumerate("HDA"): bt[f"p_{c}_{k}"] = PR[c][:, i]
    bt["disagreement"] = dis; bt.to_csv(OUT / "backtest_predictions.csv", index=False)
    print(json.dumps({k: out[k] for k in ("tuned", "weights", "calibrator")}, indent=1))
    for c in ("clim", "elo", "pois", "dc", "gb", "ensemble_raw", "ensemble"):
        s = results["test"][c]; print(f"TEST {c:13s} n={s['n']} logloss={s['log_loss']:.4f} brier={s['brier']:.4f} rps={s['rps']:.4f} acc={s['accuracy']:.3f} ece={s['ece']:.4f}")
    print("bootstrap ensemble minus X (neg = ensemble better):", {k: (round(v['mean'], 4), round(v['lo'], 4), round(v['hi'], 4)) for k, v in boot.items()})
    print("disagreement:", out["disagreement"])

if __name__ == "__main__": main()
