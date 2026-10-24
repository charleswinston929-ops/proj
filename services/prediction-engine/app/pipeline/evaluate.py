"""Evaluation of the PRODUCTION model (uncalibrated linear pool) from the walk-forward predictions. Writes artifacts/evaluation.json."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from ..data import build_dataset
from ..core import metrics as M
from .predict import ART
TEST = pd.Timestamp("2019-08-01")

def probs(bt, c): return bt[[f"p_{c}_H", f"p_{c}_D", f"p_{c}_A"]].values

def main():
    bt = pd.read_csv(ART / "backtest_predictions.csv", parse_dates=["date"]); bj = json.loads((ART / "backtest.json").read_text())
    test = bt[bt.date >= TEST]; y = test.y.values; out = {"n_test": int(len(test)), "test_from": str(TEST.date()), "test_to": str(test.date.max().date())}
    out["test"] = {c: M.summary(probs(test, c), y) for c in ("clim", "elo", "pois", "dc", "gb", "ensemble_raw", "ensemble")}
    val = bt[bt.date < TEST]; out["validation"] = {c: M.summary(probs(val, c), val.y.values) for c in ("clim", "elo", "pois", "dc", "gb", "ensemble_raw")}
    out["seasons"] = {s: {c: M.logloss(probs(g, c), g.y.values) for c in ("clim", "elo", "dc", "ensemble_raw")} | {"n": int(len(g))} for s, g in bt.groupby("season")}
    P = probs(test, "ensemble_raw"); O = M.onehot(y); out["reliability"] = {k: M.reliability(P[:, i], O[:, i]) for i, k in enumerate(("home", "draw", "away"))}
    le = M.per_match_logloss(P, y); out["bootstrap_vs"] = {c: M.bootstrap_diff(le, M.per_match_logloss(probs(test, c), y)) for c in ("clim", "elo", "pois", "dc", "gb")}
    dis = test.disagreement.values; r = spearmanr(dis, le); terc = pd.qcut(dis, 3, labels=False)
    out["disagreement"] = dict(spearman=float(r.correlation), p=float(r.pvalue), terciles=[dict(mean=float(dis[terc == t].mean()), log_loss=float(le[terc == t].mean()), n=int((terc == t).sum())) for t in range(3)])
    df = build_dataset(data_dir="../../data/raw")[0]; Pl = df[df.played]
    def n2y(team, d): m = Pl[(Pl.date < d) & (Pl.date >= d - pd.Timedelta(days=730))]; return int(((m.home == team) | (m.away == team)).sum())
    new = np.array([min(n2y(r.home, r.date), n2y(r.away, r.date)) < 15 for r in test.itertuples()])
    out["newcomers"] = {k: dict(n=int(m.sum()), log_loss=float(M.logloss(P[m], y[m])), clim=float(M.logloss(probs(test, "clim")[m], y[m]))) for k, m in (("newcomer_involved", new), ("established", ~new))}
    cal = bj["metrics"]["test"]; out["calibration_check"] = dict(raw=dict(log_loss=out["test"]["ensemble_raw"]["log_loss"], ece=out["test"]["ensemble_raw"]["ece"]), calibrated=dict(log_loss=cal["ensemble"]["log_loss"], ece=cal["ensemble"]["ece"]), decision="calibrator disabled in production: no out-of-sample benefit (decision made after viewing test results)")
    out["weights"] = bj["weights"]; out["tuned"] = bj["tuned"]
    (ART / "evaluation.json").write_text(json.dumps(out, indent=1)); 
    for c, s in out["test"].items(): print(f"{c:13s} ll={s['log_loss']:.4f} brier={s['brier']:.4f} rps={s['rps']:.4f} acc={s['accuracy']:.3f} ece={s['ece']:.4f}")
    print({k: [round(v["mean"], 4), round(v["lo"], 4), round(v["hi"], 4)] for k, v in out["bootstrap_vs"].items()}); print(out["newcomers"]); print(out["disagreement"]["spearman"], out["disagreement"]["p"])
if __name__ == "__main__": main()
