"""Build all artifacts for the UI/DB: live forecasts for upcoming fixtures, reconstructed as-of history, lineage, team table."""
import json, sys, time, pickle
import numpy as np, pandas as pd
from pathlib import Path
from ..data import build_dataset
from .predict import load_cfg, fit_context, predict, lineage, ART, MODEL_NAME, MODEL_VERSION

TODAY = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now(tz="UTC").date())
HIST_ASOF = ["2026-08-20", "2026-08-27", "2026-09-03", "2026-09-10", "2026-09-17", "2026-09-21"]

def main():
    t0 = time.time(); df, res, rep = build_dataset(data_dir="../../data/raw"); cfg = load_cfg()
    upcoming = df[(~df.played) & (df.date >= TODAY)].copy(); stale = df[(~df.played) & (df.date < TODAY)].copy()
    ctx = fit_context(df, TODAY, cfg); live = predict(ctx, df, upcoming, cfg, "live")
    print("live forecasts", len(live), "awaiting result", len(stale), f"{time.time()-t0:.0f}s", flush=True)
    focus = upcoming[upcoming.date <= TODAY + pd.Timedelta(days=14)]; hist = {m: [] for m in focus.match_id}
    for a in HIST_ASOF:
        c = fit_context(df, pd.Timestamp(a), cfg, B=20); fx = focus[focus.date >= pd.Timestamp(a)]
        for s in predict(c, df, fx, cfg, "reconstructed", n_sims=4000): hist[s["match_id"]].append(s)
        print("as-of", a, f"{time.time()-t0:.0f}s", flush=True)
    byid = {s["match_id"]: s for s in live}
    for m, snaps in hist.items():
        snaps.append(byid[m]); snaps.sort(key=lambda s: s["as_of"])
        for prev, cur in zip(snaps[:-1], snaps[1:]): cur["lineage"] = lineage(prev, cur, df); cur["previous_as_of"] = prev["as_of"]
    teams = []
    for t, name in res.canonical.items():
        if t in ctx["dc"].att and t in ctx["state"].elo.r:
            h = ctx["state"].hist[t][-5:]; teams.append(dict(key=t, name=name, elo=ctx["state"].elo.r[t], attack=ctx["dc"].att[t], defence=ctx["dc"].dfn[t], form="".join("W" if e["pts"] == 3 else "D" if e["pts"] == 1 else "L" for e in h), season_active=bool(t in set(upcoming.home) | set(upcoming.away))))
    teams.sort(key=lambda x: -x["elo"])
    slim = lambda s: {k: v for k, v in s.items() if k not in ("features",)}
    app = dict(generated_at=pd.Timestamp.now(tz="UTC").isoformat(), as_of=str(TODAY.date()), model=dict(name=MODEL_NAME, version=MODEL_VERSION, feature_version=cfg["feature_version"], weights=cfg["weights"], tuned=cfg["tuned"]),
               data=dict(source="openfootball (github.com/openfootball/england)", last_result=str(ctx["played"].date.max().date()), seasons=len(rep["seasons"]), matches_played=int(df.played.sum()), awaiting_result=len(stale),
                         awaiting=[dict(home=res.canonical[r.home], away=res.canonical[r.away], date=str(r.date.date())) for r in stale.itertuples()][:30], anomalies=rep["anomalies"]),
               names=res.canonical, teams=teams, upcoming=[slim(s) for s in live], history={m: [slim(s) for s in v] for m, v in hist.items()}, backtest=cfg)
    (ART / "app_data.json").write_text(json.dumps(app, default=float))
    pickle.dump(dict(live=live, hist=hist), open(ART / "snapshots.pkl", "wb")); print("done", f"{time.time()-t0:.0f}s", len(json.dumps(app, default=float)) // 1024, "KB")
if __name__ == "__main__": main()
