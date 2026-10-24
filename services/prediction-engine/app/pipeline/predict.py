"""Production forecasts: refit everything strictly as-of a date, forecast fixtures, simulate, score data quality, build lineage."""
from __future__ import annotations
import json, copy
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from ..core.elo import OrderedLogit
from ..core.dixon_coles import DixonColes, score_matrix, outcome_probs
from ..core import ensemble as ens, quality as Q
from ..core.simulate import bootstrap_models, simulate_fixture
from ..features.engine import run_state, forecast_features, FEATURE_VERSION

WARM_END = pd.Timestamp("2011-08-01"); MODEL_NAME, MODEL_VERSION = "ensemble", "1.0.0"
ART = Path("../../artifacts")

def load_cfg(): return json.loads((ART / "backtest.json").read_text())

def fit_context(df, as_of: pd.Timestamp, cfg, B=40):
    X, st, teams = run_state(df, cfg["tuned"]["elo"], until=as_of)
    played = df[df.played & (df.date < as_of)]; y = np.where(played.hg > played.ag, 0, np.where(played.hg == played.ag, 1, 2))
    ok = (played.date >= WARM_END).values
    om = OrderedLogit().fit(X.loc[played.index[ok], "elo_x"].values, y[ok])
    dcp = cfg["tuned"]["dc"]; dc = DixonColes(dcp["half_life_days"], dcp["ridge"]).fit(played.home, played.away, played.hg, played.ag, (as_of - played.date).dt.days)
    sw = 0.5 ** ((as_of - played.date[ok]).dt.days.values / 1500.0)
    gb = HistGradientBoostingClassifier(**cfg["tuned"]["gb"]).fit(X.loc[played.index[ok]], y[ok], sample_weight=sw)
    boots = bootstrap_models(played, as_of, dcp["half_life_days"], dcp["ridge"], B)
    return dict(as_of=as_of, state=st, teams=teams, om=om, dc=dc, gb=gb, boots=boots, played=played, X=X)

def predict(ctx, df, fixtures: pd.DataFrame, cfg, kind="live", n_sims=20000):
    as_of = ctx["as_of"]; w = cfg["weights"]; wv = [w["elo"], w["dc"], w["gb"]]; dc = ctx["dc"]
    F = forecast_features(ctx["state"], ctx["teams"], fixtures)
    p_elo = ctx["om"].proba(F["elo_x"].values); p_gb = ctx["gb"].predict_proba(F)
    cutoff = ctx["played"].date.max() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    miss = df[(~df.played) & (df.date < as_of)]; mcount = pd.concat([miss.home, miss.away]).value_counts().to_dict()
    recent = ctx["played"][ctx["played"].date >= as_of - pd.Timedelta(days=730)]; ncount = pd.concat([recent.home, recent.away]).value_counts().to_dict()
    out = []
    for j, (idx, r) in enumerate(fixtures.iterrows()):
        lh, la = dc.lambdas(r.home, r.away); mx = score_matrix(lh, la, dc.rho); p_dc = outcome_probs(mx); p_pois = outcome_probs(score_matrix(lh, la, 0.0))
        comps = [p_elo[j], p_dc, p_gb[j]]; p = ens.pool([c[None] for c in comps], wv)[0]
        sim = simulate_fixture(ctx["boots"], r.home, r.away, n_sims, seed=11 + j)
        dis = float(np.stack(comps).std(0).mean())
        missing = mcount.get(r.home, 0) + mcount.get(r.away, 0)
        parts = dict(freshness=Q.freshness_score(missing), completeness=Q.completeness_score(ncount.get(r.home, 0), ncount.get(r.away, 0), 0.0),
                     reliability=Q.SOURCE_RELIABILITY["openfootball"], entity=1.0, structural=1.0)
        dq = Q.data_quality(**parts); k = int(np.argmax(p)); width = sim["interval_90"][("home", "draw", "away")[k]]; width = width[1] - width[0]
        conf = Q.confidence(dq, dis, width); published = Q.dq_band(dq) != "blocked"
        feat = {k_: float(v) for k_, v in F.loc[idx].items()}
        out.append(dict(match_id=r.match_id, as_of=str(as_of.date()), kind=kind, season=r.season, matchday=None if pd.isna(r.matchday) else int(r.matchday),
            home=r.home, away=r.away, date=str(r.date.date()), kickoff_utc=r.kickoff_utc.isoformat(), data_cutoff=cutoff.isoformat(),
            probs=dict(home=float(p[0]), draw=float(p[1]), away=float(p[2])),
            components=dict(elo=p_elo[j].tolist(), dc=p_dc.tolist(), gb=p_gb[j].tolist(), pois=p_pois.tolist()),
            lam_h=lh, lam_a=la, rho=dc.rho, home_adv_log=dc.h, weights=w, calibrator=dict(logT=0.0, b=[0.0, 0.0]),
            published=published, disagreement=dis, data_quality=dq, dq_parts=parts, dq_band=Q.dq_band(dq), missing_results=missing, confidence=conf,
            simulation=sim, features=feat, feature_version=FEATURE_VERSION,
            strengths=dict(elo_h=feat["elo_h"], elo_a=feat["elo_a"], att_h=dc.att.get(r.home, dc.new_att), def_h=dc.dfn.get(r.home, dc.new_def),
                           att_a=dc.att.get(r.away, dc.new_att), def_a=dc.dfn.get(r.away, dc.new_def))))
    return out

def lineage(prev: dict, cur: dict, df) -> list[dict]:
    """Explain a forecast change with exact component deltas + the new results that entered the data in between (evidence, not per-event attribution)."""
    d = lambda a, b: 100 * (b - a)
    dp = d(prev["probs"]["home"], cur["probs"]["home"]); items = []
    ce = d(prev["components"]["elo"][0], cur["components"]["elo"][0]); cd = d(prev["components"]["dc"][0], cur["components"]["dc"][0]); cg = d(prev["components"]["gb"][0], cur["components"]["gb"][0])
    s0, s1 = prev["strengths"], cur["strengths"]
    new = df[df.played & (df.date >= pd.Timestamp(prev["as_of"])) & (df.date < pd.Timestamp(cur["as_of"])) & (df.home.isin([cur["home"], cur["away"]]) | df.away.isin([cur["home"], cur["away"]]))]
    for r in new.itertuples():
        items.append(dict(event_type="result", explanation=f"New result: {r.home} {int(r.hg)}-{int(r.ag)} {r.away} ({r.date.date()})", impact_score=None))
    if abs(s1["elo_h"] - s0["elo_h"]) + abs(s1["elo_a"] - s0["elo_a"]) > 0.05:
        items.append(dict(event_type="elo_change", explanation=f"Elo: {cur['home']} {s0['elo_h']:.0f}→{s1['elo_h']:.0f}, {cur['away']} {s0['elo_a']:.0f}→{s1['elo_a']:.0f}; Elo component P(home) {ce:+.1f}pp", impact_score=abs(ce)))
        items.append(dict(event_type="goal_model_change", explanation=f"Goal model: expected goals {prev['lam_h']:.2f}-{prev['lam_a']:.2f} → {cur['lam_h']:.2f}-{cur['lam_a']:.2f}; P(home) {cd:+.1f}pp", impact_score=abs(cd)))
    if cur["missing_results"] != prev["missing_results"]:
        items.append(dict(event_type="data_gap", explanation=f"{cur['missing_results']} results missing for these teams in the source (was {prev['missing_results']}); forecast unchanged by them, data quality {prev['data_quality']:.2f}→{cur['data_quality']:.2f}", impact_score=0.0))
    if not items: items.append(dict(event_type="no_change", explanation="No new information entered the model between these snapshots.", impact_score=0.0))
    return [dict(delta_home_pp=dp, **i) for i in items]
