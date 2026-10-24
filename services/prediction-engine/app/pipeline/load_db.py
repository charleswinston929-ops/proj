"""Load canonical data + immutable prediction history into PostgreSQL. Run: DATABASE_URL=... python -m app.pipeline.load_db [--reset]"""
import json, os, pickle, sys, pandas as pd, psycopg2
from psycopg2.extras import execute_values, Json
from ..data import build_dataset
from .predict import ART, MODEL_NAME, MODEL_VERSION

def main():
    df, res, rep = build_dataset(data_dir="../../data/raw"); bt = json.loads((ART / "backtest.json").read_text()); snaps = pickle.load(open(ART / "snapshots.pkl", "rb"))
    app = json.loads((ART / "app_data.json").read_text()); today = pd.Timestamp(app["as_of"], tz="UTC")
    con = psycopg2.connect(os.environ["DATABASE_URL"]); cur = con.cursor()
    if "--reset" in sys.argv:
        cur.execute("TRUNCATE sports, data_sources, teams, competitions, seasons, matches, model_versions, audit_logs RESTART IDENTITY CASCADE")
    q1 = lambda sql, a=(): (cur.execute(sql, a), cur.fetchone()[0])[1]
    sport = q1("INSERT INTO sports(name,code) VALUES ('Football','football') ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name RETURNING id")
    src = q1("INSERT INTO data_sources(name,provider_type,tier,reliability_score,api_base_url) VALUES ('openfootball','public_dataset',3,0.80,'https://github.com/openfootball/england') ON CONFLICT (name) DO UPDATE SET active=TRUE RETURNING id")
    comp = q1("INSERT INTO competitions(sport_id,name,country,code) VALUES (%s,'Premier League','England','PL') ON CONFLICT (sport_id,code) DO UPDATE SET name=EXCLUDED.name RETURNING id", (sport,))
    seas = {}
    for s, g in df.groupby("season"): seas[s] = q1("INSERT INTO seasons(competition_id,name,start_date,end_date) VALUES (%s,%s,%s,%s) ON CONFLICT (competition_id,name) DO UPDATE SET end_date=EXCLUDED.end_date RETURNING id", (comp, s, g.date.min().date(), g.date.max().date()))
    team = {k: q1("INSERT INTO teams(sport_id,canonical_key,name,country) VALUES (%s,%s,%s,'England') ON CONFLICT (sport_id,canonical_key) DO UPDATE SET name=EXCLUDED.name RETURNING id", (sport, k, n)) for k, n in res.canonical.items()}
    execute_values(cur, "INSERT INTO entity_aliases(entity_type,entity_id,source_id,alias) VALUES %s ON CONFLICT DO NOTHING", [("team", team[k], src, a) for a, k in res.aliases.items()])
    rows = []
    for r in df.itertuples():
        st = "finished" if r.played else ("awaiting_result" if r.date < today.tz_localize(None) else "scheduled")
        rows.append((comp, seas[r.season], team[r.home], team[r.away], None if pd.isna(r.matchday) else int(r.matchday), r.kickoff_utc, st, int(r.hg) if r.played else None, int(r.ag) if r.played else None, src, r.match_id))
    execute_values(cur, "INSERT INTO matches(competition_id,season_id,home_team_id,away_team_id,matchday,kickoff_at,status,home_score,away_score,source_id,natural_key) VALUES %s ON CONFLICT (natural_key) DO UPDATE SET status=EXCLUDED.status, home_score=EXCLUDED.home_score, away_score=EXCLUDED.away_score, updated_at=now()", rows, page_size=500)
    cur.execute("SELECT natural_key,id FROM matches"); mid = dict(cur.fetchall())
    cur.execute("UPDATE model_versions SET status='retired' WHERE status='active'")
    mv = q1("INSERT INTO model_versions(name,version,model_type,training_start,training_end,feature_version,hyperparameters,metrics,status) VALUES (%s,%s,'elo+poisson+gbm pool',%s,%s,%s,%s,%s,'active') ON CONFLICT (name,version) DO UPDATE SET status='active', metrics=EXCLUDED.metrics RETURNING id",
            (MODEL_NAME, MODEL_VERSION, '2010-08-01', app["data"]["last_result"], bt["feature_version"], Json(dict(tuned=bt["tuned"], weights=bt["weights"])), Json(bt["metrics"]["test"])))
    for w, (a, b) in (("validation", bt["windows"]["validation"]), ("test", (bt["windows"]["test_from"], app["data"]["last_result"]))):
        m = bt["metrics"][w]["ensemble_raw"]; cur.execute("INSERT INTO model_evaluations(model_version_id,window_name,period_start,period_end,sample_size,log_loss,brier_score,rps,accuracy,calibration_error,metrics) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (mv, w, a, b, m["n"], m["log_loss"], m["brier"], m["rps"], m["accuracy"], m["ece"], Json(bt["metrics"][w])))
    btp = pd.read_csv(ART / "backtest_predictions.csv")
    execute_values(cur, "INSERT INTO backtest_predictions(match_id,model_version_id,p_home,p_draw,p_away,components,disagreement) VALUES %s ON CONFLICT DO NOTHING",
        [(mid[r.match_id], mv, r.p_ensemble_raw_H, r.p_ensemble_raw_D, r.p_ensemble_raw_A, Json(dict(elo=[r.p_elo_H, r.p_elo_D, r.p_elo_A], dc=[r.p_dc_H, r.p_dc_D, r.p_dc_A], gb=[r.p_gb_H, r.p_gb_D, r.p_gb_A])), r.disagreement) for r in btp.itertuples()], page_size=500)
    pid = {}; n_pred = 0
    allsn = [s for m, v in snaps["hist"].items() for s in v if s["kind"] == "reconstructed"] + snaps["live"]
    for s in sorted(allsn, key=lambda x: (x["match_id"], x["as_of"])):
        key = (s["match_id"], s["as_of"], s["kind"])
        if not s["published"]: continue
        at = pd.Timestamp(s["as_of"], tz="UTC"); ko = pd.Timestamp(s["kickoff_utc"]); days = max(0, int((ko - at).days))
        fs = q1("INSERT INTO feature_snapshots(match_id,snapshot_at,data_cutoff,feature_version,features,data_quality_score) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id", (mid[s["match_id"]], at, s["data_cutoff"], s["feature_version"], Json(s.get("features", {})), s["data_quality"]))
        p = s["probs"]; pid[key] = q1("INSERT INTO prediction_snapshots(match_id,kind,prediction_at,prediction_horizon,model_version_id,feature_snapshot_id,data_cutoff,home_probability,draw_probability,away_probability,expected_home_goals,expected_away_goals,confidence_level,confidence_score,model_disagreement,data_quality_score,simulation_count,components,intervals,scorelines,explanation) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (match_id,model_version_id,prediction_at,kind) DO UPDATE SET kind=prediction_snapshots.kind RETURNING id",
            (mid[s["match_id"]], s["kind"], at, f"T-{days}d", mv, fs, s["data_cutoff"], p["home"], p["draw"], p["away"], s["lam_h"], s["lam_a"], s["confidence"]["level"], s["confidence"]["score"], s["disagreement"], s["data_quality"], s["simulation"]["simulations"], Json(s["components"]), Json(s["simulation"]["interval_90"]), Json(s["simulation"]["top_scorelines"]), Json(dict(dq_parts=s["dq_parts"], weights=s["weights"])))); n_pred += 1
        if "lineage" in s and (s["match_id"], s["previous_as_of"], "reconstructed") in pid:
            for it in s["lineage"]: cur.execute("INSERT INTO prediction_lineage(prediction_id,previous_prediction_id,event_type,impact_score,explanation) VALUES (%s,%s,%s,%s,%s)", (pid[key], pid[(s["match_id"], s["previous_as_of"], "reconstructed")], it["event_type"], it["impact_score"], it["explanation"]))
    cur.execute("INSERT INTO audit_logs(actor_type,actor_id,action,entity_type,metadata) VALUES ('job','load_db','predictions_loaded','prediction_snapshots',%s)", (Json(dict(n=n_pred, model=MODEL_VERSION)),))
    con.commit(); print("loaded", len(rows), "matches,", n_pred, "prediction snapshots,", len(btp), "backtest rows")
if __name__ == "__main__": main()
