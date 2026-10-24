"""Fetch live availability and write artifacts/availability.json (and optionally the database). Usage: python -m app.pipeline.availability_fetch [--db]"""
import json, os, sys, datetime as dt
from pathlib import Path
from ..data import build_dataset
from ..providers.fpl import fetch, parse_bootstrap
from .availability import team_summaries
ART = Path("../../artifacts")

def to_db(rows):
    import psycopg2; from psycopg2.extras import Json
    con = psycopg2.connect(os.environ["DATABASE_URL"]); cur = con.cursor(); cur.execute("SELECT id FROM sports WHERE code='football'"); sport = cur.fetchone()[0]
    cur.execute("SELECT canonical_key,id FROM teams"); tid = dict(cur.fetchall()); cur.execute("SELECT id FROM data_sources WHERE name='fpl'")
    r = cur.fetchone()
    if r: src = r[0]
    else: cur.execute("INSERT INTO data_sources(name,provider_type,tier,reliability_score,api_base_url) VALUES ('fpl','public_api',3,0.75,'https://fantasy.premierleague.com/api') RETURNING id"); src = cur.fetchone()[0]
    now = dt.datetime.now(dt.timezone.utc)
    for p in rows:
        cur.execute("""INSERT INTO players(sport_id,name,position,current_team_id,metadata) VALUES (%s,%s,%s,%s,%s)
                       ON CONFLICT ((metadata->>'fpl_id')) WHERE metadata ? 'fpl_id' DO UPDATE SET current_team_id=EXCLUDED.current_team_id, position=EXCLUDED.position RETURNING id""", (sport, p.name, p.position, tid[p.team_key], Json({"fpl_id": str(p.provider_id), "web_name": p.web_name})))
        pid = cur.fetchone()[0]; st = {"a": "available", "d": "doubtful", "i": "injured", "s": "suspended", "u": "unknown", "n": "unknown"}.get(p.status, "unknown")
        cur.execute("INSERT INTO player_availability(player_id,team_id,status,reason,confidence,source_id,observed_at) VALUES (%s,%s,%s,%s,%s,%s,%s)", (pid, tid[p.team_key], st, p.news or None, None if p.chance is None else p.chance / 100, src, now))
    cur.execute("INSERT INTO audit_logs(actor_type,actor_id,action,entity_type,metadata) VALUES ('job','availability_fetch','availability_loaded','player_availability',%s)", (Json({"players": len(rows)}),))
    con.commit(); con.close()

def main():
    res = build_dataset(data_dir="../../data/raw")[1]; rows, unresolved = parse_bootstrap(fetch(), res)
    out = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), source="fantasy.premierleague.com (FPL public API)", players=len(rows), unresolved_teams=unresolved, summaries=team_summaries(rows))
    (ART / "availability.json").write_text(json.dumps(out)); print("availability:", len(rows), "players;", sum(v["n_flagged"] for v in out["summaries"].values()), "flagged; unresolved teams:", unresolved or "none")
    if "--db" in sys.argv: to_db(rows); print("loaded to database")
if __name__ == "__main__": main()
