import os, pytest
psycopg2 = pytest.importorskip("psycopg2")
URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
@pytest.fixture()
def cur():
    con = psycopg2.connect(URL); c = con.cursor(); yield c; con.rollback(); con.close()
def test_predictions_are_immutable(cur):
    for sql in ("UPDATE prediction_snapshots SET home_probability = 0.99", "DELETE FROM prediction_snapshots", "DELETE FROM prediction_lineage", "UPDATE feature_snapshots SET features='{}'", "DELETE FROM audit_logs"):
        with pytest.raises(psycopg2.errors.RaiseException): cur.execute(sql)
        cur.connection.rollback()
def test_probabilities_must_sum_to_one(cur):
    cur.execute("SELECT id FROM matches LIMIT 1"); m = cur.fetchone()[0]
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("INSERT INTO prediction_snapshots(match_id,kind,prediction_at,prediction_horizon,model_version_id,data_cutoff,home_probability,draw_probability,away_probability,confidence_level,data_quality_score,components) SELECT %s,'live',now(),'T-1d',id,now() - interval '1 day',0.6,0.6,0.6,'low',0.5,'{}' FROM model_versions LIMIT 1", (m,))
def test_cutoff_cannot_exceed_prediction_time(cur):
    cur.execute("SELECT id FROM matches LIMIT 1"); m = cur.fetchone()[0]
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("INSERT INTO prediction_snapshots(match_id,kind,prediction_at,prediction_horizon,model_version_id,data_cutoff,home_probability,draw_probability,away_probability,confidence_level,data_quality_score,components) SELECT %s,'live',now(),'T-1d',id,now() + interval '1 day',0.5,0.3,0.2,'low',0.5,'{}' FROM model_versions LIMIT 1", (m,))
def test_match_integrity_constraints(cur):
    cur.execute("SELECT count(*) FROM matches WHERE home_team_id = away_team_id"); assert cur.fetchone()[0] == 0
    cur.execute("SELECT count(*) FROM prediction_snapshots WHERE abs(home_probability+draw_probability+away_probability-1) > 0.00001"); assert cur.fetchone()[0] == 0
    cur.execute("SELECT count(*) FROM matches WHERE status='finished' AND home_score IS NULL"); assert cur.fetchone()[0] == 0
def test_exactly_one_active_model(cur):
    cur.execute("SELECT count(*) FROM model_versions WHERE status='active'"); assert cur.fetchone()[0] == 1
