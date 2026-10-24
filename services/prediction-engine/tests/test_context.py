import json, pathlib, pytest
from app.entities import EntityResolver
from app.providers.fpl import parse_bootstrap
from app.pipeline import availability as AV, lineup as LU
from app.pipeline.availability import p_out, team_summaries
FIX = json.loads((pathlib.Path(__file__).parent / "fixtures" / "fpl_bootstrap_sample.json").read_text())   # synthetic, schema-faithful

@pytest.fixture(scope="module")
def parsed(dataset):
    return parse_bootstrap(FIX, dataset[1])
def test_fpl_parsing_resolves_teams_and_reports_unresolved(parsed):
    rows, unresolved = parsed; assert len(rows) == 10 and unresolved == ["Atlantis FC"]
    assert {r.team_key for r in rows} == {"arsenal", "manchester city", "nottingham forest"}     # "Man City", "Nott'm Forest" resolved
    saka = next(r for r in rows if r.web_name == "Saka"); assert saka.status == "i" and saka.xg == 3.5 and saka.position == "MID"
def test_probability_out_rules():
    assert p_out("a", 100) == 0 and p_out("a", None) == 0 and p_out("i", 0) == 1 and p_out("d", 50) == 0.5 and p_out("d", None) == 0.5 and p_out("s", None) == 1 and p_out("d", 75) == 0.25
def test_team_suggestions_direction_bounds_and_labelled_unvalidated(parsed):
    s = team_summaries(parsed[0]); a = s["arsenal"]
    assert a["attack_pct"] < 0 and a["concede_pct"] > 0 and abs(a["attack_pct"]) <= AV.CLAMP and a["assumptions"]["validated"] is False
    assert s["manchester city"]["attack_pct"] == 0 and s["manchester city"]["out"] == []
    assert a["out"][0]["name"] == "Saka" and AV.AVAILABILITY_IN_FORECAST is False        # never auto-applied to published forecasts
def test_lineup_validation_and_transitions(parsed):
    rows = parsed[0]; xi = ["Raya", "White", "Saliba", "Odegaard", "Jesus"] + ["Bukayo Saka"] * 0
    l = LU.Lineup("m", "arsenal", "confirmed", ["Raya", "White", "Saliba", "Odegaard", "Jesus", "Saka"] + [f"X{i}" for i in range(5)], source_tier=4)
    errs = LU.validate(l, rows); assert any("tier" in e for e in errs) and any("not in arsenal squad" in e for e in errs)
    assert LU.validate(LU.Lineup("m", "arsenal", "projected", ["a"] * 11), rows)[0].startswith("a lineup needs")
    c = LU.Lineup("m", "arsenal", "confirmed", [], source_tier=1); p = LU.Lineup("m", "arsenal", "projected", [])
    assert LU.accept(None, p)[0] and not LU.accept(c, p)[0] and LU.accept(p, c)[0]
def test_missing_regular_from_xi_creates_suggestion(parsed):
    rows = parsed[0]; l = LU.Lineup("m", "arsenal", "confirmed", ["Raya", "White", "Saliba", "Odegaard", "Jesus", "Saka"], bench=[], source_tier=1)
    miss = {r.web_name for r in LU.missing_regulars(l, rows)}; assert miss == set()                                 # all six regulars named
    l2 = LU.Lineup("m", "arsenal", "confirmed", ["Raya", "White", "Saliba", "Jesus", "Saka"], bench=["Odegaard"], source_tier=1)
    assert {r.web_name for r in LU.missing_regulars(l2, rows)} == {"Odegaard"}
    feed = team_summaries(rows)["arsenal"]["attack_pct"]
    assert LU.suggest(l, rows)["attack_pct"] > feed                                 # Saka named in the XI: lineup overrides his stale 'injured' flag
    assert LU.suggest(l2, rows)["attack_pct"] < LU.suggest(l, rows)["attack_pct"]   # Odegaard (a creator) demoted to the bench: attack suggestion worsens
