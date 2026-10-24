import pytest
from app.providers.openfootball import parse
from app.entities import EntityResolver, UnresolvedEntity, normalize

A = "= x\n▪ Matchday 1\n  Fri Aug 21 2026\n    20:00  Arsenal FC              v Coventry City FC         3-0 (2-0)\n  Sat Aug 22\n    12:30  Hull City AFC           v Manchester United FC\n"
B = "▪ Matchday 1\nSat Aug 18\n  15:00  Arsenal FC               0-0  Sunderland AFC\n         Fulham FC                5-0 (2-0)  Norwich City\n"
def test_format_a_scores_and_unplayed():
    ms, sk = parse(A, "2026-27"); assert len(ms) == 2 and not sk
    assert (ms[0].home_goals, ms[0].away_goals, ms[0].ht_home) == (3, 0, 2) and ms[0].matchday == 1
    assert not ms[1].played and ms[1].home_raw == "Hull City AFC"
def test_format_b_and_time_inheritance():
    ms, _ = parse(B, "2012-13"); assert [m.home_goals for m in ms] == [0, 5] and ms[0].kickoff_utc.hour == ms[1].kickoff_utc.hour
def test_year_inference_and_weekday_check():
    ms, _ = parse("Sat Jan 12\n  15:00 A FC  1-0  B FC\n", "2012-13"); assert ms[0].match_date.year == 2013
    _, sk = parse("Mon Aug 21 2026\n  15:00 A FC  1-0  B FC\n", "2026-27"); assert any("WEEKDAY_MISMATCH" in x for x in sk)
def test_covid_july_belongs_to_end_year():
    ms, _ = parse("Wed Jul 1\n  18:00 A FC  1-0  B FC\n", "2019-20"); assert ms[0].match_date.year == 2020
def test_entity_variants_resolve_identically():
    r = EntityResolver(); ks = {r.resolve(x) for x in ("AFC Bournemouth", "Bournemouth", "Bournemouth FC")}; assert len(ks) == 1
    assert r.resolve("Brighton & Hove Albion FC") == r.resolve("Brighton & Hove Albion") == r.resolve("Brighton")
def test_frozen_resolver_rejects_unknown_with_suggestion():
    r = EntityResolver(); r.resolve("Manchester United FC"); r.freeze()
    with pytest.raises(UnresolvedEntity) as e: r.resolve("Manchester Utd Women")
    assert isinstance(e.value.suggestions, list)
def test_full_dataset_is_clean(dataset):
    df, res, rep = dataset
    assert len(df) == 17 * 380 and rep["anomalies"] == [] and len(res.canonical) == 42
    assert df.groupby("season").apply(lambda g: len(set(g.home) | set(g.away))).eq(20).all()
    assert not df.match_id.duplicated().any()
