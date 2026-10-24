"""Whitelisted analyst tools (spec 46, 69). Every tool returns structured data; the analyst can call NOTHING else and never runs SQL."""
from __future__ import annotations
from pydantic import BaseModel, Field
from ..core.scenario import run_scenario

class FixtureArg(BaseModel): match_id: str = Field(max_length=200)
class ScenarioArg(BaseModel):
    match_id: str = Field(max_length=200)
    home_attack_pct: float = Field(0, ge=-50, le=50); away_attack_pct: float = Field(0, ge=-50, le=50)
    home_concede_pct: float = Field(0, ge=-50, le=50); away_concede_pct: float = Field(0, ge=-50, le=50); neutral_venue: bool = False

def _need(store, mid):
    s = store.fixture(mid)
    if not s: raise KeyError("unknown or unavailable fixture")
    return s
def get_fixture(store, a: FixtureArg):
    s = _need(store, a.match_id); return dict(match_id=s["match_id"], home=store.names[s["home"]], away=store.names[s["away"]], kickoff_utc=s["kickoff_utc"], matchday=s["matchday"])
def get_prediction(store, a: FixtureArg):
    s = _need(store, a.match_id)
    if not s["published"]: return dict(available=False, reason="data quality below publication threshold", data_quality=s["data_quality"])
    return dict(available=True, probs=s["probs"], expected_goals=dict(home=s["lam_h"], away=s["lam_a"]), confidence=s["confidence"], data_quality=s["data_quality"], model_version=store.d["model"]["version"], data_cutoff=s["data_cutoff"])
def compare_models(store, a: FixtureArg):
    s = _need(store, a.match_id); c = s["components"]
    return dict(elo=dict(zip(("home", "draw", "away"), c["elo"])), goal_model=dict(zip(("home", "draw", "away"), c["dc"])), gradient_boosting=dict(zip(("home", "draw", "away"), c["gb"])), ensemble=s["probs"], disagreement=s["disagreement"],
                note="Model disagreement did not predict forecast error in the backtest (Spearman ~0.02); treat as informational.")
def get_data_quality(store, a: FixtureArg):
    s = _need(store, a.match_id); return dict(score=s["data_quality"], band=s["dq_band"], parts=s["dq_parts"], missing_results=s["missing_results"])
def get_prediction_history(store, a: FixtureArg):
    _need(store, a.match_id); h = store.hist.get(a.match_id, [])
    return [dict(as_of=x["as_of"], kind=x["kind"], probs=x["probs"], data_quality=x["data_quality"], lineage=x.get("lineage", [])) for x in h]
def scenario(store, a: ScenarioArg):
    s = _need(store, a.match_id)
    base = dict(lam_h=s["lam_h"], lam_a=s["lam_a"], rho=s["rho"], home_adv_log=s["home_adv_log"], components=s["components"], weights=s["weights"], calibrator=s["calibrator"])
    return run_scenario(base, a.model_dump(exclude={"match_id"}))

TOOLS = {"get_fixture": (FixtureArg, get_fixture), "get_prediction": (FixtureArg, get_prediction), "compare_models": (FixtureArg, compare_models),
         "get_data_quality": (FixtureArg, get_data_quality), "get_prediction_history": (FixtureArg, get_prediction_history), "run_scenario": (ScenarioArg, scenario)}

def call_tool(store, name: str, args: dict):
    if name not in TOOLS: raise PermissionError(f"tool '{name}' is not allowlisted")
    model, fn = TOOLS[name]; return fn(store, model(**args))
