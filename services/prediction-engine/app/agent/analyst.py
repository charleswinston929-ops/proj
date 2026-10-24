"""AI analyst. Deterministic orchestration + a provenance guard: every probability in an answer must come from a tool result.
An LLM may be plugged in later to PHRASE the answer, but its output must pass verify_numbers() before it is shown."""
from __future__ import annotations
import re
from .tools import call_tool
INTENTS = [("SCENARIO_QUERY", r"\bwhat if\b|\bif\b.*\b(attack|defen[cs]e|neutral)|scenario"), ("WHY_QUERY", r"\bwhy\b|what changed|\bchanged\b|moved"),
           ("COMPARISON_QUERY", r"compare|models? (agree|disagree)|disagreement"), ("DATA_QUALITY_QUERY", r"data quality|how (reliable|certain|confident)|confidence"),
           ("PREDICTION_QUERY", r"predict|forecast|chance|probab|who (will|wins)|win")]
def classify(q: str) -> str:
    for name, pat in INTENTS:
        if re.search(pat, q, re.I): return name
    return "GENERAL_SPORT_QUERY"
def _nums(obj, out):
    if isinstance(obj, dict): [_nums(v, out) for v in obj.values()]
    elif isinstance(obj, (list, tuple)): [_nums(v, out) for v in obj]
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool): out.append(float(obj))
    return out
def verify_numbers(text: str, tool_results: list) -> list[str]:
    """Return numbers in `text` that cannot be traced to any tool output (percentages checked at 0.1pp, plain decimals at 0.01)."""
    pool = _nums(tool_results, []); bad = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(%|pp)", text):
        v = float(m.group(1)); ok = any(abs(v - 100 * x) <= 0.051 or abs(abs(v) - abs(100 * x)) <= 0.051 for x in pool)
        if not ok: bad.append(m.group(0))
    return bad
pct = lambda x: f"{100*x:.1f}%"
def answer(store, question: str, resolver=None):
    q = question.strip()[:300]; intent = classify(q); teams = store.team_by_text(q); fx = store.fixtures_for(teams)[:1] if teams else []
    if intent == "GENERAL_SPORT_QUERY" or not fx:
        return dict(intent=intent, answer="Name a team with an upcoming fixture (e.g. 'Arsenal') and ask about its forecast, model disagreement, data quality, history or a scenario.", facts=[], model_output=None, evidence=[], uncertainty=[], scenarios=[], sources=[])
    s = fx[0]; mid = s["match_id"]; home, away = store.names[s["home"]], store.names[s["away"]]; res = []; facts = [f"Fixture: {home} vs {away}, {s['date']}"]; ev = []; unc = []; scen = []
    pred = call_tool(store, "get_prediction", dict(match_id=mid)); res.append(pred)
    dq = call_tool(store, "get_data_quality", dict(match_id=mid)); res.append(dq)
    if not pred["available"]:
        text = f"A forecast for {home} vs {away} is withheld: data quality {dq['score']:.2f} is below the publication threshold ({dq['missing_results']} results missing from the source)."
        unc.append("critical data quality threshold not met")
        return dict(intent=intent, answer=text, facts=facts, model_output=None, evidence=[], uncertainty=unc, scenarios=[], sources=["openfootball"])
    p = pred["probs"]; text = f"The model gives {home} {pct(p['home'])}, a draw {pct(p['draw'])} and {away} {pct(p['away'])} (expected goals {pred['expected_goals']['home']:.2f}-{pred['expected_goals']['away']:.2f})."
    if intent == "COMPARISON_QUERY":
        c = call_tool(store, "compare_models", dict(match_id=mid)); res.append(c)
        text += f" Elo says home {pct(c['elo']['home'])}, the goal model {pct(c['goal_model']['home'])}, gradient boosting {pct(c['gradient_boosting']['home'])}."
        ev.append(c["note"])
    if intent == "WHY_QUERY":
        h = call_tool(store, "get_prediction_history", dict(match_id=mid)); res.append(h)
        if len(h) >= 2:
            d = 100 * (h[-1]["probs"]["home"] - h[0]["probs"]["home"]); res.append(dict(delta=d / 100))
            text += f" Since {h[0]['as_of']} the home-win probability moved {d:+.1f}pp."
            ev += [it["explanation"] for it in h[-1]["lineage"][:4]]
    if intent == "SCENARIO_QUERY":
        m = re.search(r"(\d+)\s*%", q); k = float(m.group(1)) if m else 10.0; sign = 1 if re.search(r"increase|up|boost", q, re.I) else -1
        which = "away" if (teams and teams[0] == s["away"]) else "home"; key = f"{which}_{'concede' if re.search('defen', q, re.I) else 'attack'}_pct"
        r = call_tool(store, "run_scenario", dict(match_id=mid, **{key: sign * k})); res.append(r); scen.append(r)
        dp = r["delta_pp"]; res.append(dict(x=[v / 100 for v in dp.values()]))
        text += f" Scenario ({key.replace('_', ' ')} {sign*k:+.0f}): home {dp['home']:+.1f}pp, draw {dp['draw']:+.1f}pp, away {dp['away']:+.1f}pp. {r['note']}"
    text += f" Confidence is {pred['confidence']['level']}; data quality {dq['score']:.2f} ({dq['band']})."
    if dq["missing_results"]: unc.append(f"{dq['missing_results']} recent results for these teams are missing from the data source")
    unc.append("No lineup, injury or news data is connected yet; the forecast reflects results only.")
    bad = verify_numbers(text, res + [dict(c=pred["expected_goals"])])
    assert not bad, f"untraceable numbers in answer: {bad}"
    return dict(intent=intent, answer=text, facts=facts, model_output=p, evidence=ev, uncertainty=unc, scenarios=scen, sources=["openfootball", f"model {store.d['model']['name']} {store.d['model']['version']}"])
