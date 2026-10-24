"""Player availability -> SUGGESTED scenario adjustments. UNVALIDATED: there is no historical availability data to estimate these effects, so the
coefficients below are explicit assumptions and the result is never applied to published forecasts (AVAILABILITY_IN_FORECAST is off)."""
from __future__ import annotations
from collections import defaultdict
from ..providers.fpl import PlayerRow

K_ATT, K_DEF, CLAMP = 0.6, 0.4, 25.0     # 0.6: a replacement recovers ~40% of a lost player's goal involvement; 0.4: defensive weight. Assumptions.
AVAILABILITY_IN_FORECAST = False

def p_out(status: str, chance) -> float:
    c = None if chance is None else max(0.0, min(100.0, float(chance)))
    if status == "a": return 0.0 if c is None else 1 - c / 100
    if status == "d": return 1 - (50.0 if c is None else c) / 100
    return 1.0 if c is None else 1 - c / 100          # injured / suspended / unavailable / not in squad

def team_summaries(rows: list[PlayerRow]) -> dict:
    by = defaultdict(list)
    for r in rows: by[r.team_key].append(r)
    out = {}
    for team, ps in by.items():
        att_total = sum(p.xg + p.xa for p in ps) or 1e-9; def_min = sum(p.minutes for p in ps if p.position in ("GK", "DEF")) or 1
        att = conc = 0.0; flagged = []
        for p in ps:
            po = p_out(p.status, p.chance)
            if po < 0.25: continue
            a_share = (p.xg + p.xa) / att_total; d_share = p.minutes / def_min if p.position in ("GK", "DEF") else 0.0
            if a_share < 0.01 and d_share < 0.03: continue                       # ignore fringe players
            att += po * a_share; conc += po * d_share
            flagged.append(dict(name=p.web_name or p.name, pos=p.position, status=p.status, chance=p.chance, news=p.news, p_out=round(po, 2), importance=round(a_share + d_share, 3)))
        flagged.sort(key=lambda x: -x["importance"] * x["p_out"])
        out[team] = dict(attack_pct=round(-max(0.0, min(CLAMP, 100 * K_ATT * att)), 1), concede_pct=round(max(0.0, min(CLAMP, 100 * K_DEF * conc)), 1), out=flagged[:8], n_flagged=len(flagged),
                         assumptions=dict(k_att=K_ATT, k_def=K_DEF, validated=False))
    return out
