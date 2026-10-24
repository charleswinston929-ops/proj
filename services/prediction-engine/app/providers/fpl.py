"""Fantasy Premier League public API adapter (player availability). Endpoint: https://fantasy.premierleague.com/api/bootstrap-static/
No API key. Schema as documented by the community (elements[], teams[]). NOT reachable from the build sandbox: parsing is unit-tested on a
schema-faithful fixture; the live fetch must be verified on first run (python -m app.pipeline.availability_fetch)."""
from __future__ import annotations
import json, urllib.request
from dataclasses import dataclass
from typing import Optional
from ..entities import EntityResolver, UnresolvedEntity

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

@dataclass(frozen=True)
class PlayerRow:
    provider_id: int; name: str; web_name: str; team_key: str; position: str; status: str; chance: Optional[int]; news: str; news_added: Optional[str]
    minutes: int; goals: int; assists: int; xg: float; xa: float

def fetch(timeout: int = 25) -> dict:
    req = urllib.request.Request(URL, headers={"User-Agent": "sports-intelligence/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())

def parse_bootstrap(payload: dict, resolver: EntityResolver) -> tuple[list[PlayerRow], list[str]]:
    teams = {t["id"]: t["name"] for t in payload["teams"]}; rows, unresolved = [], set()
    for e in payload["elements"]:
        try: key = resolver.resolve(teams[e["team"]])
        except (UnresolvedEntity, KeyError): unresolved.add(teams.get(e.get("team"), str(e.get("team")))); continue
        f = lambda k: float(e.get(k) or 0)
        rows.append(PlayerRow(int(e["id"]), f'{e.get("first_name", "")} {e.get("second_name", "")}'.strip(), e.get("web_name", ""), key, POS.get(e.get("element_type"), "MID"),
                              e.get("status", "a"), e.get("chance_of_playing_next_round"), (e.get("news") or "")[:240], e.get("news_added"),
                              int(e.get("minutes") or 0), int(e.get("goals_scored") or 0), int(e.get("assists") or 0), f("expected_goals"), f("expected_assists")))
    return rows, sorted(unresolved)
