"""Canonical dataset builder + data validation (spec: normalization layer, structural validity)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .entities import EntityResolver
from .providers.base import SportsProvider
from .providers.openfootball import OpenFootballProvider, parse

def build_dataset(provider: SportsProvider | None = None, data_dir: str | Path = "data/raw", offline: bool = True):
    provider = provider or OpenFootballProvider(data_dir, offline=offline)
    res = EntityResolver(); rows = []; report = {"seasons": {}, "anomalies": []}
    for season in provider.seasons():
        ms = provider.get_matches(season)
        _, skipped = parse(provider._text(season), season) if hasattr(provider, "_text") else (None, [])
        bad = [x for x in skipped if x.startswith("WEEKDAY_MISMATCH") or "'" not in x]   # scorer lines contain minute marks
        teams = set()
        for m in ms:
            h, a = res.resolve(m.home_raw), res.resolve(m.away_raw); teams |= {h, a}
            rows.append(dict(season=season, matchday=m.matchday, date=pd.Timestamp(m.match_date), kickoff_utc=m.kickoff_utc, home=h, away=a,
                             hg=m.home_goals, ag=m.away_goals, played=m.played, source=m.source))
        report["seasons"][season] = dict(matches=len(ms), teams=len(teams), played=sum(m.played for m in ms), unparsed=len(bad))
        report["anomalies"] += [f"{season}: {x}" for x in bad]
        if len(teams) != 20: report["anomalies"].append(f"{season}: {len(teams)} teams (expected 20)")
        dup = pd.Series([(m.home_raw, m.away_raw) for m in ms]).duplicated().sum()
        if dup: report["anomalies"].append(f"{season}: {dup} duplicate fixtures")
    df = pd.DataFrame(rows).sort_values(["kickoff_utc", "home"]).reset_index(drop=True)
    df["match_id"] = df["season"] + "|" + df["home"] + "|" + df["away"]
    bad_scores = df[df.played & ((df.hg < 0) | (df.ag < 0) | (df.hg > 15) | (df.ag > 15))]
    if len(bad_scores): report["anomalies"].append(f"{len(bad_scores)} implausible scores")
    res.freeze()
    return df, res, report
