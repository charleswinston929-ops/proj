"""Lineup ingestion contract (no free structured lineup feed exists; supply JSON from any provider). Validation, status transitions, and
conversion of 'regular starter missing from the matchday squad' into suggested adjustments. Confirmed lineups require a tier<=2 source."""
from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass, field
from ..providers.fpl import PlayerRow
from .availability import team_summaries

ORDER = {"projected": 0, "probable": 1, "confirmed": 2}
norm = lambda s: re.sub(r"[^a-z ]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()).strip()

@dataclass
class Lineup:
    match_id: str; team_key: str; status: str; starters: list[str]; bench: list[str] = field(default_factory=list); formation: str | None = None
    observed_at: str = ""; source: str = "manual"; source_tier: int = 5

def _known(rows): return {norm(n) for r in rows for n in (r.name, r.web_name) if n} | {norm(r.name).split()[-1] for r in rows if r.name}
def validate(l: Lineup, roster: list[PlayerRow]) -> list[str]:
    err = []; s = [norm(x) for x in l.starters]
    if l.status not in ORDER: err.append(f"unknown status {l.status}")
    if len(l.starters) != 11 or len(set(s)) != 11: err.append("a lineup needs exactly 11 distinct starters")
    if l.status == "confirmed" and l.source_tier > 2: err.append("confirmed lineups require an official or data-provider source (tier <= 2)")
    k = _known([r for r in roster if r.team_key == l.team_key]); unk = [x for x in l.starters if norm(x) not in k and norm(x).split()[-1:] and norm(x).split()[-1] not in k]
    if roster and unk: err.append(f"players not in {l.team_key} squad: {unk}")
    return err
def accept(existing: Lineup | None, new: Lineup) -> tuple[bool, str]:
    """A confirmed lineup is never replaced by a lower status; equal/higher status supersedes."""
    if existing is None: return True, "first lineup"
    if ORDER[new.status] < ORDER[existing.status]: return False, f"rejected: {new.status} cannot replace {existing.status}"
    return True, f"{existing.status} superseded by {new.status}"
def _in(r: PlayerRow, S: set) -> bool:
    return norm(r.name) in S or norm(r.web_name) in S or (bool(r.name) and norm(r.name).split()[-1] in {x.split()[-1] for x in S if x})
def missing_regulars(l: Lineup, roster: list[PlayerRow], top: int = 11) -> list[PlayerRow]:
    """Regular starters (top by minutes) absent from the named XI."""
    ps = sorted([r for r in roster if r.team_key == l.team_key], key=lambda r: -r.minutes)[:top]; xi = {norm(x) for x in l.starters}
    return [r for r in ps if not _in(r, xi)]
def suggest(l: Lineup, roster: list[PlayerRow]) -> dict:
    """Named starters are treated as available (the lineup overrides a stale injury flag); regulars missing from the XI are out if not in the
    squad at all, or 35% likely to miss if only on the bench; everyone else keeps the feed's status."""
    xi = {norm(x) for x in l.starters}; bench = {norm(x) for x in l.bench}; absent = {r.provider_id for r in missing_regulars(l, roster)}; adj = []
    for r in [r for r in roster if r.team_key == l.team_key]:
        if _in(r, xi): st, ch = "a", 100
        elif r.provider_id in absent: st, ch = ("d", 65) if _in(r, bench) else ("u", 0)
        else: st, ch = r.status, r.chance
        adj.append(PlayerRow(r.provider_id, r.name, r.web_name, r.team_key, r.position, st, ch, r.news, r.news_added, r.minutes, r.goals, r.assists, r.xg, r.xa))
    return team_summaries(adj).get(l.team_key, dict(attack_pct=0.0, concede_pct=0.0, out=[], n_flagged=0))
