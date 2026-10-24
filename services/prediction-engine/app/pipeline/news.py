"""News -> structured events (spec sections 41-42, 69). External text is DATA, never instructions.
An article cannot move a forecast: only a validated, entity-resolved, sufficiently verified event may produce an impact."""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass, field
from typing import Callable, Optional
from ..entities import EntityResolver, UnresolvedEntity

ALLOWED_EVENTS = {"injury", "suspension", "return", "available", "manager_change", "training_absence", "lineup_leak", "other"}
SOURCE_TIER = {"official": 1, "data_provider": 2, "journalism": 3, "specialist": 4, "social": 5}
INJECTION = re.compile(r"(ignore (all |any )?(previous|prior|above)|system prompt|you are now|disregard .{0,20}instructions|predict .{0,30}\b100\s*%|developer mode|override)", re.I)
NEGATIVE = {"injury", "suspension", "training_absence"}

@dataclass
class NewsEvent:
    title: str; source_kind: str; event_type: str; team_key: Optional[str]; confidence: float
    content_hash: str; injection_flagged: bool = False; verification: str = "unverified"; subject: Optional[str] = None
    @property
    def tier(self): return SOURCE_TIER.get(self.source_kind, 5)

def content_hash(text: str) -> str: return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()

def rule_extractor(text: str) -> dict:
    """Deterministic fallback extractor. An LLM extractor can be swapped in: its output is validated identically."""
    t = text.lower(); et = "other"
    for pat, e in ((r"ruled out|injur|sidelined|out for", "injury"), (r"suspend|banned", "suspension"), (r"misses training|absent from training", "training_absence"),
                   (r"returns? to|back in training|fit again|available", "return"), (r"sacked|appointed|new manager", "manager_change")):
        if re.search(pat, t): et = e; break
    return dict(event_type=et, confidence=0.6)

def ingest(text: str, source_kind: str, resolver: EntityResolver, team_hint: Optional[str] = None, extractor: Callable[[str], dict] = rule_extractor) -> NewsEvent:
    flagged = bool(INJECTION.search(text))
    cand = extractor("" if flagged else text)                       # flagged content is never passed to an extractor
    et = cand.get("event_type", "other"); et = et if et in ALLOWED_EVENTS else "other"
    conf = max(0.0, min(1.0, float(cand.get("confidence", 0.0)))) * (0.0 if flagged else 1.0)
    team = None
    if team_hint:
        try: team = resolver.resolve(team_hint)
        except UnresolvedEntity: team = None; conf *= 0.5              # unresolved entity -> cannot be trusted
    ev = NewsEvent(text[:140], source_kind, et, team, conf, content_hash(text), flagged)
    ev.verification = "official" if ev.tier == 1 and conf > 0 else "unverified"
    return ev

def corroborate(events: list[NewsEvent]) -> list[NewsEvent]:
    """>=2 independent non-social sources with the same (team,event) promote 'unverified' to 'corroborated'."""
    groups: dict = {}
    for e in events: groups.setdefault((e.team_key, e.event_type), set()).add(e.source_kind)
    for e in events:
        if e.verification == "unverified" and len({s for s in groups[(e.team_key, e.event_type)] if s != "social"}) >= 2: e.verification = "corroborated"
    return events

def detect_conflicts(events: list[NewsEvent]) -> list[dict]:
    """Do not silently overwrite: a negative report vs a positive one for the same team yields a stored conflict, official source preferred."""
    out = []; by_team: dict = {}
    for e in events:
        if e.team_key and not e.injection_flagged: by_team.setdefault(e.team_key, []).append(e)
    for team, evs in by_team.items():
        neg = [e for e in evs if e.event_type in NEGATIVE]; pos = [e for e in evs if e.event_type in {"return", "available"}]
        if neg and pos:
            best = min(neg + pos, key=lambda e: (e.tier, -e.confidence))
            out.append(dict(team=team, status="CONFLICT", sources=[(e.source_kind, e.event_type, round(e.confidence, 2)) for e in neg + pos], preferred=(best.source_kind, best.event_type)))
    return out

def may_affect_forecast(e: NewsEvent) -> bool:
    """Rumours never move a forecast automatically (spec 41)."""
    return (not e.injection_flagged) and e.team_key is not None and e.verification in {"official", "corroborated"} and e.confidence >= 0.8
