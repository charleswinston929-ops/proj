"""Entity resolution (spec section 76). Teams are never matched by raw string equality:
names are normalised, passed through a curated alias table, and unknown names are REJECTED
(with suggestions) once the canonical set is frozen, instead of silently creating duplicates."""
from __future__ import annotations
import re, unicodedata, difflib
from dataclasses import dataclass, field

DROP = {"fc", "afc", "cf"}
MANUAL_ALIASES = {
    "wolves": "wolverhampton wanderers", "man city": "manchester city", "man united": "manchester united",
    "man utd": "manchester united", "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
    "newcastle": "newcastle united", "west brom": "west bromwich albion", "qpr": "queens park rangers",
    "brighton": "brighton and hove albion", "nott m forest": "nottingham forest", "nottm forest": "nottingham forest", "leeds": "leeds united", "nottm forest": "nottingham forest",
    "sheffield utd": "sheffield united", "leicester": "leicester city", "norwich": "norwich city",
    "cardiff": "cardiff city", "swansea": "swansea city", "stoke": "stoke city", "hull": "hull city",
    "huddersfield": "huddersfield town", "ipswich": "ipswich town", "luton": "luton town",
    "coventry": "coventry city", "west ham": "west ham united", "birmingham": "birmingham city",
    "blackburn": "blackburn rovers", "bolton": "bolton wanderers", "wigan": "wigan athletic",
}

def normalize(raw: str) -> str:
    s = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    toks = [t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t not in DROP]
    return " ".join(toks)

def display_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw).strip()
    s = re.sub(r"^AFC\s+", "", s); s = re.sub(r"\s+(A?FC)$", "", s)
    return s

class UnresolvedEntity(Exception):
    def __init__(self, raw, suggestions): super().__init__(f"Unresolved team '{raw}'. Suggestions: {suggestions}"); self.raw, self.suggestions = raw, suggestions

@dataclass
class EntityResolver:
    canonical: dict[str, str] = field(default_factory=dict)   # key -> display name
    aliases: dict[str, str] = field(default_factory=dict)     # raw alias -> key (audit trail)
    frozen: bool = False
    def key_for(self, raw: str) -> str:
        n = normalize(raw); return MANUAL_ALIASES.get(n, n)
    def resolve(self, raw: str) -> str:
        k = self.key_for(raw)
        if k not in self.canonical:
            if self.frozen: raise UnresolvedEntity(raw, difflib.get_close_matches(k, list(self.canonical), n=3, cutoff=0.6))
            self.canonical[k] = display_name(raw)
        self.aliases[raw] = k
        return k
    def freeze(self): self.frozen = True; return self
