"""Provider abstraction (spec section 78). The ML layer never sees provider-specific shapes."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

@dataclass(frozen=True)
class RawMatch:
    source: str
    season: str                 # "2025-26"
    match_date: date
    kickoff_utc: Optional[datetime]
    home_raw: str
    away_raw: str
    home_goals: Optional[int]   # None = not played yet
    away_goals: Optional[int]
    ht_home: Optional[int] = None
    ht_away: Optional[int] = None
    matchday: Optional[int] = None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

class SportsProvider(ABC):
    name: str
    @abstractmethod
    def seasons(self) -> list[str]: ...
    @abstractmethod
    def get_matches(self, season: str) -> list[RawMatch]: ...
