"""openfootball/england text-format provider (public domain data, github.com/openfootball).
Handles both file layouts used across seasons:
  A) 'Home v Away 3-0 (2-0)'      B) 'Home  3-0 (2-0)  Away'
Goal-scorer lines are ignored. Source provides goals only (no xG / shots / lineups)."""
from __future__ import annotations
import re, urllib.request
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .base import RawMatch, SportsProvider

BASE = "https://raw.githubusercontent.com/openfootball/england/master/{season}/1-premierleague.txt"
MONTHS = {m: i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
DATE_RE = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?(?:\s+\[.*\])?\s*$")
TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s+(.*)$")
SCORE = r"(?P<hg>\d+)-(?P<ag>\d+)(?:\s*\((?P<hh>\d+)-(?P<ha>\d+)\))?"
FMT_A = re.compile(rf"^(?P<h>.+?)\s+v\s+(?P<a>.+?)(?:\s+{SCORE})?\s*$")
FMT_B = re.compile(rf"^(?P<h>.+?)\s{{2,}}{SCORE}\s{{2,}}(?P<a>.+?)\s*$")
MD_RE = re.compile(r"Matchday\s+(\d+)")
LONDON = ZoneInfo("Europe/London")

def parse(text: str, season: str, source: str = "openfootball") -> tuple[list[RawMatch], list[str]]:
    start_year = int(season[:4]); out: list[RawMatch] = []; skipped: list[str] = []
    cur_date: date | None = None; cur_time = None; md = None
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if not s or s.startswith(("=", "#")) or s.startswith("("): continue
        m = MD_RE.search(s)
        if s.startswith("▪") or m:
            md = int(m.group(1)) if m else md; continue
        d = DATE_RE.match(line)
        if d:
            mon = MONTHS[d.group(2)]; yr = int(d.group(4)) if d.group(4) else (start_year if mon >= 8 else start_year + 1)
            cur_date = date(yr, mon, int(d.group(3))); cur_time = None
            if ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][cur_date.weekday()] != d.group(1): skipped.append(f"WEEKDAY_MISMATCH {s}")
            continue
        if cur_date is None: skipped.append(s); continue
        body = line
        t = TIME_RE.match(line)
        if t: cur_time = (int(t.group(1)), int(t.group(2))); body = t.group(3)
        body = body.strip()
        mm = FMT_A.match(body) if " v " in body else None
        if mm is None: mm = FMT_B.match(body)
        if mm is None: skipped.append(s); continue
        g = mm.groupdict(); hh, mi = cur_time or (15, 0)
        ko = datetime(cur_date.year, cur_date.month, cur_date.day, hh, mi, tzinfo=LONDON).astimezone(ZoneInfo("UTC"))
        toi = lambda k: int(g[k]) if g.get(k) is not None else None
        out.append(RawMatch(source, season, cur_date, ko, re.sub(r"\s+", " ", g["h"]).strip(), re.sub(r"\s+", " ", g["a"]).strip(),
                            toi("hg"), toi("ag"), toi("hh"), toi("ha"), md))
    return out, skipped

class OpenFootballProvider(SportsProvider):
    name = "openfootball"
    def __init__(self, cache_dir: str | Path = "data/raw", offline: bool = False):
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True); self.offline = offline
    def seasons(self) -> list[str]:
        return [f"{y}-{(y + 1) % 100:02d}" for y in range(2010, 2027)]
    def _text(self, season: str, refresh: bool = False) -> str:
        p = self.cache / f"pl_{season}.txt"
        if refresh or not p.exists():
            if self.offline: raise FileNotFoundError(p)
            with urllib.request.urlopen(BASE.format(season=season), timeout=30) as r: p.write_bytes(r.read())
        return p.read_text(encoding="utf-8")
    def get_matches(self, season: str, refresh: bool = False) -> list[RawMatch]:
        return parse(self._text(season, refresh), season, self.name)[0]
