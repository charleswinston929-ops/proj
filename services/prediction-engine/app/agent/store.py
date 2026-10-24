from __future__ import annotations
import json, os
from pathlib import Path
class JsonStore:
    """Read-only view over the exported artifacts (the DB is the system of record; the pipeline exports from it)."""
    def __init__(self, path: str | None = None):
        p = Path(path or os.getenv("ARTIFACT_PATH", "../../artifacts/app_data.json")); self.d = json.loads(p.read_text())
        self.names = self.d["names"]; self.up = {s["match_id"]: s for s in self.d["upcoming"]}; self.hist = self.d["history"]
    def fixture(self, mid): return self.up.get(mid)
    def team_by_text(self, text: str):
        t = text.lower(); return [k for k, n in self.names.items() if n.lower() in t or k in t]
    def fixtures_for(self, teams): return sorted([s for s in self.up.values() if set(teams) & {s["home"], s["away"]}], key=lambda s: s["kickoff_utc"])
