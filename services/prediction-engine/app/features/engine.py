"""Leak-free feature engine. One chronological state machine is used for training, backtesting AND serving,
so features can never differ between training and production. Every feature only uses matches played strictly
before the fixture's date (spec section 67). No xG/shot data exists in the source: features are goals/Elo based."""
from __future__ import annotations
import copy
from collections import defaultdict
import numpy as np, pandas as pd
from ..core.elo import Elo

FEATURE_VERSION = "features-1.0"
PRIOR = (1.55, 1.25); K_SHRINK = 3.0

class FormState:
    def __init__(self, elo_params=None):
        self.elo = Elo(**(elo_params or {})); self.hist = defaultdict(list); self.goals = 0.0; self.games = 0
        self.last_date = None
    @property
    def lg(self): return self.goals / (2 * self.games) if self.games else sum(PRIOR) / 2
    def prepare_for(self, season, teams):
        if self.elo.season != season: self.elo.new_season(season, teams)
    def _win(self, t, n, venue=None):
        h = [e for e in self.hist[t] if venue is None or e["v"] == venue][-n:]
        k, lg = K_SHRINK, self.lg
        gf = (sum(e["gf"] for e in h) + k * lg) / (len(h) + k); ga = (sum(e["ga"] for e in h) + k * lg) / (len(h) + k)
        ppg = (sum(e["pts"] for e in h) + k * 1.35) / (len(h) + k)
        return gf, ga, ppg
    def _team(self, t, date, venue, pre):
        f = {}
        for n in (3, 5, 10):
            gf, ga, ppg = self._win(t, n); f[f"gf{n}"], f[f"ga{n}"], f[f"ppg{n}"] = gf, ga, ppg
        f["v_gf5"], f["v_ga5"], _ = self._win(t, 5, venue)
        h = self.hist[t][-15:]
        if h:
            age = np.array([(date - e["d"]).days for e in h], float); w = 0.5 ** (age / 45.0)
            f["dgd"] = float((w * np.array([e["gf"] - e["ga"] for e in h])).sum() / (w.sum() + 1.0))
            f["res5"] = float(np.mean([e["res"] for e in h[-5:]])); f["rest"] = float(min((date - h[-1]["d"]).days, 21))
        else: f["dgd"], f["res5"], f["rest"] = 0.0, 0.0, 14.0
        f["n"] = float(min(len(self.hist[t]), 30))
        return {f"{pre}_{k}": v for k, v in f.items()}
    def featurize(self, home, away, date: pd.Timestamp):
        e = self.elo; f = {"elo_h": e.rating(home), "elo_a": e.rating(away)}
        f["elo_x"] = (f["elo_h"] - f["elo_a"] + e.h) / 400.0
        f.update(self._team(home, date, "H", "h")); f.update(self._team(away, date, "A", "a"))
        for k in ("gf5", "ga5", "ppg5", "ppg10", "dgd", "res5", "rest"): f[f"d_{k}"] = f[f"h_{k}"] - f[f"a_{k}"]
        f["lg"] = self.lg
        return f
    def update(self, home, away, hg, ag, date):
        res_h = self.elo.update(home, away, hg, ag)
        pts_h = 3 if hg > ag else 1 if hg == ag else 0
        self.hist[home].append(dict(d=date, gf=hg, ga=ag, pts=pts_h, v="H", res=res_h))
        self.hist[away].append(dict(d=date, gf=ag, ga=hg, pts=3 - pts_h if pts_h != 1 else 1, v="A", res=-res_h))
        self.goals += hg + ag; self.games += 1; self.last_date = date

def run_state(df: pd.DataFrame, elo_params=None, until: pd.Timestamp | None = None):
    """Chronological pass over `df` (all seasons incl. unplayed). Returns pre-match features for every PLAYED match
    strictly before `until`, plus the final state."""
    st = FormState(elo_params); teams = df.groupby("season").apply(lambda g: sorted(set(g.home) | set(g.away))).to_dict()
    rows = []; idx = []
    played = df[df.played & ((df.date < until) if until is not None else True)]
    for date, g in played.groupby("date", sort=True):
        for i, r in g.iterrows():
            st.prepare_for(r.season, teams[r.season]); rows.append(st.featurize(r.home, r.away, date)); idx.append(i)
        for i, r in g.iterrows(): st.update(r.home, r.away, int(r.hg), int(r.ag), date)
    return pd.DataFrame(rows, index=idx), st, teams

def forecast_features(state: FormState, teams: dict, fixtures: pd.DataFrame):
    """Features for unplayed fixtures from an existing state (state is copied, never mutated)."""
    st = copy.deepcopy(state); rows = []
    for _, r in fixtures.iterrows():
        st.prepare_for(r.season, teams[r.season]); rows.append(st.featurize(r.home, r.away, r.date))
    return pd.DataFrame(rows, index=fixtures.index)
