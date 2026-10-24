"""Elo ratings with goal-difference scaling + ordered-logit mapping to P(H),P(D),P(A).
(The v0.1 code treated Elo's expected SCORE as a win probability and halved the draw: mathematically wrong.)"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

class Elo:
    def __init__(self, k=20.0, home_adv=65.0, regress=0.25, newcomer=1430.0, init=1500.0):
        self.k, self.h, self.regress, self.newcomer, self.init = k, home_adv, regress, newcomer, init
        self.r: dict[str, float] = {}; self.season: str | None = None
    def rating(self, t): return self.r.get(t, self.newcomer)
    def new_season(self, season, teams):
        if self.season is not None:
            for t in list(self.r): self.r[t] += self.regress * (self.init - self.r[t])
        for t in teams: self.r.setdefault(t, self.newcomer)
        self.season = season
    def expected(self, h, a): return 1.0 / (1.0 + 10 ** (-(self.rating(h) - self.rating(a) + self.h) / 400.0))
    def update(self, h, a, hg, ag):
        e = self.expected(h, a); s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag); g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        d = self.k * g * (s - e); self.r[h] = self.rating(h) + d; self.r[a] = self.rating(a) - d
        return s - e

def _sig(z): return 1.0 / (1.0 + np.exp(-z))

class OrderedLogit:
    """P(H)=σ(a·x+b−c), P(A)=σ(−a·x−b−c), P(D)=remainder; x = Elo diff incl. home adv, in 400-pt units."""
    def __init__(self): self.theta = np.array([1.0, 0.0, 0.6])
    def proba(self, x):
        a, b, c = self.theta; x = np.asarray(x, float)
        ph, pa = _sig(a * x + b - c), _sig(-a * x - b - c)
        p = np.stack([ph, 1 - ph - pa, pa], 1); p = np.clip(p, 1e-6, 1); return p / p.sum(1, keepdims=True)
    def fit(self, x, y):   # y in {0:H,1:D,2:A}
        x = np.asarray(x, float); y = np.asarray(y, int)
        f = lambda th: (setattr(self, "theta", th), -np.log(self.proba(x)[np.arange(len(y)), y]).mean())[1]
        r = minimize(f, self.theta, method="L-BFGS-B", bounds=[(0.05, 5), (-1, 1), (0.05, 3)]); self.theta = r.x; return self
