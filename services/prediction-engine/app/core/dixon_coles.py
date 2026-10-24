"""Time-decayed Poisson / Dixon-Coles goal model (fit by weighted MLE, analytic gradient).
Step 1: μ, home advantage, attack/defence with ridge shrinkage (this IS the plain Poisson model, rho=0).
Step 2: Dixon-Coles low-score dependence ρ by 1-D likelihood maximisation (standard two-step approximation)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

def tau(hg, ag, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1); m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
    t[m00] = 1 - lh[m00] * la[m00] * rho; t[m01] = 1 + lh[m01] * rho; t[m10] = 1 + la[m10] * rho; t[m11] = 1 - rho
    return np.clip(t, 1e-6, None)

class DixonColes:
    def __init__(self, half_life_days=540.0, ridge=2.0):
        self.hl, self.ridge = half_life_days, ridge
        self.mu = 0.0; self.h = 0.0; self.rho = 0.0; self.att: dict = {}; self.dfn: dict = {}
        self.new_att = 0.0; self.new_def = 0.0; self.n_eff: dict = {}
    def fit(self, home, away, hg, ag, age_days, sample_weight=None):
        home, away = np.asarray(home), np.asarray(away); hg = np.asarray(hg, float); ag = np.asarray(ag, float)
        w = 0.5 ** (np.asarray(age_days, float) / self.hl)
        if sample_weight is not None: w = w * sample_weight
        teams = sorted(set(home) | set(away)); ix = {t: i for i, t in enumerate(teams)}; T = len(teams)
        hi = np.array([ix[t] for t in home]); ai = np.array([ix[t] for t in away])
        def nll(p):
            mu, h, a, d = p[0], p[1], p[2:2 + T], p[2 + T:]
            lh = np.exp(mu + h + a[hi] - d[ai]); la = np.exp(mu + a[ai] - d[hi])
            ll = (w * (hg * np.log(lh) - lh + ag * np.log(la) - la)).sum()
            rh, ra = w * (hg - lh), w * (ag - la)
            ga = np.bincount(hi, rh, T) + np.bincount(ai, ra, T); gd = -(np.bincount(ai, rh, T) + np.bincount(hi, ra, T))
            g = np.concatenate([[rh.sum() + ra.sum(), rh.sum()], ga, gd])
            pen = self.ridge * (a @ a + d @ d); g_pen = np.concatenate([[0, 0], 2 * self.ridge * a, 2 * self.ridge * d])
            return -ll + pen, -g + g_pen
        p0 = np.zeros(2 + 2 * T); p0[0] = np.log(max(hg.mean(), .5))
        r = minimize(nll, p0, jac=True, method="L-BFGS-B")
        self.mu, self.h = r.x[0], r.x[1]; a, d = r.x[2:2 + T], r.x[2 + T:]
        self.att = dict(zip(teams, a)); self.dfn = dict(zip(teams, d))
        self.n_eff = {t: float(w[(home == t) | (away == t)].sum()) for t in teams}
        strength = a + d; q = strength <= np.quantile(strength, 0.25)    # unseen/promoted sides ≈ weakest quartile
        self.new_att, self.new_def = float(a[q].mean()), float(d[q].mean())
        lh = np.exp(self.mu + self.h + a[hi] - d[ai]); la = np.exp(self.mu + a[ai] - d[hi])
        f = lambda rho: -(w * np.log(tau(hg.astype(int), ag.astype(int), lh, la, rho))).sum()
        self.rho = float(minimize_scalar(f, bounds=(-0.3, 0.3), method="bounded").x)
        return self
    def lambdas(self, home, away, neutral=False, att_mult=(1, 1), def_mult=(1, 1)):
        ah, dh = self.att.get(home, self.new_att), self.dfn.get(home, self.new_def)
        aa, da = self.att.get(away, self.new_att), self.dfn.get(away, self.new_def)
        lh = np.exp(self.mu + (0 if neutral else self.h) + ah - da) * att_mult[0] / def_mult[1]
        la = np.exp(self.mu + aa - dh) * att_mult[1] / def_mult[0]
        return float(lh), float(la)
    def known(self, t): return t in self.att

def score_matrix(lh, la, rho=0.0, max_goals=10):
    g = np.arange(max_goals + 1)
    ph = np.exp(-lh + g * np.log(lh) - gammaln(g + 1)); pa = np.exp(-la + g * np.log(la) - gammaln(g + 1))
    m = np.outer(ph, pa)
    m[0, 0] *= max(1 - lh * la * rho, 1e-6); m[0, 1] *= max(1 + lh * rho, 1e-6); m[1, 0] *= max(1 + la * rho, 1e-6); m[1, 1] *= max(1 - rho, 1e-6)
    return m / m.sum()

def outcome_probs(m):
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])   # H (h>a), D, A
