"""Linear opinion pool with weights LEARNED from out-of-sample predictions (spec section 33) + vector-scaling calibration."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .metrics import logloss

def pool(Ps: list[np.ndarray], w) -> np.ndarray:
    P = sum(wi * Pi for wi, Pi in zip(w, Ps)); return P / P.sum(1, keepdims=True)

def fit_weights(Ps, y):
    k = len(Ps)
    f = lambda z: logloss(pool(Ps, np.exp(z) / np.exp(z).sum()), y)
    r = minimize(f, np.zeros(k), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7, maxiter=800))
    w = np.exp(r.x) / np.exp(r.x).sum(); return w

class Calibrator:
    """z' = log(p)/T + b (b_home fixed 0). 3 parameters: deliberately low capacity so it cannot overfit."""
    def __init__(self): self.logT = 0.0; self.b = np.zeros(2)
    def apply(self, P):
        z = np.log(np.clip(P, 1e-9, 1)) / np.exp(self.logT) + np.array([0.0, *self.b]); z -= z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)
    def fit(self, P, y):
        def f(t): self.logT, self.b = t[0], t[1:]; return logloss(self.apply(P), y)
        r = minimize(f, [0.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-8, maxiter=500))
        self.logT, self.b = r.x[0], r.x[1:]; return self
    def to_dict(self): return dict(logT=float(self.logT), b=[float(x) for x in self.b])
    @classmethod
    def from_dict(cls, d): c = cls(); c.logT = d["logT"]; c.b = np.array(d["b"]); return c

def disagreement(components: list[np.ndarray]) -> np.ndarray:
    """Mean across outcomes of the std-dev across component models (per match)."""
    return np.stack(components, 0).std(0).mean(1)
