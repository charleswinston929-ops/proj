from __future__ import annotations
import numpy as np
EPS = 1e-12
def onehot(y): o = np.zeros((len(y), 3)); o[np.arange(len(y)), y] = 1; return o
def per_match_logloss(P, y): return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))
def logloss(P, y): return float(per_match_logloss(P, y).mean())
def brier(P, y): return float(((P - onehot(y)) ** 2).sum(1).mean())
def rps(P, y):   # ranked probability score over ordered outcomes H<D<A (standard football metric)
    return float((0.5 * ((np.cumsum(P, 1) - np.cumsum(onehot(y), 1)) ** 2)[:, :2].sum(1)).mean())
def accuracy(P, y): return float((P.argmax(1) == y).mean())
def reliability(p, o, bins=10):
    edges = np.linspace(0, 1, bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum(): out.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), predicted=float(p[m].mean()), observed=float(o[m].mean())))
    return out
def ece(P, y, bins=10):
    O = onehot(y); tot = 0.0
    for k in range(3):
        for b in reliability(P[:, k], O[:, k], bins): tot += b["n"] * abs(b["predicted"] - b["observed"])
    return float(tot / (3 * len(y)))
def summary(P, y): return dict(n=int(len(y)), log_loss=logloss(P, y), brier=brier(P, y), rps=rps(P, y), accuracy=accuracy(P, y), ece=ece(P, y))
def bootstrap_diff(l_a, l_b, n=2000, seed=0):
    """Paired bootstrap of mean(l_a - l_b) over matches. Negative = A better."""
    d = np.asarray(l_a) - np.asarray(l_b); rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return dict(mean=float(d.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))
