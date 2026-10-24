"""Monte-Carlo with REAL parameter uncertainty: the goal model is refit on B bootstrap resamples of the training matches;
each simulated match first draws a parameter set, then a scoreline. (v0.1 simply resampled the same Poisson, adding nothing.)"""
from __future__ import annotations
import numpy as np
from .dixon_coles import DixonColes, score_matrix, outcome_probs

def bootstrap_models(tr, as_of, hl, ridge, B=40, seed=7):
    rng = np.random.default_rng(seed); models = []; age = (as_of - tr.date).dt.days.values
    for _ in range(B):
        i = rng.integers(0, len(tr), len(tr))
        models.append(DixonColes(hl, ridge).fit(tr.home.values[i], tr.away.values[i], tr.hg.values[i], tr.ag.values[i], age[i]))
    return models

def simulate_fixture(models, home, away, n=20000, seed=11, max_goals=10):
    rng = np.random.default_rng(seed); B = len(models); per = np.zeros((B, 3)); mats = []; lams = []
    for b, m in enumerate(models):
        l = m.lambdas(home, away); mx = score_matrix(*l, m.rho, max_goals); mats.append(mx); per[b] = outcome_probs(mx); lams.append(l)
    mats = np.array(mats); k = max_goals + 1
    per_b = max(1, n // B); hg_l, ag_l = [], []
    for b in range(B):
        flat = rng.choice(k * k, per_b, p=mats[b].ravel()); hg_l.append(flat // k); ag_l.append(flat % k)
    hg, ag = np.concatenate(hg_l), np.concatenate(ag_l); n = len(hg)
    mean = mats.mean(0); top = np.dstack(np.unravel_index(np.argsort(-mean.ravel())[:8], mean.shape))[0]
    lo, hi = np.percentile(per, 5, 0), np.percentile(per, 95, 0)
    return dict(simulations=n, bootstrap_models=B, p_home=float((hg > ag).mean()), p_draw=float((hg == ag).mean()), p_away=float((hg < ag).mean()),
                interval_90=dict(home=[float(lo[0]), float(hi[0])], draw=[float(lo[1]), float(hi[1])], away=[float(lo[2]), float(hi[2])]),
                expected_goals=[float(np.mean([l[0] for l in lams])), float(np.mean([l[1] for l in lams]))],
                over_2_5=float(((hg + ag) > 2.5).mean()), btts=float(((hg > 0) & (ag > 0)).mean()),
                top_scorelines=[dict(score=f"{int(h)}-{int(a)}", probability=float(mean[h, a])) for h, a in top],
                matrix_6x6=[[float(mean[h, a]) for a in range(6)] for h in range(6)])
