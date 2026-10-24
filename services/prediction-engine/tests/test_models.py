import numpy as np, pandas as pd, pytest
from scipy.optimize import check_grad
from app.core.elo import Elo, OrderedLogit
from app.core.dixon_coles import DixonColes, score_matrix, outcome_probs
from app.core import metrics as M, ensemble as E, quality as Q
from app.core.scenario import run_scenario

def test_elo_update_is_zero_sum_and_directional():
    e = Elo(); e.new_season("s", ["a", "b"]); tot = e.rating("a") + e.rating("b"); e.update("a", "b", 3, 0)
    assert abs(e.rating("a") + e.rating("b") - tot) < 1e-9 and e.rating("a") > e.rating("b")
def test_ordered_logit_sums_to_one_and_monotone():
    om = OrderedLogit(); P = om.proba(np.linspace(-2, 2, 9)); assert np.allclose(P.sum(1), 1) and (np.diff(P[:, 0]) > 0).all() and (np.diff(P[:, 2]) < 0).all()
def test_ordered_logit_fit_recovers_signal():
    rng = np.random.default_rng(0); x = rng.normal(0, .6, 4000); P = OrderedLogit().proba(x); y = np.array([rng.choice(3, p=p) for p in P])
    om = OrderedLogit().fit(x, y); assert abs(om.theta[0] - 1.0) < 0.25
def test_score_matrix_normalised_and_dc_effect():
    m0, m1 = score_matrix(1.4, 1.1, 0.0), score_matrix(1.4, 1.1, -0.1); assert abs(m1.sum() - 1) < 1e-12 and m1[0, 0] > m0[0, 0] and abs(outcome_probs(m1).sum() - 1) < 1e-12
def test_dc_recovers_known_parameters():
    rng = np.random.default_rng(1); teams = [f"t{i}" for i in range(8)]; att = dict(zip(teams, rng.normal(0, .25, 8))); de = dict(zip(teams, rng.normal(0, .25, 8)))
    rows = [(h, a) for h in teams for a in teams if h != a] * 12; hg, ag = [], []
    for h, a in rows: hg.append(rng.poisson(np.exp(0.1 + 0.3 + att[h] - de[a]))); ag.append(rng.poisson(np.exp(0.1 + att[a] - de[h])))
    m = DixonColes(half_life_days=1e9, ridge=0.1).fit([r[0] for r in rows], [r[1] for r in rows], hg, ag, np.zeros(len(rows)))
    assert abs(m.h - 0.3) < 0.08 and np.corrcoef([m.att[t] for t in teams], [att[t] for t in teams])[0, 1] > 0.9
def test_dc_time_decay_downweights_old_matches():
    # old period: a beats b at both venues; recent period: b beats a at both venues
    h = ["a"] * 20 + ["b"] * 20 + ["a"] * 20 + ["b"] * 20; aw = ["b"] * 20 + ["a"] * 20 + ["b"] * 20 + ["a"] * 20
    hg = [3] * 20 + [0] * 20 + [0] * 20 + [3] * 20; ag = [0] * 20 + [3] * 20 + [3] * 20 + [0] * 20
    age = np.array([1000] * 40 + [1] * 40)
    decayed = DixonColes(100, 0.5).fit(h, aw, hg, ag, age); flat = DixonColes(1e9, 0.5).fit(h, aw, hg, ag, age)
    lh, la = decayed.lambdas("a", "b"); assert la > 1.5 * lh                      # recent evidence dominates: b is stronger
    lh2, la2 = flat.lambdas("a", "b"); assert abs(np.log(la2 / lh2)) < 0.25      # without decay the two periods cancel

def test_metrics_known_values():
    y = np.array([0, 1, 2]); U = np.full((3, 3), 1 / 3); assert abs(M.logloss(U, y) - np.log(3)) < 1e-9
    Pf = np.eye(3); assert M.logloss(Pf, y) < 1e-9 and M.brier(Pf, y) == 0 and M.rps(Pf, y) == 0 and M.accuracy(Pf, y) == 1
def test_ensemble_weights_simplex_and_pool():
    rng = np.random.default_rng(2); y = rng.integers(0, 3, 600); good = np.eye(3)[y] * .5 + .5 / 3; bad = np.full((600, 3), 1 / 3)
    w = E.fit_weights([good, bad], y); assert abs(w.sum() - 1) < 1e-9 and w[0] > .9 and np.allclose(E.pool([good, bad], w).sum(1), 1)
def test_calibrator_default_is_identity_and_fixes_overconfidence():
    P = np.array([[.5, .3, .2]]); assert np.allclose(E.Calibrator().apply(P), P)
    rng = np.random.default_rng(3); true = np.tile([.5, .3, .2], (3000, 1)); y = np.array([rng.choice(3, p=p) for p in true]); over = np.tile([.8, .1, .1], (3000, 1))
    c = E.Calibrator().fit(over, y); assert M.logloss(c.apply(over), y) < M.logloss(over, y) - .2
def test_quality_formula_bands_and_confidence_cap():
    assert Q.data_quality(1, 1, 1, 1, 1) == 1.0 and Q.dq_band(.49) == "blocked" and Q.dq_band(.62) == "low-data-confidence" and Q.dq_band(.9) == "normal"
    assert Q.confidence(.62, 0.0, 0.0)["level"] == "low" and Q.confidence(.4, 0, 0)["level"] == "blocked" and Q.confidence(.95, 0.0, 0.05)["level"] == "high"
    assert Q.freshness_score(0) == 1 and Q.freshness_score(4) < Q.freshness_score(2) and Q.completeness_score(0, 0, 0) >= 0.24

BASE = dict(lam_h=1.6, lam_a=1.0, rho=-.06, home_adv_log=.28, components=dict(elo=[.5, .26, .24], dc=[.52, .25, .23], gb=[.5, .25, .25]), weights=dict(elo=.6, dc=.4, gb=0), calibrator=dict(logT=0, b=[0, 0]))
def test_scenario_noop_boost_and_immutability():
    import copy; b0 = copy.deepcopy(BASE); r0 = run_scenario(BASE, {}); assert all(abs(v) < 1e-9 for v in r0["delta_pp"].values())
    r = run_scenario(BASE, {"home_attack_pct": 15}); assert r["delta_pp"]["home"] > 0 > r["delta_pp"]["away"] and abs(sum(r["delta_pp"].values())) < 1e-9 and BASE == b0
    assert run_scenario(BASE, {"neutral_venue": True})["delta_pp"]["home"] < 0
def test_scenario_rejects_nonsense():
    with pytest.raises(ValueError): run_scenario(BASE, {"home_attack_pct": -100})
