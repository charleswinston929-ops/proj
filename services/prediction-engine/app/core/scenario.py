"""Scenario engine (spec section 38): pure function over a stored baseline; never mutates history.
Only the goal model (Dixon-Coles) can react to a scenario; under a linear opinion pool the ensemble moves by w_dc * ΔP_dc.
IMPORTANT: no player-level data exists yet, so 'key player out' style inputs are user ASSUMPTIONS about team attack/defence, not estimates."""
from __future__ import annotations
import numpy as np
from .dixon_coles import score_matrix, outcome_probs
from .ensemble import Calibrator

def run_scenario(baseline: dict, changes: dict) -> dict:
    """baseline: lam_h, lam_a, rho, home_adv_log, components{elo,dc,gb}, weights{elo,dc,gb}, calibrator."""
    lh, la, rho = baseline["lam_h"], baseline["lam_a"], baseline["rho"]
    f = lambda k, d=0.0: 1.0 + changes.get(k, d) / 100.0
    lh2 = lh * f("home_attack_pct") * f("away_concede_pct"); la2 = la * f("away_attack_pct") * f("home_concede_pct")
    if changes.get("neutral_venue"): lh2 *= float(np.exp(-baseline["home_adv_log"]))
    if lh2 <= 0 or la2 <= 0: raise ValueError("adjustments must keep expected goals positive")
    dc0 = outcome_probs(score_matrix(lh, la, rho)); dc1 = outcome_probs(score_matrix(lh2, la2, rho))
    w = baseline["weights"]; comp = {k: np.array(v) for k, v in baseline["components"].items()}
    raw0 = sum(w[k] * comp[k] for k in w); raw1 = raw0 + w["dc"] * (dc1 - dc0)           # only the DC component changes, measured against its own baseline (a no-op scenario is exactly zero)
    cal = Calibrator.from_dict(baseline["calibrator"])
    p0 = cal.apply((raw0 / raw0.sum())[None])[0]; p1 = cal.apply((raw1 / raw1.sum())[None])[0]
    return dict(changes=changes, baseline=dict(home=p0[0], draw=p0[1], away=p0[2], lam_h=lh, lam_a=la),
                scenario=dict(home=p1[0], draw=p1[1], away=p1[2], lam_h=lh2, lam_a=la2),
                delta_pp=dict(home=100 * (p1[0] - p0[0]), draw=100 * (p1[1] - p0[1]), away=100 * (p1[2] - p0[2])),
                note="Scenario applies to the goal model only; inputs are assumptions, not player-level estimates.")
