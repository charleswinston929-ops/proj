import numpy as np, pandas as pd, pytest
from app.features.engine import run_state
from app.pipeline.predict import fit_context, predict, load_cfg
from conftest import ART

def test_features_never_use_the_future(dataset):
    df = dataset[0]; X, _, _ = run_state(df); i = 3000; cut = df.loc[i, "date"]
    X2, _, _ = run_state(df, until=cut + pd.Timedelta(days=1)); assert np.allclose(X.loc[i].values.astype(float), X2.loc[i].values.astype(float))
def test_walk_forward_state_ignores_future_results(dataset):
    """Corrupt every result after the as-of date: forecasts must be bit-identical (spec 67)."""
    df = dataset[0]; cfg = load_cfg(); as_of = pd.Timestamp("2024-01-10"); fx = df[(df.date >= as_of) & (df.date < as_of + pd.Timedelta(days=4))].head(3)
    a = predict(fit_context(df, as_of, cfg, B=3), df, fx, cfg, n_sims=2000)
    d2 = df.copy(); m = d2.date >= as_of; d2.loc[m & d2.played, "hg"] = 9; d2.loc[m & d2.played, "ag"] = 0
    b = predict(fit_context(d2, as_of, cfg, B=3), d2, fx, cfg, n_sims=2000)
    assert [x["probs"] for x in a] == [x["probs"] for x in b]
def test_forecast_properties(dataset):
    df = dataset[0]; cfg = load_cfg(); as_of = pd.Timestamp("2026-10-23"); fx = df[(~df.played) & (df.date >= as_of)].head(5)
    out = predict(fit_context(df, as_of, cfg, B=10), df, fx, cfg, n_sims=4000)
    for s in out:
        assert abs(sum(s["probs"].values()) - 1) < 1e-9 and 0 <= s["data_quality"] <= 1 and s["data_cutoff"] <= s["kickoff_utc"]
        lo, hi = s["simulation"]["interval_90"]["home"]; assert lo <= hi and abs(s["simulation"]["p_home"] + s["simulation"]["p_draw"] + s["simulation"]["p_away"] - 1) < 1e-9
def test_backtest_artifact_beats_climatology_and_is_internally_consistent():
    import json; b = json.loads((ART / "backtest.json").read_text()); t = b["metrics"]["test"]
    assert t["ensemble_raw"]["log_loss"] < t["clim"]["log_loss"] - 0.05 and t["elo"]["n"] == t["ensemble"]["n"] > 2000
    assert b["bootstrap_vs"]["clim"]["hi"] < 0 and abs(sum(b["weights"].values()) - 1) < 1e-9
