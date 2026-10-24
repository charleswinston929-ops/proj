# Sports Intelligence: football forecasting platform

Real Premier League results (2010-11 to today) → entity resolution → leak-free features → Elo + time-decayed Poisson/Dixon-Coles + gradient boosting →
walk-forward backtest → ensemble → bootstrap Monte-Carlo → immutable prediction snapshots in PostgreSQL → API → dashboard.
The numbers come from models; the analyst layer can only call allowlisted tools and its output is checked so every figure traces to a tool result.

## Honest status
| Area | State |
|---|---|
| Data (openfootball, 17 seasons, 6,460 matches, 42 clubs) | **Real**, validated. Goals only: no xG, shots, lineups. Feed can lag (currently ends 2026-09-20). |
| Models, backtest, calibration check, simulation, scenarios, lineage, data quality | **Built and tested** (41 tests, incl. PostgreSQL immutability). |
| Out-of-sample result (2,710 matches since Aug 2019) | log loss **0.9826** vs 1.0716 base rates; beats Elo by only ~0.002 (marginal). Dixon-Coles ≈ Poisson; GBM does not help; calibrator disabled. |
| Injuries / availability | FPL public-API adapter, DB upsert and *suggested-scenario* logic **built and unit-tested on a synthetic fixture; the live endpoint is untested** (sandbox could not reach it). Effects are unvalidated assumptions and never applied to published forecasts. |
| Lineups | Ingestion contract, validation, status transitions and missing-regular suggestions built and tested. **No lineup feed is connected**; supply JSON from any provider. |
| News | Safe pipeline (injection, conflict, corroboration rules) exists. **No live feed connected → no effect on forecasts.** |
| AI analyst | Deterministic tool-orchestrated analyst + number-provenance guard. **No LLM connected.** |
| Dashboard | `apps/dashboard/index.html` (static, real data embedded). The v0.1 Next.js app is in `legacy/` and is not wired to the new API. |
| Deployment | `netlify.toml` + `DEPLOY.md` + daily-refresh workflow written; **not deployed by me** and the workflow is untested. Auth and monitoring dashboards not done (API has rate limits, size limits, security headers, `/metrics`). |

## Quick start
```
make setup && make all        # backtest (~5 min), forecast, evaluate, build dashboard, run tests
docker compose up -d postgres # migrations auto-apply
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/sports_intelligence make db
make api                      # http://localhost:8000/docs
make refresh                  # re-pull current season, re-forecast, rebuild dashboard
```
## Layout
`services/prediction-engine/app`: `providers/` (adapter interface + openfootball), `entities.py`, `features/engine.py` (single state machine for train/backtest/serve), `core/` (elo, dixon_coles, ensemble, metrics, quality, simulate, scenario), `pipeline/` (backtest, predict, evaluate, load_db, news), `agent/`, `main.py`.
`database/migrations/` PostgreSQL schema (predictions, feature snapshots and lineage are append-only via triggers).

## Before using this for anything with money
Forecasts are probabilistic estimates, not advice. Without odds data there is no edge claim and none is made. If you add odds or a paid tier, get legal advice first (Germany: GlüStV) and add age gating and responsible-gambling messaging.
