# Run from repo root. Python 3.12+. `make setup && make all`
E = cd services/prediction-engine &&
setup: ; pip install -r services/prediction-engine/requirements.txt
fetch: ; $(E) python -m app.pipeline.fetch          # refresh current season from the provider
backtest: ; $(E) python -m app.pipeline.backtest    # ~5 min: tunes on validation, walk-forward, writes artifacts/backtest*.{json,csv}
forecast: ; $(E) python -m app.pipeline.run_all $(shell date -u +%F)
evaluate: ; $(E) python -m app.pipeline.evaluate
db: ; $(E) python -m app.pipeline.load_db --reset   # needs DATABASE_URL
ui: ; python scripts/build_ui.py
test: ; $(E) python -m pytest -q
api: ; $(E) uvicorn app.main:app --port 8000
availability: ; $(E) python -m app.pipeline.availability_fetch
refresh: ; -$(MAKE) fetch; -$(MAKE) availability; $(MAKE) forecast evaluate ui
all: backtest forecast evaluate ui test
