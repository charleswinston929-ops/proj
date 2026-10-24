"""FastAPI service: validated inputs, rate limiting, size limits, security headers, structured errors, metrics, no raw exceptions to clients."""
from __future__ import annotations
import logging, os, time, uuid, collections
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .agent.store import JsonStore
from .agent.tools import ScenarioArg, call_tool
from .agent.analyst import answer

logging.basicConfig(level=logging.INFO, format='{"t":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}'); log = logging.getLogger("si")
app = FastAPI(title="Sports Intelligence API", version="1.0.0"); store: JsonStore | None = None
RATE, WINDOW, MAX_BODY = int(os.getenv("RATE_LIMIT_PER_MIN", "60")), 60.0, 16_384
hits: dict[str, collections.deque] = collections.defaultdict(collections.deque); metrics = collections.Counter(); lat: list[float] = []

def S() -> JsonStore:
    global store
    if store is None: store = JsonStore()
    return store

@app.middleware("http")
async def guard(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]; t0 = time.perf_counter(); ip = (request.client.host if request.client else "anon")
    if int(request.headers.get("content-length") or 0) > MAX_BODY: return JSONResponse({"error": "Request too large."}, 413)
    q = hits[ip]; now = time.time()
    while q and now - q[0] > WINDOW: q.popleft()
    if len(q) >= RATE: metrics["rate_limited"] += 1; return JSONResponse({"error": "Rate limit exceeded."}, 429, headers={"Retry-After": "60"})
    q.append(now)
    try: resp = await call_next(request)
    except Exception:
        log.exception('"unhandled"'); metrics["errors"] += 1; resp = JSONResponse({"error": "Service temporarily unavailable.", "request_id": rid}, 500)
    resp.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store", "X-Request-ID": rid,
                         "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'"})
    ms = (time.perf_counter() - t0) * 1000; metrics["requests"] += 1; lat.append(ms); del lat[:-500]
    log.info('"%s %s %s %.1fms %s"', request.method, request.url.path, resp.status_code, ms, rid); return resp

@app.exception_handler(KeyError)
async def _nf(_, e): return JSONResponse({"error": "Not found."}, 404)

@app.get("/health")
def health(): d = S().d; return dict(status="ok", model=d["model"]["version"], as_of=d["as_of"], last_result=d["data"]["last_result"], awaiting_result=d["data"]["awaiting_result"])
@app.get("/v1/matches")
def matches(limit: int = Query(50, ge=1, le=400)):
    st = S(); rows = sorted(st.up.values(), key=lambda s: s["kickoff_utc"])[:limit]
    return [dict(match_id=s["match_id"], home=st.names[s["home"]], away=st.names[s["away"]], kickoff_utc=s["kickoff_utc"], published=s["published"], probs=s["probs"] if s["published"] else None,
                 confidence=s["confidence"]["level"], data_quality=s["data_quality"]) for s in rows]
@app.get("/v1/predictions/{match_id}")
def prediction(match_id: str): return call_tool(S(), "get_prediction", dict(match_id=match_id[:200]))
@app.get("/v1/predictions/{match_id}/history")
def history(match_id: str): return call_tool(S(), "get_prediction_history", dict(match_id=match_id[:200]))
@app.get("/v1/data-quality/{match_id}")
def dq(match_id: str): return call_tool(S(), "get_data_quality", dict(match_id=match_id[:200]))
@app.get("/v1/models")
def models(): d = S().d; return dict(model=d["model"], metrics=d["backtest"]["metrics"], windows=d["backtest"]["windows"])
@app.post("/v1/scenario")
def scenario(a: ScenarioArg): return call_tool(S(), "run_scenario", a.model_dump())
class Chat(BaseModel): question: str = Field(min_length=2, max_length=300)
@app.post("/v1/chat")
def chat(c: Chat): return answer(S(), c.question)
@app.get("/metrics")
def get_metrics():
    l = sorted(lat); p = lambda q: l[int(q * (len(l) - 1))] if l else 0.0
    return dict(counters=dict(metrics), latency_ms=dict(p50=p(.5), p95=p(.95)), data_age_days=None)
