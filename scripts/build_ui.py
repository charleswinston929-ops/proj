"""Build the standalone dashboard (apps/dashboard/index.html) with real model output embedded. Run from repo root: python scripts/build_ui.py"""
import json, pathlib
R = pathlib.Path(__file__).resolve().parents[1]; A = R / "artifacts"
def rnd(o, d=4):
    if isinstance(o, float): return round(o, d)
    if isinstance(o, list): return [rnd(x, d) for x in o]
    if isinstance(o, dict): return {k: rnd(v, d) for k, v in o.items()}
    return o
app = json.loads((A / "app_data.json").read_text()); ev = json.loads((A / "evaluation.json").read_text())
fx = []
for s in sorted(app["upcoming"], key=lambda s: (s["kickoff_utc"], s["home"])):
    h = [dict(as_of=x["as_of"], kind=x["kind"], p=[x["probs"]["home"], x["probs"]["draw"], x["probs"]["away"]], dq=x["data_quality"], lin=x.get("lineage", []), lh=x["lam_h"], la=x["lam_a"],
              st=x["strengths"]) for x in app["history"].get(s["match_id"], [])]
    sim = s["simulation"]
    fx.append(dict(id=s["match_id"], h=s["home"], a=s["away"], date=s["date"], ko=s["kickoff_utc"], md=s["matchday"], p=[s["probs"]["home"], s["probs"]["draw"], s["probs"]["away"]], comp=s["components"],
                   lh=s["lam_h"], la=s["lam_a"], rho=s["rho"], hadv=s["home_adv_log"], w=s["weights"], dq=s["data_quality"], dqp=s["dq_parts"], band=s["dq_band"], miss=s["missing_results"],
                   conf=s["confidence"], dis=s["disagreement"], pub=s["published"], st=s["strengths"],
                   sim=dict(iv=sim["interval_90"], top=sim["top_scorelines"], mat=sim["matrix_6x6"], o25=sim["over_2_5"], btts=sim["btts"], n=sim["simulations"], b=sim["bootstrap_models"]), hist=h))
D = dict(meta=dict(as_of=app["as_of"], generated=app["generated_at"], model=app["model"]["name"] + " " + app["model"]["version"], last_result=app["data"]["last_result"], awaiting=app["data"]["awaiting_result"],
                   awaiting_list=app["data"]["awaiting"][:12], played=app["data"]["matches_played"], source=app["data"]["source"], weights=app["model"]["weights"], tuned=app["model"]["tuned"]),
         names=app["names"], teams=app["teams"], fx=fx, ev=ev)
if (A / "availability.json").exists():
    av = json.loads((A / "availability.json").read_text()); D["avail"] = dict(as_of=av["as_of"], source=av["source"], summaries=av["summaries"])
tpl = (R / "apps/dashboard/template.html").read_text(); out = tpl.replace("__DATA__", json.dumps(rnd(D), separators=(",", ":")).replace("</", "<\\/"))
(R / "apps/dashboard/index.html").write_text(out); print("built", len(out) // 1024, "KB;", len(fx), "fixtures")
