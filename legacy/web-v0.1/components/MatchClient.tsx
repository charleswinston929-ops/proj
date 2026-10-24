"use client";
import {useEffect,useState} from "react";

const API=process.env.NEXT_PUBLIC_PREDICTION_API_URL||"http://localhost:8000";

export default function MatchClient({matchId}:{matchId:string}){
 const [data,setData]=useState<any>(null); const [error,setError]=useState(""); const [loading,setLoading]=useState(true);
 useEffect(()=>{fetch(`${API}/v1/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({match_id:matchId})})
 .then(r=>{if(!r.ok)throw Error("Prediction service unavailable");return r.json()}).then(setData).catch(e=>setError(e.message)).finally(()=>setLoading(false))},[matchId]);

 return <main className="container">
  <div style={{padding:"30px 0"}}><div className="label">MATCH INTELLIGENCE</div><h1>Northbridge FC vs Riverside United</h1><p className="muted">Demo fixture · pre-match forecast</p></div>
  {loading&&<div className="card">Running models…</div>}
  {error&&<div className="card"><strong>Connection error</strong><p>{error}. Start FastAPI on port 8000.</p></div>}
  {data&&<>
   <div className="grid">{[["Home",data.probabilities.home],["Draw",data.probabilities.draw],["Away",data.probabilities.away]].map(([label,value])=><div className="card" key={String(label)}><div className="label">{label}</div><div className="big">{(Number(value)*100).toFixed(1)}%</div><div className="bar"><div className="fill" style={{width:`${Number(value)*100}%`}}/></div></div>)}</div>
   <div className="grid2" style={{marginTop:16}}><div className="card"><div className="label">Expected goals</div><div className="big">{data.expected_goals.home.toFixed(2)} — {data.expected_goals.away.toFixed(2)}</div></div><div className="card"><div className="label">Data quality / confidence</div><div className="big">{(data.data_quality*100).toFixed(0)}%</div><p className="muted">{data.confidence}</p></div></div>
   <div className="grid2" style={{marginTop:16}}><div className="card"><div className="label">Likely scorelines</div>{data.top_scorelines.map((x:any)=><div key={x.score} style={{display:"flex",justifyContent:"space-between",padding:"8px 0"}}><span>{x.score}</span><span>{(x.probability*100).toFixed(1)}%</span></div>)}</div>
   <div className="card"><div className="label">Independent simulation</div><p>{data.monte_carlo.simulations.toLocaleString()} simulations</p><p>Home {(data.monte_carlo.home*100).toFixed(1)}%</p><p>Draw {(data.monte_carlo.draw*100).toFixed(1)}%</p><p>Away {(data.monte_carlo.away*100).toFixed(1)}%</p><p className="muted">Model disagreement: {(data.model_disagreement*100).toFixed(2)} percentage points.</p></div></div>
   <div className="card" style={{marginTop:16}}><div className="label">Forecast provenance</div><ul>{data.explanation.map((x:string)=><li key={x}>{x}</li>)}</ul><p className="muted">Model {data.model_version} · features {data.feature_version}</p></div>
  </>}
 </main>
}
