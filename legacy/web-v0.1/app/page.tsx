import Link from "next/link";

export default function Home(){
 return <main className="container">
  <div style={{maxWidth:760,padding:"70px 0 45px"}}>
   <div className="label">FOOTBALL FORECASTING INTELLIGENCE</div>
   <h1 style={{fontSize:52,lineHeight:1.05}}>Forecasts you can inspect, not predictions you are asked to trust.</h1>
   <p className="muted" style={{fontSize:18}}>Team strength, goal models, simulation, data quality and prediction history.</p>
  </div>
  <div className="grid">
   <div className="card"><div className="label">Model</div><div className="big">Elo + Poisson</div><p className="muted">Baseline ensemble</p></div>
   <div className="card"><div className="label">Simulation</div><div className="big">100K</div><p className="muted">Monte Carlo trials</p></div>
   <div className="card"><div className="label">Status</div><div className="big">V0.1</div><p className="muted">Demo / validation stage</p></div>
  </div>
  <div className="card" style={{marginTop:16}}><div className="label">Featured match</div><h2>Northbridge FC vs Riverside United</h2><p className="muted">Synthetic fixture used to verify the complete application loop.</p><Link href="/match/demo-001"><button>Open Match Intelligence</button></Link></div>
 </main>;
}
