from __future__ import annotations
import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from .detector import RoboSentinelDetector
from .evaluation import evaluate_labeled
from .simulator import generate_telemetry

app = FastAPI(title="RoboSentinel API", version="0.1.0", description="Explainable anomaly detection for robot telemetry.")

@app.get("/health")
def health(): return {"status":"ok","service":"robosentinel"}

def _summary(df):
    anomalies = df[df["is_anomaly"] == 1]
    return {"samples":int(len(df)), "anomalies":int(len(anomalies)), "anomaly_rate":round(float(len(anomalies)/max(1,len(df))),4), "max_score":round(float(df["anomaly_score"].max()),4), "top_root_causes":anomalies["root_cause_1"].value_counts().head(5).to_dict()}

@app.get("/api/demo")
def demo(n: int = 900, seed: int = 42):
    n = min(max(n,150),5000); result = RoboSentinelDetector().fit_score_split(generate_telemetry(n=n,seed=seed)); scored = result.frame
    return {"threshold":result.threshold,"metrics":evaluate_labeled(scored),"summary":_summary(scored),"rows":scored.round(5).to_dict(orient="records")}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), baseline_fraction: float = .35):
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(status_code=400, detail="Upload a CSV file.")
    try:
        df = pd.read_csv(io.BytesIO(await file.read())); result = RoboSentinelDetector().fit_score_split(df, baseline_fraction=baseline_fraction)
    except (ValueError,pd.errors.ParserError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    scored=result.frame; return {"threshold":result.threshold,"metrics":evaluate_labeled(scored),"summary":_summary(scored),"rows":scored.round(5).to_dict(orient="records")}

_DASHBOARD='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>RoboSentinel</title><style>body{font-family:system-ui;background:#07111f;color:#e8f0f7;max-width:1100px;margin:auto;padding:40px}h1{font-size:42px;margin-bottom:5px}b{color:#55d6ff}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#12263a;padding:18px;border-radius:14px}.v{font-size:28px;font-weight:800}.muted{color:#8fa6b8}button{background:#55d6ff;border:0;padding:10px 15px;border-radius:9px;font-weight:700;margin:8px}table{width:100%;margin-top:20px;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #234;text-align:left}@media(max-width:700px){.grid{grid-template-columns:1fr 1fr}}</style></head><body><h1>Robo<b>Sentinel</b></h1><p class="muted">Explainable anomaly detection for robot telemetry.</p><button onclick="loadDemo()">Generate Demo</button><input id="file" type="file" accept=".csv"><button onclick="upload()">Analyze CSV</button><div class="grid"><div class="card">Samples<div class="v" id="samples">-</div></div><div class="card">Anomalies<div class="v" id="anomalies">-</div></div><div class="card">Rate<div class="v" id="rate">-</div></div><div class="card">F1<div class="v" id="f1">-</div></div></div><table><thead><tr><th>Time</th><th>Score</th><th>Severity</th><th>Root cause</th></tr></thead><tbody id="rows"></tbody></table><script>function render(d){let s=d.summary,m=d.metrics||{};samples.textContent=s.samples;anomalies.textContent=s.anomalies;rate.textContent=(s.anomaly_rate*100).toFixed(1)+'%';f1.textContent=m.f1??'N/A';rows.innerHTML=[...d.rows].sort((a,b)=>b.anomaly_score-a.anomaly_score).slice(0,15).map(x=>`<tr><td>${x.timestamp_ms??'-'}</td><td>${x.anomaly_score.toFixed(3)}</td><td>${x.severity}</td><td>${x.root_cause_1}</td></tr>`).join('')}async function loadDemo(){render(await(await fetch('/api/demo?n=1000&seed='+Math.floor(Math.random()*1000))).json())}async function upload(){let f=document.getElementById('file').files[0];if(!f)return alert('Choose a CSV');let fd=new FormData();fd.append('file',f);let r=await fetch('/api/analyze',{method:'POST',body:fd});let d=await r.json();if(!r.ok)return alert(d.detail);render(d)}loadDemo()</script></body></html>'''

@app.get("/", response_class=HTMLResponse)
def dashboard(): return _DASHBOARD
