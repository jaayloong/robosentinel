from fastapi.testclient import TestClient
from robosentinel.api import app

def test_health():
    r=TestClient(app).get("/health"); assert r.status_code==200; assert r.json()["status"]=="ok"

def test_demo_endpoint():
    r=TestClient(app).get("/api/demo?n=300&seed=5"); assert r.status_code==200; body=r.json(); assert len(body["rows"])==300; assert "summary" in body
