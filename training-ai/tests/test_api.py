"""HTTP-level tests. Skipped automatically if fastapi/httpx are not installed."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_recommend_by_employee():
    r = client.post("/recommend", json={"employee_id": "E1001"})
    assert r.status_code == 200 and len(r.json()["recommendations"]) <= 3
    assert "X-Trace-ID" in r.headers


def test_recommend_unknown_employee_is_404():
    r = client.post("/recommend", json={"employee_id": "NOPE"})
    assert r.status_code == 404 and r.json()["error"] == "unknown_employee"


def test_recommend_needs_input():
    assert client.post("/recommend", json={}).status_code == 422


def test_ask_and_feedback_roundtrip(tmp_path, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "FEEDBACK_PATH", tmp_path / "fb.jsonl")
    a = client.post("/ask", json={"question": "How long is a safety certificate valid?"}).json()
    assert "12 months" in a["answer"] and a["sources"] == ["certificate_validity.txt"]
    f = client.post("/feedback", json={"trace_id": a["trace_id"], "comment": "wrong, ask aarav@corp.com"})
    assert f.status_code == 202
    assert "aarav@corp.com" not in (tmp_path / "fb.jsonl").read_text()
