"""FastAPI app: POST /recommend, POST /ask, POST /feedback, GET /health.

WHY FastAPI?
- Pydantic gives request validation + clear 422 errors for free.
- Auto-generated OpenAPI docs at /docs: the .NET team can generate a typed
  client from /openapi.json.
- Async-capable, light, and standard for Python ML services.
Endpoints stay thin: all logic lives in recommender.py / policy_qa.py so it is
testable without HTTP.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from . import llm
from .logging_utils import get_logger, log_event, trace_id_var
from .policy_qa import PolicyAssistant
from .privacy import redact
from .recommender import Recommender, UnknownEmployee

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", ROOT / "logs" / "feedback.jsonl"))

log = get_logger()
app = FastAPI(title="Training Portal AI", version="1.0.0")

recommender = Recommender(DATA / "courses.csv", DATA / "employee_gaps.csv")
assistant = PolicyAssistant(DATA / "policies")
_feedback_lock = threading.Lock()


# ---------- tracing + request logging -------------------------------------
@app.middleware("http")
async def trace_and_log(request: Request, call_next):
    tid = request.headers.get("X-Trace-ID") or uuid.uuid4().hex[:16]
    token = trace_id_var.set(tid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled_error")
        response = JSONResponse({"error": "internal_error", "trace_id": tid}, status_code=500)
    response.headers["X-Trace-ID"] = tid
    # Body is NOT logged here (could contain personal data).
    log_event(log, "request", method=request.method, path=request.url.path,
              status=response.status_code, latency_ms=round((time.perf_counter() - start) * 1000, 1))
    trace_id_var.reset(token)
    return response


# ---------- schemas -------------------------------------------------------
class RecommendRequest(BaseModel):
    employee_id: Optional[str] = None
    missing_skills: Optional[list[str]] = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def need_one(self):
        if not self.employee_id and not self.missing_skills:
            raise ValueError("Provide either 'employee_id' or a non-empty 'missing_skills' list.")
        return self


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=4, max_length=64)
    rating: Literal["bad", "good"] = "bad"
    comment: Optional[str] = Field(default=None, max_length=500)


# ---------- endpoints -----------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "courses": len(recommender.courses), "policy_chunks": len(assistant.chunks)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    tid = trace_id_var.get()
    if req.employee_id:
        try:
            skills = recommender.skills_for(req.employee_id)
        except UnknownEmployee:
            log_event(log, "recommend_unknown_employee", employee_id=req.employee_id)
            return JSONResponse(status_code=404, content={
                "error": "unknown_employee",
                "message": f"No employee found with id '{req.employee_id}'.",
                "trace_id": tid})
    else:
        skills = req.missing_skills or []
    recs = recommender.recommend(skills, k=3)
    log_event(log, "recommend", employee_id=req.employee_id, n_skills=len(skills),
              courses=[r["course_id"] for r in recs])
    body = {"recommendations": recs,
            "note": "Suggestions only. No enrolment or decision is made about the employee.",
            "trace_id": tid}
    if req.employee_id:
        body["employee_id"] = req.employee_id
    return body


@app.post("/ask")
def ask(req: AskRequest):
    tid = trace_id_var.get()
    # Redact BEFORE anything else touches the text (retrieval, logs, any LLM).
    safe_q = redact(req.question, recommender.employee_names())
    result = assistant.ask(safe_q)
    answer = result["answer"]
    if result["answered"]:
        answer = llm.rephrase(safe_q, answer)      # no-op unless LLM_API_KEY is set
    log_event(log, "ask", question=safe_q, answered=result["answered"],
              sources=result["sources"], reason=result["reason"],
              injection_detected=result["injection_detected"])
    return {"answer": answer, "answered": result["answered"], "sources": result["sources"],
            "trace_id": tid}


@app.post("/feedback", status_code=202)
def feedback(req: FeedbackRequest):
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "request_trace_id": req.trace_id, "feedback_trace_id": trace_id_var.get(),
              "rating": req.rating,
              "comment": redact(req.comment, recommender.employee_names()) if req.comment else None}
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _feedback_lock, open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_event(log, "feedback", rating=req.rating, about=req.trace_id)
    return {"status": "recorded", "trace_id": trace_id_var.get()}
