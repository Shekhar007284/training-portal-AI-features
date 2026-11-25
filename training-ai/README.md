# Training Portal AI

Two small AI features for the training portal:

| Endpoint | What it does |
|---|---|
| `POST /recommend` | Top-3 course suggestions for an employee ID or a list of missing skills (TF-IDF + cosine similarity). Suggestions only: it never enrols or decides anything. |
| `POST /ask` | Answers training-policy questions **only** from `data/policies/`, names the source file, and says so when the answer isn't there. |
| `POST /feedback` | Flags a bad answer using the `trace_id` returned by any call. |
| `GET /health` | Liveness check. |

## Run

```bash
# Local (Python 3.10+)
pip install -r requirements-dev.txt
uvicorn app.main:app --reload          # docs at http://localhost:8000/docs
pytest -q                              # unit + API tests
python scripts/run_eval.py             # the 6 policy test questions

# Docker
docker build -t training-ai .
docker run -p 8000:8000 -v "$PWD/logs:/srv/logs" training-ai
```

Optional LLM rewording (off by default; answers are already grounded without it):
`LLM_API_KEY=... LLM_MODEL=gpt-4o-mini LLM_BASE_URL=https://api.openai.com/v1` (any OpenAI-compatible endpoint, incl. Ollama/vLLM).

## Example requests

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{"employee_id":"E1001"}'
# {"recommendations":[{"course_id":"C006","title":"Data Handling Basics","score":0.51,
#   "reason":"Covers the missing skill: data privacy"}, ...],"note":"Suggestions only...","trace_id":"..."}

curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{"missing_skills":["phishing","sql"]}'
curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{"employee_id":"E9999"}'
# 404 {"error":"unknown_employee","message":"No employee found with id 'E9999'.","trace_id":"..."}

curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"How long is a safety certificate valid?"}'
# {"answer":"Safety certificates are valid for 12 months.","answered":true,
#  "sources":["certificate_validity.txt"],"trace_id":"..."}

curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What is the policy on working from home?"}'
# {"answer":"I could not find this in the training policies, so I can't answer it.","answered":false,"sources":[],...}

curl -s localhost:8000/feedback -H 'content-type: application/json' -d '{"trace_id":"<from a previous response>","comment":"Answer was incomplete"}'
```

## Deploying and monitoring next to a .NET application

- **Deploy:** run the container as its own internal service (Azure Container Apps / AKS / App Service for Containers), reachable only from the .NET app over the private network. The .NET app calls it with `HttpClient`; a typed client can be generated from `/openapi.json`. Keep the LLM key in Key Vault/secret store, never in the image. Scale on request count; the service is stateless (feedback goes to a mounted volume now, a database/queue in production). Use a short timeout and a fallback in the .NET caller, so the portal still works if the AI service is down.
- **Monitor:** every request emits one JSON log line with `trace_id`, status and latency (ship stdout to Azure Monitor / Application Insights). The .NET app forwards its own correlation ID as the `X-Trace-ID` header, so a request can be followed across both services. Alert on 5xx rate, p95 latency, the share of `/ask` answers with `answered=false`, and the volume of `/feedback` with `rating=bad`.
- **One way to cut model cost:** the default path uses no model at all (retrieval + extractive answers). When an LLM is enabled, only call it for answers that already passed the confidence gate, send the top 1-2 sentences, use a small model, cache by normalised question (already done with an LRU cache), and cap output tokens.

## Layout

`app/` code · `data/` courses, employees, policies, test questions · `scripts/run_eval.py` · `tests/` · `NOTES.md` (results + safety note) · `INTERVIEW_GUIDE.md`
