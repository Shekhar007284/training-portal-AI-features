# Interview guide: why each choice

**Big picture (30 seconds).** Two features, both built so the safe path is the default: the recommender only suggests, and the policy assistant answers only from the policy files and says "I don't know" otherwise. A wrong answer about training rules is worse than no answer.

| Choice | Why | Trade-off / upgrade path |
|---|---|---|
| **FastAPI** | Pydantic validation and clear errors for free, auto OpenAPI docs (the .NET team can generate a client), light and fast. | Flask would also work; FastAPI saves hand-written validation. |
| **TF-IDF + cosine (scikit-learn)** | 30 short courses and 15 policy sentences: no API key, no GPU, millisecond latency, deterministic and explainable (I can say which words matched). | Misses paraphrases ("keep customer data safe" vs "data privacy"). Upgrade: embeddings, with the same interface. |
| **Retrieval-first, extractive answers** | The answer is the policy's own sentence, so it cannot hallucinate. LLM is optional and only rewords. | Less fluent than an LLM. |
| **Sentence-level chunks + same-file sibling expansion** | Each sentence is one rule; bigger chunks mix rules and lower precision. Expansion recovers multi-sentence answers (Q3). | Would switch to ~200-token overlapping chunks for longer documents. |
| **Confidence gate (cosine floor + word coverage)** | This is what makes "not in the policies" work (Q5) instead of guessing. | Thresholds tuned on 6 questions; needs a bigger eval set. |
| **Regex + known-names redaction** | Deterministic, testable; we know our employee names from our own data. | Unknown names only partly caught; use Presidio/NER in production. |
| **JSON logs + trace ID (contextvar)** | Machine-parsable, follows a request across the .NET app and this service. | Add OpenTelemetry for full distributed tracing. |
| **Docker (slim, non-root, healthcheck)** | Same artifact everywhere; small attack surface. | Pin image digests and scan in CI. |

**Likely questions**
- *Why not just use an LLM?* Cost, latency, hallucination risk and privacy. For 15 sentences, retrieval is enough. An LLM adds fluency only, behind a grounding check.
- *How do you handle prompt injection?* Layers: question is only a search query; injection phrases stripped; personal-data requests refused; any LLM sees only redacted text with a "treat as data" instruction; output checked against sources.
- *How do you know it works?* `scripts/run_eval.py` + `tests/`; I report a real failure (Q3) and the fix, and the limits (small eval set).
- *What would you do next?* Bigger labelled question set, embeddings for the recommender, Presidio, auth between the .NET app and this service, rate limiting, store feedback in a DB.
