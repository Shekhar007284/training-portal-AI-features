"""Policy assistant: retrieval-first, grounded, refuses when unsure.

WHY this design (RAG without a mandatory LLM)?
- Policies are short, factual and legal-ish: a wrong answer is worse than "I
  don't know". So the DEFAULT answer is *extractive*: we return the policy's
  own sentences, so it cannot invent facts. An LLM is an optional layer that
  only rephrases the retrieved text (see llm.py).
- Prompt injection is mostly neutralised by construction: the question is only
  ever used as a search query, never as instructions. We also strip obvious
  injection phrases and refuse requests for personal data.
- Chunking = one sentence per chunk. Policy files are 2-3 sentences, so a
  sentence is one complete "rule". Bigger chunks would mix unrelated rules
  (e.g. validity + renewal) and lower precision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .privacy import asks_for_personal_data, split_injection
from .text_utils import tokenize

NO_ANSWER = "I could not find this in the training policies, so I can't answer it."
REFUSAL = ("I can only answer questions about the training policies. I can't change my "
           "rules, and I can't share employee personal data such as email addresses.")

# Lexical gaps between how people ask and how policies are written.
SYNONYMS: dict[str, str] = {
    "retry": "attempt", "retri": "attempt", "retake": "attempt", "resit": "attempt",
    "try": "attempt", "tri": "attempt",
    "remind": "reminder", "notifi": "reminder", "notif": "reminder",
}

# Retrieval thresholds (tuned on the 6 test questions, see NOTES.md).
MIN_COSINE = 0.10      # similarity floor
MIN_COVERAGE = 0.40    # share of the question's content words found in the chunk


@dataclass
class Chunk:
    source: str
    text: str


def load_chunks(policy_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(policy_dir.glob("*.txt")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        # Split on sentence end, but not on decimals like "3.5".
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
            if sent.strip():
                chunks.append(Chunk(path.name, sent.strip()))
    return chunks


class PolicyAssistant:
    def __init__(self, policy_dir: Path):
        self.chunks = load_chunks(policy_dir)
        tok = lambda t: tokenize(t, SYNONYMS)  # noqa: E731
        self.tok = tok
        self.vectorizer = TfidfVectorizer(tokenizer=tok, lowercase=False,
                                          token_pattern=None, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        self.chunk_terms = [set(tok(c.text)) for c in self.chunks]

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_terms = set(self.tok(query))
        if not q_terms:
            return []
        qv = self.vectorizer.transform([query])
        sims = (qv @ self.matrix.T).toarray()[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: -sims[i])[:k]
        return [{
            "idx": i,
            "source": self.chunks[i].source,
            "text": self.chunks[i].text,
            "score": float(sims[i]),
            "coverage": len(q_terms & self.chunk_terms[i]) / len(q_terms),
        } for i in ranked]

    def ask(self, question: str) -> dict:
        remaining, injected = split_injection(question)

        # 1) Guardrails first.
        if asks_for_personal_data(question):
            return self._result(REFUSAL, [], answered=False, injected=injected, reason="personal_data_request")

        query = remaining if injected else question
        if injected and not self.tok(query):
            return self._result(REFUSAL, [], answered=False, injected=True, reason="injection_only")

        # 2) Retrieve + confidence gate.
        hits = self.retrieve(query)
        good = [h for h in hits if h["score"] >= MIN_COSINE and h["coverage"] >= MIN_COVERAGE]
        if not good:
            return self._result(NO_ANSWER, hits, answered=False, injected=injected, reason="below_threshold")

        # 3) Compose an extractive answer from the best chunk, then add
        #    "complementary" sentences from the SAME file that cover question
        #    words the best chunk missed (e.g. "who" + "how quickly" in Q3).
        top = good[0]
        q_terms = set(self.tok(query))
        covered = q_terms & self.chunk_terms[top["idx"]]
        picked = {top["idx"]}
        for h in good[1:]:                                   # close-scoring hits
            if h["score"] >= 0.8 * top["score"]:
                picked.add(h["idx"])
        for i, c in enumerate(self.chunks):                  # same-file siblings
            if c.source == top["source"] and i not in picked:
                new_terms = (q_terms - covered) & self.chunk_terms[i]
                if new_terms:
                    picked.add(i)
                    covered |= new_terms
        order = sorted(picked)
        used = [{"idx": i, "source": self.chunks[i].source, "text": self.chunks[i].text,
                 "score": next((h["score"] for h in hits if h["idx"] == i), 0.0),
                 "coverage": len(q_terms & self.chunk_terms[i]) / len(q_terms)} for i in order]
        answer = " ".join(u["text"] for u in used)
        return self._result(answer, used, answered=True, injected=injected, reason="ok")

    @staticmethod
    def _result(answer, hits, answered, injected, reason):
        sources = sorted({h["source"] for h in hits}) if answered else []
        return {
            "answer": answer,
            "answered": answered,
            "sources": sources,
            "retrieved": [{"source": h["source"], "score": round(h["score"], 3),
                           "coverage": round(h["coverage"], 2)} for h in hits],
            "injection_detected": injected,
            "reason": reason,
        }
