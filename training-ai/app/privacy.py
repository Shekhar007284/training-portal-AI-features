"""PII redaction + prompt-injection guard.

WHY regex + a known-names list (and not an NER model)?
- Deterministic, zero dependencies, easy to explain and unit-test.
- We already KNOW the sensitive strings: employee names come from our own CSV.
  Matching against that list is more reliable than guessing names with ML.
- Trade-off: names we have never seen ("my name is ...") are only caught by a
  simple pattern. In production add Microsoft Presidio / an NER model.
"""
from __future__ import annotations

import re
from typing import Iterable

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s\-]?)?(?:\d[\s\-]?){9,11}\d(?!\w)")
SELF_INTRO_RE = re.compile(
    r"\b((?i:my name is|i am|i'm|this is))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)


def redact(text: str, known_names: Iterable[str] = ()) -> str:
    """Replace emails, phone numbers and names with placeholders."""
    out = EMAIL_RE.sub("[EMAIL]", text)
    out = PHONE_RE.sub("[PHONE]", out)
    # Longest names first so "Aarav Mehta" is replaced before "Aarav".
    for name in sorted({n for n in known_names if n}, key=len, reverse=True):
        for part in [name] + name.split():
            if len(part) >= 3:
                out = re.sub(rf"\b{re.escape(part)}\b", "[NAME]", out, flags=re.I)
    out = SELF_INTRO_RE.sub(lambda m: f"{m.group(1)} [NAME]", out)
    return out


# --- Prompt-injection guard -------------------------------------------------
# Phrases that try to change the assistant's rules.
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+)?(?:your\s+|the\s+|my\s+)?(?:previous|prior|above|earlier)?\s*(?:instructions?|rules?|prompts?)",
    r"disregard\s+(?:all\s+|your\s+|the\s+)?(?:previous\s+|prior\s+)?(?:instructions?|rules?)",
    r"forget\s+(?:all\s+|your\s+|everything)(?:\s+(?:previous|prior))?(?:\s+instructions?|\s+rules?)?",
    r"you\s+are\s+now\b",
    r"act\s+as\s+(?:a|an)\b",
    r"pretend\s+(?:to\s+be|you)",
    r"(?:reveal|show|print)\s+(?:your\s+)?(?:system\s+)?prompt",
    r"developer\s+mode|jailbreak",
    r"override\s+(?:your\s+)?(?:rules|instructions|safety)",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)

# Requests for personal / employee data. The assistant has NO access to such
# data and is not allowed to produce it, so we refuse explicitly.
PERSONAL_DATA_RE = re.compile(
    r"(?:e-?mail|phone|salary|address|password|home address|personal data|records?)\b.{0,40}\b(?:employees?|staff|everyone|users?|people)"
    r"|(?:employees?|staff|everyone|users?|people)\b.{0,40}\b(?:e-?mail|phone|salary|address|password)"
    r"|list\s+(?:every|all)\s+(?:employees?|staff|users?)",
    re.I,
)


def split_injection(question: str) -> tuple[str, bool]:
    """Remove injection phrases. Return (remaining_question, injection_found)."""
    found = bool(INJECTION_RE.search(question))
    cleaned = INJECTION_RE.sub(" ", question)
    cleaned = re.sub(r"^\W*(?:and|then|now|,|\.)\W*", "", cleaned.strip(), flags=re.I)
    return cleaned.strip(), found


def asks_for_personal_data(text: str) -> bool:
    return bool(PERSONAL_DATA_RE.search(text))
