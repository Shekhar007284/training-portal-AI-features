"""Tiny shared tokenizer used by the TF-IDF index.

WHY a custom tokenizer? scikit-learn's default treats "certificate" and
"certificates" as different words, which breaks retrieval on tiny documents
(only ~15 sentences). A light stemmer + a small synonym map fixes the most
common lexical gaps without any model download.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that carry no meaning for matching in this domain.
# "Question-intent" words ask for a quantity/time, they are not the TOPIC, so
# they must not count against a chunk in the coverage check (fix for Q3).
QUESTION_WORDS = {"quickly", "soon", "long", "many", "much", "often", "fast", "when", "how"}
STOP = set(ENGLISH_STOP_WORDS) | {"please", "tell"} | QUESTION_WORDS

WORD_RE = re.compile(r"[a-z0-9%]+")


def stem(word: str) -> str:
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4 and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokenize(text: str, synonyms: dict[str, str] | None = None) -> list[str]:
    tokens = []
    for raw in WORD_RE.findall(text.lower()):
        if raw in STOP or len(raw) < 2:
            continue
        t = stem(raw)
        if synonyms:
            t = synonyms.get(t, t)
        tokens.append(t)
    return tokens
