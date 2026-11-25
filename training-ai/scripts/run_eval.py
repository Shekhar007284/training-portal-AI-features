"""Run the 6 test questions through the assistant and print a results table.

Usage:  python scripts/run_eval.py
Checks (a) was the answer correct, (b) was the right source retrieved.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.policy_qa import PolicyAssistant  # noqa: E402


def main() -> int:
    qa = PolicyAssistant(ROOT / "data" / "policies")
    questions = json.loads((ROOT / "data" / "questions.json").read_text())
    rows, failures = [], 0
    for q in questions:
        r = qa.ask(q["question"])
        if q["should_answer"]:
            src_ok = q["expected_source"] in r["sources"]
            ans_ok = r["answered"] and all(s.lower() in r["answer"].lower() for s in q["expected_contains"])
        else:  # must refuse / say "not in the policies"
            src_ok = not r["sources"]
            ans_ok = not r["answered"]
        failures += not (src_ok and ans_ok)
        rows.append((q["id"], q["question"], ans_ok, src_ok, r["sources"] or ["-"], r["answer"], r["retrieved"][:1]))

    print(f"{'#':<2} {'Answer ok':<10} {'Source ok':<10} Question")
    for id_, qn, a, s, src, ans, top in rows:
        print(f"{id_:<2} {'PASS' if a else 'FAIL':<10} {'PASS' if s else 'FAIL':<10} {qn}")
        print(f"   sources={src}  top_hit={top}")
        print(f"   answer: {ans}")
    print(f"\n{len(rows) - failures}/{len(rows)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
