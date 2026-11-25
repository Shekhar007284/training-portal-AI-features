"""Course recommender: TF-IDF + cosine similarity.

WHY TF-IDF and not embeddings?
- 30 short course descriptions: TF-IDF is accurate enough, runs in milliseconds,
  needs no API key, no GPU, no model download, and is fully deterministic
  (same input -> same output, which matters for audits in a training portal).
- Explainable: I can show WHICH words matched, which gives the one-line reason.
- Embeddings are the upgrade path when skills are phrased very differently from
  course text ("keep customer info safe" vs "data privacy"). The class has one
  seam (`_vectorize`) so swapping is a small change.

The recommender is ADVISORY ONLY: it returns suggestions and never assigns,
enrols, ranks or judges an employee.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_utils import tokenize


class UnknownEmployee(Exception):
    pass


@dataclass
class Course:
    id: str
    title: str
    description: str
    skills: list[str]


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").split())


class Recommender:
    def __init__(self, courses_csv: Path, employees_csv: Path):
        self.courses: list[Course] = []
        with open(courses_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.courses.append(
                    Course(r["id"], r["title"], r["description"],
                           [s.strip() for s in r["skills"].split(";") if s.strip()])
                )
        self.employees: dict[str, dict] = {}
        with open(employees_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.employees[r["employee_id"]] = {
                    "name": r["name"],
                    "missing_skills": [s.strip() for s in r["missing_skills"].split(";") if s.strip()],
                }
        # Skills are repeated in the indexed text so a match on a *tagged skill*
        # outweighs a casual word in the description.
        docs = [f"{c.title}. {c.description}. " + (" ".join(c.skills) + " ") * 2 for c in self.courses]
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                                          token_pattern=None, ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)

    # ---- public helpers -------------------------------------------------
    def employee_names(self) -> list[str]:
        return [e["name"] for e in self.employees.values()]

    def skills_for(self, employee_id: str) -> list[str]:
        if employee_id not in self.employees:
            raise UnknownEmployee(employee_id)
        return self.employees[employee_id]["missing_skills"]

    # ---- core -----------------------------------------------------------
    def recommend(self, skills: list[str], k: int = 3) -> list[dict]:
        skills = [s for s in (x.strip() for x in skills) if s]
        if not skills:
            return []
        # One query vector per skill. A course's score = how well it fits its
        # BEST-matching gap (a cosine in 0..1, easy to explain). Total fit across
        # all gaps breaks ties, so a course covering two gaps beats one covering one.
        skill_vecs = self.vectorizer.transform(skills)              # (n_skills, vocab)
        per_skill = (skill_vecs @ self.matrix.T).toarray()          # (n_skills, n_courses)
        scores = per_skill.max(axis=0) + 0.01 * per_skill.sum(axis=0)
        order = np.argsort(-scores)[:k]
        out = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            course = self.courses[idx]
            best_skill = skills[int(np.argmax(per_skill[:, idx]))]
            out.append({
                "course_id": course.id,
                "title": course.title,
                "score": round(float(per_skill[:, idx].max()), 2),
                "reason": self._reason(course, best_skill),
            })
        return out

    @staticmethod
    def _reason(course: Course, skill: str) -> str:
        tagged = {_norm(s) for s in course.skills}
        if _norm(skill) in tagged:
            return f"Covers the missing skill: {skill}"
        return f"Closest match for the missing skill '{skill}' (related to: {', '.join(course.skills)})"
