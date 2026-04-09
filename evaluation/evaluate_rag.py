from __future__ import annotations

import json
from pathlib import Path
from typing import List

import requests

BACKEND_BASE_URL = "http://localhost:8000"
QUERY_PATH = "/query"


def keyword_score(answer: str, expected_keywords: List[str]) -> float:
    text = (answer or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / max(len(expected_keywords), 1)


def run_evaluation():
    test_path = Path(__file__).parent / "sample_test_questions.json"
    test_cases = json.loads(test_path.read_text(encoding="utf-8"))

    rows = []
    for case in test_cases:
        payload = {
            "question": case["question"],
            "top_k_retrieval": 8,
            "top_k_context": 4,
            "filters": None
        }
        try:
            response = requests.post(
                f"{BACKEND_BASE_URL}{QUERY_PATH}",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            score = keyword_score(body.get("answer", ""), case["expected_keywords"])
            rows.append(
                {
                    "question": case["question"],
                    "latency_seconds": body.get("latency_seconds"),
                    "retrieved_count": body.get("retrieved_count"),
                    "citation_count": len(body.get("citations", [])),
                    "keyword_score": round(score, 2),
                    "status": "success",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "question": case["question"],
                    "latency_seconds": None,
                    "retrieved_count": 0,
                    "citation_count": 0,
                    "keyword_score": 0.0,
                    "status": f"failed: {exc}",
                }
            )

    out_path = Path(__file__).parent / "evaluation_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved results to: {out_path}")

    avg_score = sum(r["keyword_score"] for r in rows) / max(len(rows), 1)
    print("\nEvaluation Summary")
    print("-" * 40)
    print("Total test cases :", len(rows))
    print("Average score    :", round(avg_score, 2))


if __name__ == "__main__":
    run_evaluation()
