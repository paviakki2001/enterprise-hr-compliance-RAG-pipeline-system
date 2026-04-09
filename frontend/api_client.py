from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from config import settings


class APIClientError(Exception):
    pass


def health_check() -> Dict[str, Any]:
    url = f"{settings.BACKEND_BASE_URL.rstrip('/')}{settings.HEALTH_PATH}"
    try:
        response = requests.get(url, timeout=15)
        if not response.ok:
            return {"ok": False, "detail": response.text[:300]}
        return {"ok": True, "data": response.json()}
    except requests.RequestException as exc:
        return {"ok": False, "detail": str(exc)}


def ask_question(
    question: str,
    top_k_retrieval: int = 8,
    top_k_context: int = 4,
    filters: Optional[dict] = None,
) -> Dict[str, Any]:
    if not question or not question.strip():
        raise APIClientError("Question cannot be empty.")

    url = f"{settings.BACKEND_BASE_URL.rstrip('/')}{settings.BACKEND_QUERY_PATH}"
    payload = {
        "question": question.strip(),
        "top_k_retrieval": int(top_k_retrieval),
        "top_k_context": int(top_k_context),
        "filters": filters,
    }

    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=settings.REQUEST_TIMEOUT_SECONDS)
        elapsed = round(time.perf_counter() - started, 3)

        if not response.ok:
            raise APIClientError(f"Backend error {response.status_code}: {response.text[:400]}")

        body = response.json()
        return {
            "question": body.get("question", question),
            "answer": body.get("answer", ""),
            "citations": body.get("citations", []),
            "latency_seconds": body.get("latency_seconds", elapsed),
            "retrieved_count": body.get("retrieved_count", len(body.get("citations", []))),
            "raw": body,
        }

    except requests.RequestException as exc:
        raise APIClientError(f"Unable to reach backend API: {exc}") from exc
    except ValueError as exc:
        raise APIClientError("Backend returned invalid JSON.") from exc
