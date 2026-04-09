from __future__ import annotations

import time
from functools import wraps
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("week4_rag_requests_total", "Total RAG requests")
REQUEST_ERRORS = Counter("week4_rag_request_errors_total", "Total RAG request errors")
TOTAL_LATENCY = Histogram("week4_rag_total_latency_seconds", "Total end-to-end latency")
RETRIEVAL_LATENCY = Histogram("week4_rag_retrieval_latency_seconds", "Retrieval latency")
GENERATION_LATENCY = Histogram("week4_rag_generation_latency_seconds", "Generation latency")


def track_total_latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            TOTAL_LATENCY.observe(time.perf_counter() - started)
            return result
        except Exception:
            REQUEST_ERRORS.inc()
            TOTAL_LATENCY.observe(time.perf_counter() - started)
            raise
    return wrapper
