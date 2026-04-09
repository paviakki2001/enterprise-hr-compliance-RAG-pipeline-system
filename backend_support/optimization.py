from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List


@lru_cache(maxsize=256)
def cached_prompt(question: str, context_block: str) -> str:
    return f'''
You are a retrieval-grounded HR and compliance assistant.

Rules:
- Answer only from the provided context.
- Do not use outside knowledge.
- If the context is insufficient, say so clearly.
- Keep the answer concise and faithful to the retrieved text.

Context:
{context_block}

Question:
{question}

Answer:
'''.strip()


def trim_context_texts(texts: List[str], max_chars: int = 5000) -> List[str]:
    kept = []
    total = 0
    for txt in texts:
        txt = (txt or "").strip()
        if not txt:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = txt[:remaining]
        kept.append(snippet)
        total += len(snippet)
    return kept


def normalize_filters(filters: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not filters:
        return None
    cleaned = {str(k): v for k, v in filters.items() if str(v).strip()}
    return cleaned or None
