"""Voyage AI reranker over the REST API (docs.voyageai.com/reference/reranker-api).

Embeddings score a question and a text separately and compare the two vectors. A reranker
reads the question and each text together, which tells apart texts that are close in topic but
answer different questions ("lãi suất cho vay" against "lãi suất huy động"). It is slower and
costs per call, so it re-orders a short list that a vector search has already found.

Same rules as the embeddings client: plain httpx, the key only in the Authorization header and
never in a log or error message.
"""

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from app.integrations.embeddings.online import (
    MAX_BACKOFF_SECONDS,
    RETRIES,
    RETRY_STATUSES,
    EmbeddingError,
)
from app.integrations.providers import ProviderConfigError, get_provider

logger = logging.getLogger(__name__)

# API limit: at most 1,000 documents, and (query tokens x documents + document tokens) <= 600K.
MAX_DOCUMENTS = 200


@dataclass
class Reranked:
    # (index into the documents sent, relevance score), best first.
    ranked: list[tuple[int, float]]
    tokens: int


async def rerank(
    query: str,
    documents: Sequence[str],
    *,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Reranked:
    """Score `documents` against `query`; every document comes back, best first.

    `model` overrides the model of the default rerank provider (the eval compares models).
    """
    provider = get_provider("rerank")
    try:
        key = provider.api_key()
    except ProviderConfigError as exc:
        raise EmbeddingError(str(exc)) from None
    if not documents:
        return Reranked([], 0)
    if len(documents) > MAX_DOCUMENTS:
        raise ValueError(f"at most {MAX_DOCUMENTS} documents per call")
    payload = {
        "query": query,
        "documents": list(documents),
        "model": model or provider.model,
    }
    owned = client is None
    client = client or httpx.AsyncClient(timeout=60)
    try:
        return await _post(client, provider.url, key, payload)
    finally:
        if owned:
            await client.aclose()


async def _post(
    client: httpx.AsyncClient, url: str, key: str, payload: dict
) -> Reranked:
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(
                url, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
        except httpx.TransportError as exc:
            if attempt == RETRIES:
                raise EmbeddingError(
                    f"Voyage API unreachable: {type(exc).__name__}"
                ) from None
        else:
            if response.status_code == 200:
                body = response.json()
                ranked = [(d["index"], d["relevance_score"]) for d in body["data"]]
                ranked.sort(key=lambda pair: pair[1], reverse=True)
                return Reranked(ranked, body["usage"]["total_tokens"])
            if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                raise EmbeddingError(
                    f"Voyage API {response.status_code}: {response.text[:200]}"
                )
        wait = min(MAX_BACKOFF_SECONDS, 2**attempt) * (0.5 + random.random() / 2)
        logger.warning("Voyage API busy, retrying in %.1fs", wait)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")
