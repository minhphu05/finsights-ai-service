"""Text embeddings for the running service, with whichever model the admin picked.

The model comes from the `embedding` section of `configs/models.yaml` (`default`), and its
`dialect` says which API it speaks: `voyage` or `cloudflare` (Workers AI). Plain httpx, no SDK.
The API key only travels in the Authorization header and is never logged or put in an error
message. The Dagster ingestion job has its own synchronous client in `client.py`.

For Voyage, `input_type` matters for retrieval: it prepends a different instruction to queries
and to documents, so embed questions with "query" and the searched texts with "document".
Cloudflare models take no such hint.
"""

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.integrations.embeddings.client import (
    EmbeddingRequestError,
    reported_tokens,
    vectors_from_cloudflare,
)
from app.integrations.providers import ProviderConfig, ProviderConfigError, get_provider

logger = logging.getLogger(__name__)

InputType = Literal["query", "document"]

# API limits per request: at most 1,000 texts and 320K tokens for Voyage, 64 texts for the
# Workers AI models used here. Batches stay far below, so one slow or rate-limited request
# never holds much work.
MAX_TEXTS_PER_BATCH = 128
MAX_TEXTS_PER_BATCH_CLOUDFLARE = 64
MAX_TOKENS_PER_BATCH = 60_000
# No tokenizer here: Vietnamese runs about 3 characters per token, so count high.
CHARS_PER_TOKEN = 2.5
RETRIES = 6
MAX_BACKOFF_SECONDS = 60
RETRY_STATUSES = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """The embedding provider did not return embeddings."""


@dataclass
class Embeddings:
    vectors: list[list[float]]
    tokens: int


def batches(
    texts: Sequence[str], max_texts: int = MAX_TEXTS_PER_BATCH
) -> list[list[str]]:
    """Split texts into requests under the per-request text and (estimated) token limits."""
    result: list[list[str]] = [[]]
    budget = 0.0
    for text in texts:
        cost = len(text) / CHARS_PER_TOKEN + 1
        if result[-1] and (
            len(result[-1]) >= max_texts or budget + cost > MAX_TOKENS_PER_BATCH
        ):
            result.append([])
            budget = 0.0
        result[-1].append(text)
        budget += cost
    return [b for b in result if b]


def build_request(
    provider: ProviderConfig, texts: Sequence[str], input_type: InputType
) -> tuple[str, dict[str, Any]]:
    """The URL and JSON body that embed `texts` with `provider`, in its own dialect."""
    match provider.dialect:
        case "voyage":
            body: dict[str, Any] = {
                "input": list(texts),
                "model": provider.model,
                "input_type": input_type,
            }
            if provider.dimensions:
                body["output_dimension"] = provider.dimensions
            return provider.url, body
        case "cloudflare":
            url = provider.url.format(
                account_id=provider.account_id(), model=provider.model
            )
            return url, {"text": list(texts)}
    raise EmbeddingError(f"unsupported embedding dialect {provider.dialect!r}")


def parse_response(
    provider: ProviderConfig, payload: dict[str, Any]
) -> tuple[list[list[float]], int]:
    """The vectors, in input order, and the tokens the provider counted (0 if it gives none)."""
    try:
        if provider.dialect == "cloudflare":
            if payload.get("success") is False:
                raise EmbeddingError("Cloudflare Workers AI rejected the request")
            result = payload["result"]
            return vectors_from_cloudflare(result), reported_tokens(
                payload, result
            ) or 0
        data = sorted(payload["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data], payload["usage"]["total_tokens"]
    except (KeyError, TypeError, EmbeddingRequestError) as exc:
        raise EmbeddingError(
            f"unrecognized embedding response ({type(exc).__name__})"
        ) from None


async def _request(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    key: str,
    url: str,
    body: dict[str, Any],
) -> tuple[list[list[float]], int]:
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(
                url, json=body, headers={"Authorization": f"Bearer {key}"}
            )
        except httpx.TransportError as exc:
            if attempt == RETRIES:
                raise EmbeddingError(
                    f"Embedding API unreachable: {type(exc).__name__}"
                ) from None
        else:
            if response.status_code == 200:
                return parse_response(provider, response.json())
            if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                # Voyage error bodies carry a reason, never the key; others may echo input.
                reason = (
                    f": {response.text[:200]}" if provider.dialect == "voyage" else ""
                )
                raise EmbeddingError(f"Embedding API {response.status_code}{reason}")
        wait = min(MAX_BACKOFF_SECONDS, 2**attempt) * (0.5 + random.random() / 2)
        logger.warning("Embedding API busy, retrying in %.1fs", wait)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def embed(
    texts: Sequence[str],
    input_type: InputType,
    *,
    client: httpx.AsyncClient | None = None,
) -> Embeddings:
    """Embed `texts` in order, in as many requests as the limits need."""
    provider = get_provider("embedding")
    try:
        key = provider.api_key()
        # Built once to fail on a missing account id before any request.
        build_request(provider, [], input_type)
    except ProviderConfigError as exc:
        raise EmbeddingError(str(exc)) from None
    max_texts = (
        MAX_TEXTS_PER_BATCH_CLOUDFLARE
        if provider.dialect == "cloudflare"
        else MAX_TEXTS_PER_BATCH
    )
    owned = client is None
    client = client or httpx.AsyncClient(timeout=60)
    vectors: list[list[float]] = []
    tokens = 0
    try:
        for batch in batches(texts, max_texts):
            url, body = build_request(provider, batch, input_type)
            got, used = await _request(client, provider, key, url, body)
            if len(got) != len(batch):
                raise EmbeddingError(
                    f"Embedding API returned {len(got)} of {len(batch)} vectors"
                )
            vectors.extend(got)
            tokens += used
    finally:
        if owned:
            await client.aclose()
    return Embeddings(vectors, tokens)
