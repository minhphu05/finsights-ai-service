"""Provider health for the admin page: embeddings, rerankers, Jev and the LLM profiles.

The providers come from `configs/models.yaml` (the embedding, rerank and llm sections), so a new
backend shows up here without a code change. Jev has its own module and is probed through it.
Each embedding and rerank provider gets one tiny real request (a fraction of a cent) with a
single attempt, no retries; Jev goes through its client (with its retries, capped by the
probe timeout). The LLM profiles
are only checked for a model name and an API key, unless `?live=true`, which also sends a
one-word prompt to the cheap tier of every configured profile.
"""

import asyncio
import time
from typing import Any, Literal

import httpx
from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.api.routes.keys import JEV_KEY
from app.integrations import key_store
from app.integrations.embeddings.online import build_request
from app.integrations.jev import client as jev
from app.integrations.llm.catalog import TIERS, Catalog, load_catalog
from app.integrations.llm.config import ModelConfigError, resolve_api_key
from app.integrations.llm.factory import get_chat_model
from app.integrations.providers import (
    KINDS,
    Kind,
    ProviderConfig,
    ProviderConfigError,
    get_registry,
)

router = APIRouter(tags=["health"])

PROBE_TIMEOUT_SECONDS = 15
Status = Literal["ok", "error", "not_configured"]


class ProviderHealth(BaseModel):
    id: str
    status: Status
    model: str
    # The one the service uses for its kind
    default: bool
    # Names of the saved keys it needs (see /admin/keys)
    keys: list[str]
    latency_ms: int | None = None
    # A short reason; never a key, a response body or submitted text.
    detail: str | None = None


class LlmProfileHealth(BaseModel):
    status: Status
    # tier -> model name (a tier is missing when it cannot be resolved)
    models: dict[str, str]
    keys: list[str]
    latency_ms: int | None = None
    detail: str | None = None


def _names(*refs: str | None) -> list[str]:
    return list(dict.fromkeys(ref.partition(":")[2] for ref in refs if ref))


class ProvidersHealth(BaseModel):
    status: Status
    embedding: list[ProviderHealth]
    rerank: list[ProviderHealth]
    jev: ProviderHealth
    llm: dict[str, LlmProfileHealth]


def _request(provider: ProviderConfig) -> tuple[str, dict[str, Any]]:
    """The URL and the smallest valid body of each kind."""
    if provider.kind == "embedding":
        return build_request(provider, ["ping"], "query")
    return provider.url, {
        "query": "ping",
        "documents": ["pong"],
        "model": provider.model,
    }


async def _probe(
    provider: ProviderConfig, default: bool, client: httpx.AsyncClient
) -> ProviderHealth:
    def result(status: Status, **extra: Any) -> ProviderHealth:
        return ProviderHealth(
            id=provider.id,
            status=status,
            model=provider.model,
            default=default,
            keys=_names(provider.api_key_ref, provider.account_id_ref),
            **extra,
        )

    try:
        key = provider.api_key()
        url, body = _request(provider)
    except ProviderConfigError as exc:
        return result("not_configured", detail=str(exc))
    start = time.perf_counter()
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            response = await client.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {key}"},
            )
    except TimeoutError:
        return result("error", detail="timed out")
    except httpx.HTTPError as exc:
        return result("error", detail=f"unreachable: {type(exc).__name__}")
    if response.status_code != 200:
        # Not the body: providers may echo what was sent.
        return result("error", detail=f"HTTP {response.status_code}")
    return result("ok", latency_ms=round((time.perf_counter() - start) * 1000))


async def _probe_jev(client: httpx.AsyncClient) -> ProviderHealth:
    """Jev has its own module (key in settings, URL and model in the client): ask it."""

    def result(status: Status, **extra: Any) -> ProviderHealth:
        return ProviderHealth(
            id="jev",
            status=status,
            model=jev.MODEL,
            default=True,
            keys=[JEV_KEY],
            **extra,
        )

    try:
        key_store.resolve(f"key:{JEV_KEY}")
    except key_store.KeyStoreError as exc:
        return result("not_configured", detail=str(exc))
    start = time.perf_counter()
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            await jev.system_one(
                client,
                {"question": "ping"},
                {"ok": {"type": "noul", "instructions": "Is this a test?"}},
            )
    except TimeoutError:
        return result("error", detail="timed out")
    except jev.JevError as exc:
        return result("error", detail=str(exc)[:200])
    return result("ok", latency_ms=round((time.perf_counter() - start) * 1000))


async def _llm_profile(profile: str, catalog: Catalog, live: bool) -> LlmProfileHealth:
    models: dict[str, str] = {}
    used = set(catalog.profiles[profile].values())
    keys = _names(*(m.api_key_ref for m in catalog.models if m.id in used))
    try:
        for tier in TIERS:
            config = catalog.select(profile, tier)
            models[tier] = config.model
            resolve_api_key(config.api_key_ref)
    except ModelConfigError as exc:
        return LlmProfileHealth(
            status="not_configured", models=models, keys=keys, detail=str(exc)
        )
    if not live:
        return LlmProfileHealth(status="ok", models=models, keys=keys)
    start = time.perf_counter()
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            await get_chat_model(profile, "cheap").ainvoke(
                [HumanMessage(content="ping")]
            )
    except TimeoutError:
        return LlmProfileHealth(
            status="error", models=models, keys=keys, detail="timed out"
        )
    except Exception as exc:  # noqa: BLE001 - provider errors may carry credentials
        return LlmProfileHealth(
            status="error", models=models, keys=keys, detail=type(exc).__name__
        )
    return LlmProfileHealth(
        status="ok",
        models=models,
        keys=keys,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )


@router.get("/health/providers", response_model=ProvidersHealth)
async def providers_health(live: bool = False) -> ProvidersHealth:
    registry = get_registry()
    catalog = load_catalog()
    async with httpx.AsyncClient() as client:
        probed, jev_health, profiles = await asyncio.gather(
            asyncio.gather(
                *(
                    _probe(p, registry.defaults[p.kind] == p.id, client)
                    for p in registry.providers
                )
            ),
            _probe_jev(client),
            asyncio.gather(*(_llm_profile(p, catalog, live) for p in catalog.profiles)),
        )
    by_kind: dict[Kind, list[ProviderHealth]] = {kind: [] for kind in KINDS}
    for provider, health in zip(registry.providers, probed, strict=True):
        by_kind[provider.kind].append(health)
    llm = dict(zip(catalog.profiles, profiles, strict=True))

    # What the service needs: the default provider of each kind, Jev and one ready LLM profile.
    needed = [h.status for hs in by_kind.values() for h in hs if h.default]
    needed.append(jev_health.status)
    needed.append(
        "ok" if any(p.status == "ok" for p in llm.values()) else "not_configured"
    )
    failed = any(h.status == "error" for hs in by_kind.values() for h in hs)
    failed = failed or jev_health.status == "error"
    failed = failed or any(p.status == "error" for p in llm.values())
    overall: Status = (
        "error"
        if failed
        else "ok"
        if all(s == "ok" for s in needed)
        else "not_configured"
    )
    return ProvidersHealth(status=overall, jev=jev_health, llm=llm, **by_kind)
