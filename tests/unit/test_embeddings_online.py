import json
from typing import Any

import httpx
import pytest

from app.integrations.embeddings import online as mod

KEY = "pa-secret-key-for-tests"


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)


async def _no_sleep(seconds: float) -> None:
    return None


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    # Out of order on purpose: results must follow `index`, not response order.
    data = [
        {"object": "embedding", "embedding": [float(i), 1.0], "index": i}
        for i in reversed(range(len(texts)))
    ]
    return httpx.Response(
        200, json={"data": data, "usage": {"total_tokens": 3 * len(texts)}}
    )


def test_batches_respect_text_and_token_limits() -> None:
    assert [len(b) for b in mod.batches(["a"] * 300)] == [128, 128, 44]
    long = "x" * int(mod.MAX_TOKENS_PER_BATCH * mod.CHARS_PER_TOKEN * 0.6)
    assert [len(b) for b in mod.batches([long, long, long])] == [1, 1, 1]
    assert mod.batches([]) == []


@pytest.mark.anyio
async def test_embeds_in_order_across_batches() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return _ok(request)

    result = await mod.embed(["t"] * 130, "document", client=_client(handler))

    assert len(result.vectors) == 130 and result.tokens == 390
    assert result.vectors[0] == [0.0, 1.0] and result.vectors[129] == [1.0, 1.0]
    assert [len(b["input"]) for b in seen] == [128, 2]
    assert seen[0]["input_type"] == "document" and seen[0]["model"] == "voyage-4"


@pytest.mark.anyio
async def test_rate_limit_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(429, json={"detail": "slow down"})
            if calls < 3
            else _ok(request)
        )

    result = await mod.embed(["q"], "query", client=_client(handler))
    assert calls == 3 and len(result.vectors) == 1


@pytest.mark.anyio
async def test_client_error_is_not_retried_and_hides_the_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"detail": "bad input"})

    with pytest.raises(mod.EmbeddingError) as err:
        await mod.embed(["q"], "query", client=_client(handler))
    assert calls == 1
    assert "400" in str(err.value) and KEY not in str(err.value)


@pytest.mark.anyio
async def test_missing_key_fails_before_any_request(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(mod.EmbeddingError, match="not set"):
        await mod.embed(["q"], "query", client=_client(handler))


def _cloudflare(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.integrations.providers import ProviderConfig

    provider = ProviderConfig(
        id="bge-m3",
        kind="embedding",
        dialect="cloudflare",
        model="@cf/baai/bge-m3",
        url="https://cf.test/accounts/{account_id}/ai/run/{model}",
        api_key_ref="env:CF_TOKEN",
        account_id_ref="env:CF_ACCOUNT",
    )
    monkeypatch.setenv("CF_TOKEN", KEY)
    monkeypatch.setenv("CF_ACCOUNT", "acc-1")
    monkeypatch.setattr(mod, "get_provider", lambda kind: provider)


@pytest.mark.anyio
async def test_cloudflare_dialect_uses_its_url_body_and_batch_size(monkeypatch) -> None:
    _cloudflare(monkeypatch)
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        vectors = [[float(i), 1.0] for i in range(len(body["text"]))]
        return httpx.Response(200, json={"success": True, "result": {"data": vectors}})

    result = await mod.embed(["t"] * 70, "document", client=_client(handler))

    assert len(result.vectors) == 70 and result.tokens == 0
    assert [len(b["text"]) for _, b in seen] == [64, 6]
    assert seen[0][0] == "https://cf.test/accounts/acc-1/ai/run/@cf/baai/bge-m3"
    assert "model" not in seen[0][1] and "input_type" not in seen[0][1]


@pytest.mark.anyio
async def test_cloudflare_missing_account_id_fails_before_any_request(
    monkeypatch,
) -> None:
    _cloudflare(monkeypatch)
    monkeypatch.setenv("CF_ACCOUNT", "")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(mod.EmbeddingError, match="CF_ACCOUNT is not set"):
        await mod.embed(["q"], "query", client=_client(handler))


def test_an_embedding_model_needs_a_dialect_and_cloudflare_an_account() -> None:
    from app.integrations.providers import ProviderConfig

    base: dict[str, Any] = {
        "id": "m",
        "kind": "embedding",
        "model": "m",
        "url": "u",
        "api_key_ref": "env:K",
    }
    with pytest.raises(ValueError, match="needs a dialect"):
        ProviderConfig(**base)
    with pytest.raises(ValueError, match="account_id_ref"):
        ProviderConfig(**base, dialect="cloudflare")
