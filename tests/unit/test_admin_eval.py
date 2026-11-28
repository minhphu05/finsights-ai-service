import asyncio
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.integrations import key_store
from app.integrations.embeddings import online as embeddings
from app.integrations.rerank import Reranked
from app.main import app
from app.schemas.evals import EvalRequest, EvalRun
from app.storage.postgres import news_search
from evals.runners import admin_run, retrieval

client = TestClient(app)


class _Connection:
    async def close(self) -> None:
        pass


@pytest.fixture
def pgvector(monkeypatch, tmp_path) -> dict:
    """pgvector holding two bge-m3 chunks of October, and a fake embedding API."""
    seen: dict = {"models": set()}
    published = datetime(2026, 10, 15, tzinfo=UTC)
    chunks = [
        (1, "a1", "Tiêu đề: Giá vàng\n\ngiá vàng tăng", published),
        (2, "a2", "Tiêu đề: Lãi suất\n\nlãi suất giảm", published),
    ]

    async def connect():
        return _Connection()

    async def has_chunks(connection, model, start, end):
        return model == "@cf/baai/bge-m3" and start <= published < end

    async def load_chunks(connection, model, start, end):
        seen["models"].add(model)
        return chunks

    async def nearest_chunks(connection, vector, **kwargs):
        seen["period"] = (kwargs["start"], kwargs["end"])
        return [(1, 0.9), (2, 0.5)]

    async def embed(texts, input_type, *, provider):
        seen["embedded_with"] = provider.id
        seen["embed_calls"] = seen.get("embed_calls", 0) + 1
        return embeddings.Embeddings([[1.0, 0.0]], 1_000_000)

    for name, fake in [
        ("connect", connect),
        ("has_chunks", has_chunks),
        ("load_chunks", load_chunks),
        ("nearest_chunks", nearest_chunks),
    ]:
        monkeypatch.setattr(news_search, name, fake)
    monkeypatch.setattr(embeddings, "embed", embed)
    monkeypatch.setenv("AI_EVAL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        admin_run.get_settings().mysql_source_readonly_dsn, "_secret_value", ""
    )
    for name in ("cloudflare", "cloudflare_account_id", "voyage"):
        key_store.save(name, "test-key-1234")
    return seen


def _body(**changes) -> dict:
    return {
        "dataset": "set.jsonl",
        "embedding": "bge-m3",
        "cases": [
            {
                "question": "giá vàng",
                "date_from": "2026-10-01",
                "date_to": "2026-10-31",
                "relevant": ["a1"],
            }
        ],
    } | changes


def _finished(run_id: str) -> dict:
    for _ in range(200):
        run = client.get(f"/admin/evals/retrieval/{run_id}").json()
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_run_without_rerank_and_decision_runs_what_it_can(pgvector) -> None:
    started = client.post("/admin/evals/retrieval", json=_body())
    assert started.status_code == 202, started.text

    run = _finished(started.json()["id"])

    assert run["status"] == "done", run
    assert [r["id"] for r in run["results"]] == ["bm25", "dense", "hybrid"]
    dense = run["results"][1]
    assert dense["overall"]["mrr@10"] == 1.0
    assert dense["byCategory"]["general"]["cases"] == 1
    assert {s["id"] for s in run["skipped"]} == {
        "like",
        "rerank80",
        "hybrid-rerank80",
        "jev-point80",
        "hybrid-jev80",
        "gate10",
        "hybrid-gate10",
    }
    assert run["progress"] == {"done": 3, "total": 3}
    assert pgvector["models"] == {"@cf/baai/bge-m3"}
    assert pgvector["embedded_with"] == "bge-m3"
    # 1-31 October in Vietnam (UTC+7), end exclusive.
    assert pgvector["period"] == (
        datetime(2026, 9, 30, 17, tzinfo=UTC),
        datetime(2026, 10, 31, 17, tzinfo=UTC),
    )
    listed = client.get("/admin/evals/retrieval").json()["runs"]
    assert [r["id"] for r in listed] == [run["id"]]


def test_a_failing_strategy_does_not_stop_the_others(pgvector, monkeypatch) -> None:
    async def broken(*args, **kwargs):
        raise embeddings.EmbeddingError("Voyage API 400: unknown model")

    monkeypatch.setattr("app.integrations.rerank.rerank", broken)
    rerank = admin_run.options().reranks[0].id

    started = client.post("/admin/evals/retrieval", json=_body(rerank=rerank))
    run = _finished(started.json()["id"])

    assert run["status"] == "done"
    assert [r["id"] for r in run["results"]] == ["bm25", "dense", "hybrid"]
    assert {f["id"] for f in run["failed"]} == {"rerank80", "hybrid-rerank80"}
    assert "unknown model" in run["failed"][0]["reason"]


def test_a_model_without_vectors_is_rejected(pgvector) -> None:
    response = client.post("/admin/evals/retrieval", json=_body(embedding="voyage-4"))
    assert response.status_code == 409
    assert "voyage-4" in response.json()["detail"]
    assert client.get("/admin/evals/retrieval").json() == {"runs": []}


def test_a_period_without_vectors_is_rejected(pgvector) -> None:
    case = _body()["cases"][0] | {"date_from": "2026-01-01", "date_to": "2026-01-31"}
    response = client.post("/admin/evals/retrieval", json=_body(cases=[case]))
    assert response.status_code == 409
    assert "2026-01-01" in response.json()["detail"]


@pytest.mark.parametrize(
    "changes",
    [
        {"embedding": "nope"},
        {"decision": "jev-1.14.0"},  # no key saved
        {"cases": []},
        {"cases": [{"question": "q"}]},  # nothing to score against
    ],
)
def test_bad_input_is_rejected(pgvector, changes) -> None:
    response = client.post("/admin/evals/retrieval", json=_body(**changes))
    assert response.status_code == 422


def test_cost_counts_cached_calls_per_strategy_but_pays_them_once(
    pgvector, monkeypatch, tmp_path
) -> None:
    reranked: list[str] = []

    async def fake_rerank(query, documents, **kwargs):
        reranked.append(query)
        return Reranked([(i, 1.0 - i / 10) for i in range(len(documents))], 2_000_000)

    monkeypatch.setattr("app.integrations.rerank.rerank", fake_rerank)
    monkeypatch.setattr(retrieval, "SNAPSHOTS", tmp_path)
    monkeypatch.setattr(retrieval, "_rerank_cache", None)

    started = client.post(
        "/admin/evals/retrieval", json=_body(rerank="voyage-rerank-3")
    )
    run = _finished(started.json()["id"])

    by_id = {r["id"]: r for r in run["results"]}
    assert by_id["bm25"]["spend"] == {"cost_usd": 0, "estimated": False, "usage": []}
    # 1M embedding tokens at 0.012; both dense and hybrid need them.
    assert by_id["dense"]["spend"]["cost_usd"] == pytest.approx(0.012)
    assert by_id["hybrid"]["spend"]["cost_usd"] == pytest.approx(0.012)
    # Plus 2M rerank tokens at 0.05.
    assert by_id["rerank80"]["spend"]["cost_usd"] == pytest.approx(0.112)
    # Both rerank strategies saw the same chunks: one real call, counted in each.
    assert by_id["hybrid-rerank80"]["spend"]["cost_usd"] == pytest.approx(0.112)
    assert len(reranked) == 1 and pgvector["embed_calls"] == 1
    assert run["spend"]["cost_usd"] == pytest.approx(0.112)
    assert [u["calls"] for u in run["spend"]["usage"]] == [1, 1]
    assert run["seconds"] >= run["load_seconds"] >= 0
    assert all(r["seconds_per_case"] >= 0 for r in run["results"])


def test_spend_of_unpriced_and_self_hosted_models() -> None:
    calls = [retrieval.Call("local", 500, 0, 0.1), retrieval.Call("x", 10, 5, 0.1)]

    unknown = admin_run.spend(calls, {"local": (0, 0), "x": (1.0, None)})
    assert unknown.cost_usd is None
    assert [u.cost_usd for u in unknown.usage] == [0, None]

    # No output price is needed when nothing was billed as output.
    free = admin_run.spend(calls[:1], {"local": (0, None)})
    assert free.cost_usd == 0


@pytest.mark.anyio
@pytest.mark.parametrize("hangs", [1, 2])
async def test_a_hanging_like_query_is_cancelled_and_tried_once_more(
    monkeypatch, hangs
) -> None:
    started: list[str] = []
    cancelled: list[bool] = []

    async def like(case):
        started.append(case.id)
        if len(started) <= hangs:
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)
        return [("a1", "t", "x")]

    monkeypatch.setattr(retrieval, "like_retriever", like)
    monkeypatch.setattr(admin_run, "LIKE_TIMEOUT", 0.05)
    run = admin_run.like_on(asyncio.get_running_loop())
    case = admin_run._case(EvalRequest(**_body()).cases[0])

    if hangs == 1:
        assert await run(case) == [("a1", "t", "x")]
    else:
        with pytest.raises(TimeoutError):
            await run(case)
    await asyncio.sleep(0.01)  # let the cancellation reach the query

    assert len(started) == 2 and len(cancelled) == hangs


def test_history_lists_the_newest_runs_only(pgvector) -> None:
    ids = [f"{n:032x}" for n in range(admin_run.MAX_LISTED_RUNS + 1)]
    for n, run_id in enumerate(ids):
        admin_run.store.save(
            EvalRun(
                id=run_id,
                status="done",
                created_at=f"2026-11-01T00:00:{n:02d}+00:00",
                dataset="set.jsonl",
                embedding="bge-m3",
                rerank="none",
                decision="none",
            )
        )

    listed = client.get("/admin/evals/retrieval").json()["runs"]

    assert [r["id"] for r in listed] == ids[:0:-1]  # all but the oldest
    assert client.get(f"/admin/evals/retrieval/{ids[0]}").status_code == 200


def test_unknown_run_and_path_tricks(pgvector) -> None:
    assert client.get(f"/admin/evals/retrieval/{'0' * 32}").status_code == 404
    assert client.get("/admin/evals/retrieval/..%5Cprovider_keys").status_code == 422
