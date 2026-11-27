import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.integrations import key_store
from app.integrations.embeddings import online as embeddings
from app.main import app
from app.storage.postgres import news_search
from evals.runners import admin_run

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
        return embeddings.Embeddings([[1.0, 0.0]], 1)

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


def test_unknown_run_and_path_tricks(pgvector) -> None:
    assert client.get(f"/admin/evals/retrieval/{'0' * 32}").status_code == 404
    assert client.get("/admin/evals/retrieval/..%5Cprovider_keys").status_code == 422
