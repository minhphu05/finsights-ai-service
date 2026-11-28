"""The benchmark started from the admin page: its question set, models and every strategy.

Vector search runs in pgvector over the chunks the ingestion job already embedded, so only
questions are embedded. BM25, rank fusion, rerank, Jev and the metrics are those of
evals.runners.retrieval.

Runs execute one at a time in a background thread and are kept as JSON files in
`data/eval_runs` (override with AI_EVAL_RUNS_DIR). This holds for one process: more replicas
need a shared job table instead of `RunStore` and `submit`.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.settings import get_settings
from app.integrations import key_store
from app.integrations.embeddings import online as embeddings
from app.integrations.jev import client as jev
from app.integrations.providers import (
    ProviderConfig,
    ProviderConfigError,
    get_registry,
    read_models_file,
)
from app.rag.retrieval.bm25 import Bm25Index
from app.schemas.evals import (
    NONE,
    EvalOptions,
    EvalRequest,
    EvalRun,
    ModelUsage,
    NotRun,
    Option,
    Progress,
    Spend,
    StrategyResult,
)
from app.storage.postgres import news_search
from app.tools.sql.search_news import vn_day_start_utc
from evals.runners import retrieval

logger = logging.getLogger(__name__)

# Chunks taken from pgvector per question: enough for the rerank pool, the fusion depth and
# TOP_K articles even when long articles hold many of the closest chunks.
DENSE_DEPTH = 400
PER_ARTICLE = 2
# Runs the history lists: the gateway cuts a response at 1 MB, and a run takes about 15 KB.
MAX_LISTED_RUNS = 50
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_runs"


@dataclass(frozen=True)
class Strategy:
    id: str
    method: str  # like | dense | bm25 | hybrid
    pool: int = 0
    reranker: str = "voyage"
    gate: float = 0.0

    @property
    def needs_rerank(self) -> bool:
        return self.pool > 0 and self.reranker == "voyage"

    @property
    def needs_decision(self) -> bool:
        return self.reranker != "voyage" or self.gate > 0


# The ids the admin page lists.
STRATEGIES = (
    Strategy("like", "like"),
    Strategy("bm25", "bm25"),
    Strategy("dense", "dense"),
    Strategy("hybrid", "hybrid"),
    Strategy("rerank80", "dense", 80),
    Strategy("hybrid-rerank80", "hybrid", 80),
    Strategy("jev-point80", "dense", 80, "jev-point"),
    Strategy("hybrid-jev80", "hybrid", 80, "jev-point"),
    Strategy("gate10", "dense", 80, gate=0.1),
    Strategy("hybrid-gate10", "hybrid", 80, gate=0.1),
)


class Rejected(Exception):
    """The run cannot start with these inputs; the message is shown to the admin."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Models:
    embedding: ProviderConfig
    rerank: ProviderConfig | None
    decision: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _period(date_from: date, date_to: date) -> tuple[datetime, datetime]:
    """The Vietnam days from `date_from` to `date_to` as [start, end) in UTC."""
    return (
        vn_day_start_utc(date_from).replace(tzinfo=UTC),
        vn_day_start_utc(date_to + timedelta(days=1)).replace(tzinfo=UTC),
    )


def _configured(provider: ProviderConfig) -> bool:
    try:
        provider.api_key()
        if provider.account_id_ref:
            provider.account_id()
    except ProviderConfigError:
        return False
    return True


def _jev_configured() -> bool:
    try:
        key_store.resolve("key:typesafe")
    except key_store.KeyStoreError:
        return False
    return True


def options() -> EvalOptions:
    """The models an admin can pick."""
    providers = get_registry().providers
    by_kind = {
        kind: [
            Option(id=p.id, configured=_configured(p))
            for p in providers
            if p.kind == kind
        ]
        for kind in ("embedding", "rerank")
    }
    return EvalOptions(
        embeddings=by_kind["embedding"],
        reranks=by_kind["rerank"],
        decisions=[Option(id=jev.MODEL, configured=_jev_configured())],
    )


def choose_models(request: EvalRequest) -> Models:
    """The chosen models, rejected when unknown or without a saved key."""

    def pick(kind: str, wanted: str) -> ProviderConfig:
        found = next(
            (p for p in get_registry().providers if p.kind == kind and p.id == wanted),
            None,
        )
        if found is None:
            raise Rejected(422, f"Không có model {kind} '{wanted}'.")
        if not _configured(found):
            raise Rejected(422, f"Chưa lưu API key cho model '{wanted}'.")
        return found

    embedding = pick("embedding", request.embedding)
    rerank = None if request.rerank == NONE else pick("rerank", request.rerank)
    decision = None
    if request.decision != NONE:
        if request.decision != jev.MODEL:
            raise Rejected(422, f"Không có model quyết định '{request.decision}'.")
        if not _jev_configured():
            raise Rejected(422, "Chưa lưu API key cho model quyết định.")
        decision = request.decision
    return Models(embedding, rerank, decision)


async def check_vectors(request: EvalRequest, model: ProviderConfig) -> None:
    """Reject the run when pgvector has nothing of this model for a question's period."""
    try:
        connection = await news_search.connect()
        try:
            missing = [
                (first, last)
                for first, last in sorted(
                    {(c.date_from, c.date_to) for c in request.cases}
                )
                if not await news_search.has_chunks(
                    connection, model.model, *_period(first, last)
                )
            ]
        finally:
            await connection.close()
    except Exception:
        # The message of a connection error can hold the DSN: log it, do not return it.
        logger.exception("pgvector check failed")
        raise Rejected(503, "Không đọc được pgvector.") from None
    if missing:
        periods = ", ".join(f"{first} đến {last}" for first, last in missing[:5])
        raise Rejected(
            409,
            f"pgvector chưa có dữ liệu của model '{model.id}' cho khoảng: {periods}.",
        )


Price = tuple[float | None, float | None]  # USD per million (input, output) tokens


def prices(models: Models) -> dict[str, Price]:
    """The price of each chosen model, under the name its calls carry."""
    chosen = [p for p in (models.embedding, models.rerank) if p is not None]
    found = {p.model: (p.price_input_per_mtok, p.price_output_per_mtok) for p in chosen}
    if models.decision:
        listed = read_models_file().get("decision", {}).get("models", [])
        entry = next((m for m in listed if m.get("id") == models.decision), {})
        found[models.decision] = (
            entry.get("price_input_per_mtok"),
            entry.get("price_output_per_mtok"),
        )
    return found


def spend(calls: Sequence[retrieval.Call], price: dict[str, Price]) -> Spend:
    """Tokens and cost of `calls`, per model."""
    by_model: dict[str, list[retrieval.Call]] = defaultdict(list)
    for call in calls:
        by_model[call.model].append(call)
    usage = []
    for model, found in sorted(by_model.items()):
        used = (
            sum(c.input_tokens for c in found),
            sum(c.output_tokens for c in found),
        )
        rates = price.get(model, (None, None))
        # A missing price matters only for tokens actually used on that side.
        unknown = any(n and rate is None for n, rate in zip(used, rates))
        usage.append(
            ModelUsage(
                model=model,
                calls=len(found),
                input_tokens=used[0],
                output_tokens=used[1],
                estimated=any(c.estimated for c in found),
                cost_usd=None
                if unknown
                else sum(n * (rate or 0) for n, rate in zip(used, rates)) / 1e6,
            )
        )
    costs = [u.cost_usd for u in usage]
    return Spend(
        cost_usd=None if None in costs else sum(c or 0 for c in costs),
        estimated=any(u.estimated for u in usage),
        usage=usage,
    )


class PgVectorIndex(retrieval.DenseIndex):
    """The stored chunks of one model: texts in memory for BM25, vectors in pgvector."""

    def __init__(self, connection: Any, provider: ProviderConfig, cases: list) -> None:
        super().__init__(
            min(c.date_from for c in cases), max(c.date_to for c in cases), "chunks"
        )
        self.connection, self.provider = connection, provider
        self._item: dict[int, int] = {}  # chunk id -> position in `texts`
        self._found: dict[tuple, tuple[dict[int, float], retrieval.Call]] = {}

    async def load(self) -> None:
        rows = await news_search.load_chunks(
            self.connection,
            self.provider.model,
            *_period(self.date_from, self.date_to),
        )
        articles: dict[str, int] = {}
        for chunk_id, uid, text, published_at in rows:
            if uid not in articles:
                articles[uid] = len(self.uids)
                self.uids.append(uid)
                # Each stored chunk starts with a "Tiêu đề: <title>" line.
                self.titles.append(text.partition("\n")[0].removeprefix("Tiêu đề: "))
                # Naive UTC, as DenseIndex compares it.
                self.published.append(published_at.astimezone(UTC).replace(tzinfo=None))
            self._item[chunk_id] = len(self.texts)
            self.owner.append(articles[uid])
            self.texts.append(text)

    async def _by_vector(
        self, case: retrieval.Case, in_range: Callable[[int], bool]
    ) -> dict[int, float]:
        key = (case.question, case.date_from, case.date_to)
        if key not in self._found:
            started = time.perf_counter()
            embedded = await embeddings.embed(
                [case.question], "query", provider=self.provider
            )
            start, end = _period(case.date_from, case.date_to)
            rows = await news_search.nearest_chunks(
                self.connection,
                embedded.vectors[0],
                model=self.provider.model,
                start=start,
                end=end,
                limit=DENSE_DEPTH,
            )
            # A chunk added after `load` is left out: the run keeps one corpus.
            found = {self._item[i]: score for i, score in rows if i in self._item}
            call = retrieval.Call(
                self.provider.model,
                embedded.tokens
                or int(len(case.question) / embeddings.CHARS_PER_TOKEN) + 1,
                0,
                # The question's embedding and its vector search.
                time.perf_counter() - started,
                estimated=not embedded.tokens,
            )
            self._found[key] = (found, call)
        else:
            found, call = self._found[key]
            call = replace(call, cached=True)
        if self.calls is not None:
            self.calls.append(call)
        return found


def skip_reason(strategy: Strategy, models: Models, has_mysql: bool) -> str | None:
    if strategy.method == "like" and not has_mysql:
        return "Chưa cấu hình MySQL."
    if strategy.needs_rerank and models.rerank is None:
        return "Không chọn model rerank."
    if strategy.needs_decision and models.decision is None:
        return "Không chọn model quyết định."
    return None


class RunStore:
    """One JSON file per run."""

    def __init__(self) -> None:
        self.active: set[str] = set()  # queued or running in this process
        # Windows cannot replace a file while a request is reading it.
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return Path(os.environ.get("AI_EVAL_RUNS_DIR") or _DEFAULT_DIR)

    def save(self, run: EvalRun) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(run.model_dump_json())
            with self._lock:
                os.replace(temporary, self.directory / f"{run.id}.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    def get(self, run_id: str) -> EvalRun | None:
        try:
            with self._lock:
                path = self.directory / f"{run_id}.json"
                text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        run = EvalRun.model_validate_json(text)
        # Left unfinished by a process that stopped.
        if run.status in ("queued", "running") and run.id not in self.active:
            run.status = "interrupted"
        return run

    def list(self) -> list[EvalRun]:
        """The newest runs first; older ones stay readable by id."""
        found = (self.get(path.stem) for path in self.directory.glob("*.json"))
        newest = sorted(
            (run for run in found if run), key=lambda r: r.created_at, reverse=True
        )
        return newest[:MAX_LISTED_RUNS]


store = RunStore()
# One run at a time: it holds the corpus in memory and calls rate-limited APIs.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval")


def _case(case: Any) -> retrieval.Case:
    return retrieval.Case(
        id=case.id,
        category=case.category,
        question=case.question,
        query=case.query or case.question[:200],
        date_from=case.date_from,
        date_to=case.date_to,
        relevant=frozenset(case.relevant),
        irrelevant=frozenset(case.irrelevant),
        evidence=tuple(case.evidence),
    )


def _reason(exc: Exception) -> str:
    # These carry the API's status and reason, never a key; others may hold a DSN.
    if isinstance(exc, (embeddings.EmbeddingError, jev.JevError)):
        return str(exc)
    return f"Lỗi {type(exc).__name__}, xem log của AI service."


async def execute(
    run: EvalRun,
    cases: list[retrieval.Case],
    models: Models,
    like: retrieval.Retriever | None,
) -> None:
    """Run every possible strategy; one that fails does not stop the others."""
    runnable = []
    for strategy in STRATEGIES:
        reason = skip_reason(strategy, models, like is not None)
        if reason:
            run.skipped.append(NotRun(id=strategy.id, reason=reason))
        else:
            runnable.append(strategy)
    run.status = "running"
    run.progress = Progress(total=len(runnable) * len(cases))
    store.save(run)

    price = prices(models)
    paid: list[retrieval.Call] = []  # the calls really made
    began = time.perf_counter()
    connection = await news_search.connect()
    try:
        index = PgVectorIndex(connection, models.embedding, cases)
        if models.rerank:
            index.rerank_model = models.rerank.model
        await index.load()
        # Built here so that no strategy's time includes it.
        if any(s.method in ("bm25", "hybrid") for s in runnable):
            index.bm25 = Bm25Index(index.texts)
        run.load_seconds = round(time.perf_counter() - began, 3)
        for number, strategy in enumerate(runnable):
            scores: list[retrieval.Score] = []
            needed: list[retrieval.Call] = []
            seconds = 0.0
            try:
                for case in cases:
                    index.calls = []
                    started = time.perf_counter()
                    if strategy.method == "like":
                        assert like is not None
                        ranked = await like(case)
                    else:
                        ranked = await index.search(
                            case,
                            PER_ARTICLE,
                            strategy.pool,
                            strategy.method,
                            strategy.reranker,
                            strategy.gate,
                        )
                    # Alone, the strategy would have waited for the cached calls too.
                    seconds += time.perf_counter() - started
                    seconds += sum(c.seconds for c in index.calls if c.cached)
                    needed.extend(index.calls)
                    paid.extend(c for c in index.calls if not c.cached)
                    index.calls = []
                    scores.append(retrieval.score(case, ranked))
                    run.progress.done += 1
                    store.save(run)
            except Exception as exc:
                # Calls made before the error are paid for all the same.
                paid.extend(c for c in index.calls or [] if not c.cached)
                logger.exception("strategy %s failed", strategy.id)
                run.failed.append(NotRun(id=strategy.id, reason=_reason(exc)))
                run.progress.done = (number + 1) * len(cases)
            else:
                by_category: dict[str, list[retrieval.Score]] = defaultdict(list)
                for case, scored in zip(cases, scores, strict=True):
                    by_category[case.category].append(scored)
                run.results.append(
                    StrategyResult(
                        id=strategy.id,
                        overall=retrieval.summarize(scores),
                        byCategory={
                            name: retrieval.summarize(found)
                            for name, found in sorted(by_category.items())
                        },
                        seconds=round(seconds, 3),
                        seconds_per_case=round(seconds / len(cases), 3),
                        spend=spend(needed, price),
                    )
                )
            store.save(run)
    finally:
        run.seconds = round(time.perf_counter() - began, 3)
        run.spend = spend(paid, price)
        await connection.close()


def _work(
    run: EvalRun,
    cases: list[retrieval.Case],
    models: Models,
    main_loop: asyncio.AbstractEventLoop,
) -> None:
    like = None
    if get_settings().mysql_source_readonly_dsn.get_secret_value():
        # The MySQL pool belongs to the server's event loop, so `like` queries run there.
        async def like(case: retrieval.Case) -> list[retrieval.Hit]:
            return await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    retrieval.like_retriever(case), main_loop
                )
            )

    try:
        run_in_selector_loop(execute(run, cases, models, like))
        run.status = "done" if run.results else "failed"
        if not run.results:
            run.error = "Không chiến lược nào chạy được."
    except Exception as exc:
        logger.exception("eval run %s failed", run.id)
        run.status, run.error = "failed", _reason(exc)
    finally:
        run.finished_at = _now()
        try:
            store.save(run)
        finally:
            store.active.discard(run.id)


def run_in_selector_loop(coroutine: Any) -> Any:
    """psycopg cannot use the Proactor loop Windows picks by default."""
    return asyncio.run(coroutine, loop_factory=asyncio.SelectorEventLoop)


def submit(
    run_id: str,
    request: EvalRequest,
    models: Models,
    main_loop: asyncio.AbstractEventLoop,
) -> EvalRun:
    """Record the run as queued and start it when the runs before it are done."""
    run = EvalRun(
        id=run_id,
        status="queued",
        created_at=_now(),
        dataset=request.dataset,
        embedding=request.embedding,
        rerank=request.rerank,
        decision=request.decision,
    )
    store.active.add(run.id)
    store.save(run)
    queued = run.model_copy(deep=True)
    _executor.submit(_work, run, [_case(c) for c in request.cases], models, main_loop)
    return queued
