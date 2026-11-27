"""Read-only access to the news chunk table filled by the ingestion job, for evaluation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from typing import LiteralString

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from sqlalchemy.engine import make_url

_PERIOD = """
    FROM public.news_article_chunks
    WHERE embedding_model = %(model)s
      AND published_at >= %(start)s AND published_at < %(end)s
"""

HAS_CHUNKS_SQL = f"SELECT EXISTS (SELECT 1 {_PERIOD})"

LOAD_SQL = f"SELECT id, article_uid, text, published_at {_PERIOD} ORDER BY id"

# `+ 0` keeps the HNSW index out on purpose, so every chunk of the period is compared: an
# index scan with a date filter can return fewer rows than asked and distort an evaluation.
NEAREST_SQL = f"""
    SELECT id, 1 - (embedding <=> %(vector)s) AS similarity {_PERIOD}
    ORDER BY (embedding <=> %(vector)s) + 0, id
    LIMIT %(limit)s
"""


async def connect(dsn: str | None = None) -> AsyncConnection:
    """A read-only connection from POSTGRES_RAG_DSN and POSTGRES_EMBEDDINGS_PASSWORD.

    psycopg needs a selector event loop, which is not the Windows default.
    """
    raw_dsn = dsn or os.getenv("POSTGRES_RAG_DSN", "").strip()
    if not raw_dsn:
        raise RuntimeError("POSTGRES_RAG_DSN is required for vector search")
    url = make_url(raw_dsn)
    password = os.getenv("POSTGRES_EMBEDDINGS_PASSWORD", "")
    if password and not url.password:
        url = url.set(password=password)
    url = url.set(drivername="postgresql")
    connection = await AsyncConnection.connect(
        url.render_as_string(hide_password=False),
        autocommit=True,
        connect_timeout=10,
    )
    await connection.set_read_only(True)
    await register_vector_async(connection)
    return connection


async def _fetch(
    connection: AsyncConnection, query: LiteralString, params: dict
) -> list[tuple]:
    async with connection.cursor() as cursor:
        await cursor.execute(query, params)
        return await cursor.fetchall()


async def has_chunks(
    connection: AsyncConnection, model: str, start: datetime, end: datetime
) -> bool:
    """Whether `model` has any vector for articles published in [start, end)."""
    params = {"model": model, "start": start, "end": end}
    return bool((await _fetch(connection, HAS_CHUNKS_SQL, params))[0][0])


async def load_chunks(
    connection: AsyncConnection, model: str, start: datetime, end: datetime
) -> list[tuple[int, str, str, datetime]]:
    """(id, article_uid, text, published_at) of `model`'s chunks in [start, end)."""
    params = {"model": model, "start": start, "end": end}
    return await _fetch(connection, LOAD_SQL, params)


async def nearest_chunks(
    connection: AsyncConnection,
    vector: Sequence[float],
    *,
    model: str,
    start: datetime,
    end: datetime,
    limit: int,
) -> list[tuple[int, float]]:
    """(id, cosine similarity) of the chunks in [start, end) closest to `vector`."""
    params = {
        "vector": Vector(list(vector)),
        "model": model,
        "start": start,
        "end": end,
        "limit": limit,
    }
    return await _fetch(connection, NEAREST_SQL, params)
