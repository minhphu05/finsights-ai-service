"""Idempotent persistence for the CoreDB-managed pgvector news chunk table."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import psycopg
from pgvector.psycopg import register_vector
from sqlalchemy.engine import make_url

TABLE = "public.news_article_chunks"


@dataclass(frozen=True)
class ExistingChunk:
    chunk_index: int
    content_hash: str
    embedding_model: str


class NewsChunkStore:
    """Owns transactions for the news chunk rows; database/schema are provisioned by CoreDB."""

    def __init__(self, *, dsn: str | None = None, password: str | None = None) -> None:
        raw_dsn = dsn or os.getenv("POSTGRES_RAG_DSN", "").strip()
        if not raw_dsn:
            raise RuntimeError("POSTGRES_RAG_DSN is required for vector ingestion")
        password = (
            password
            if password is not None
            else os.getenv("POSTGRES_EMBEDDINGS_PASSWORD", "")
        )
        url = make_url(raw_dsn)
        if password and not url.password:
            url = url.set(password=password)
        # psycopg.connect accepts the native scheme; the app's SQLAlchemy DSN may include
        # the explicit +psycopg driver suffix.
        url = url.set(drivername="postgresql")
        self.connection = psycopg.connect(
            url.render_as_string(hide_password=False), autocommit=True
        )
        register_vector(self.connection)
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM {TABLE} LIMIT 0")

    def close(self) -> None:
        self.connection.close()

    def assert_model_compatible(self, model: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT DISTINCT embedding_model FROM {TABLE} LIMIT 3")
            existing = {row[0] for row in cursor.fetchall()}
        if existing and existing != {model}:
            names = ", ".join(sorted(existing))
            raise RuntimeError(
                f"Embedding table already contains model(s) [{names}]; refusing to mix vector spaces. "
                "Use the same model or run a full re-embedding migration."
            )

    def existing_for_articles(
        self, article_uids: Sequence[str]
    ) -> dict[str, dict[int, ExistingChunk]]:
        if not article_uids:
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT article_uid, chunk_index, content_hash, embedding_model FROM {TABLE} "
                "WHERE article_uid = ANY(%s)",
                (list(article_uids),),
            )
            rows = cursor.fetchall()
        result: dict[str, dict[int, ExistingChunk]] = {}
        for article_uid, chunk_index, content_hash, model in rows:
            result.setdefault(str(article_uid), {})[int(chunk_index)] = ExistingChunk(
                chunk_index=int(chunk_index),
                content_hash=str(content_hash),
                embedding_model=str(model),
            )
        return result

    def remove_articles(self, article_uids: Sequence[str]) -> int:
        if not article_uids:
            return 0
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {TABLE} WHERE article_uid = ANY(%s)",
                (list(article_uids),),
            )
            return cursor.rowcount

    def persist_article(
        self,
        *,
        article_uid: str,
        published_at: datetime | None,
        chunk_count: int,
        changed_chunks: Sequence[dict],
        embedding_model: str,
    ) -> int:
        """Upsert changed vectors, refresh publication time, and remove obsolete tail chunks."""
        values = [
            (
                article_uid,
                int(chunk["chunk_index"]),
                published_at,
                str(chunk["text"]),
                chunk["embedding"],
                embedding_model,
                str(chunk["content_hash"]),
            )
            for chunk in changed_chunks
        ]
        with self.connection.transaction(), self.connection.cursor() as cursor:
            if values:
                cursor.executemany(
                    f"""INSERT INTO {TABLE}
                    (article_uid, chunk_index, published_at, text, embedding, embedding_model, content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_uid, chunk_index) DO UPDATE SET
                        published_at = EXCLUDED.published_at,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        content_hash = EXCLUDED.content_hash
                    WHERE {TABLE}.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                       OR {TABLE}.embedding_model IS DISTINCT FROM EXCLUDED.embedding_model
                       OR {TABLE}.published_at IS DISTINCT FROM EXCLUDED.published_at""",
                    values,
                )
            cursor.execute(
                f"UPDATE {TABLE} SET published_at = %s WHERE article_uid = %s AND published_at IS DISTINCT FROM %s",
                (published_at, article_uid, published_at),
            )
            cursor.execute(
                f"DELETE FROM {TABLE} WHERE article_uid = %s AND chunk_index >= %s",
                (article_uid, chunk_count),
            )
        return len(values)
