"""Dagster definitions for the standalone AI ingestion code location."""

from __future__ import annotations

import os
from datetime import timedelta
from zoneinfo import ZoneInfo

from dagster import (
    DefaultScheduleStatus,
    Definitions,
    Field,
    Int,
    Nothing,
    Out,
    RunRequest,
    String,
    StringSource,
    job,
    op,
    schedule,
)
from sqlalchemy.engine import make_url

from app.ingestion.news_embedding_pipeline import run_news_embedding_window

LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _run_tags() -> dict:
    return {
        "dagster-k8s/config": {
            "container_config": {
                "resources": {
                    "requests": {"cpu": "250m", "memory": "512Mi"},
                    "limits": {"cpu": "1", "memory": "2Gi"},
                },
            },
            "pod_spec_config": {
                "restart_policy": "Never",
                "node_selector": {
                    "kubernetes.io/hostname": os.getenv(
                        "DAGSTER_AI_RUN_NODE", "finsights-prod-node-01"
                    )
                },
            },
            "pod_template_spec_metadata": {
                "labels": {
                    "app.kubernetes.io/part-of": "finsights-ai-pipelines",
                    "app.kubernetes.io/component": "news-embedding",
                }
            },
        }
    }


def _op_config_schema(
    *,
    default_start_date: str = "2026-11-01",
    default_end_date: str = "2026-11-23",
    default_model: str = "voyage-4-large",
    default_provider: str = "auto",
    default_base_url: str = "",
    default_postgres_target: str = "primary",
):
    return {
        "start_date": Field(
            String,
            default_value=default_start_date,
            description="Inclusive Vietnam calendar date YYYY-MM-DD",
        ),
        "end_date": Field(
            String,
            default_value=default_end_date,
            description="Inclusive Vietnam calendar date YYYY-MM-DD",
        ),
        "embedding_model": Field(
            String,
            default_value=default_model,
            description="Model ID served by the selected embedding provider",
        ),
        "embedding_provider": Field(
            String,
            default_value=default_provider,
            description="auto, voyage, cloudflare, or ollama",
        ),
        "embedding_base_url": Field(
            String,
            default_value=default_base_url,
            description="Base URL for an OpenAI-compatible embedding endpoint such as Ollama",
        ),
        "postgres_target": Field(
            String,
            default_value=default_postgres_target,
            description="primary writes to finsights_embeddings; qwen writes to finsights_embeddings_qwen",
        ),
        "cloudflare_account_id": Field(
            String,
            default_value="",
            description="Optional per-run Cloudflare account ID; otherwise use CLOUDFLARE_ACCOUNT_ID",
        ),
        "api_key": Field(
            StringSource,
            default_value="",
            is_required=False,
            description="Use an environment source for hosted providers; Ollama does not require a key",
        ),
        "page_size": Field(
            Int,
            default_value=100,
            description="Maximum MySQL articles held in one processing page (1–500)",
        ),
        "dry_run": Field(
            bool,
            default_value=False,
            description="Clean and chunk only; do not call embedding APIs or write pgvector",
        ),
    }


def _postgres_dsn_for_target(target: str) -> str:
    base_dsn = os.getenv("POSTGRES_RAG_DSN", "").strip()
    if not base_dsn:
        raise RuntimeError("POSTGRES_RAG_DSN is required for vector ingestion")
    if target == "primary":
        return base_dsn
    if target == "qwen":
        return (
            make_url(base_dsn)
            .set(database="finsights_embeddings_qwen")
            .render_as_string(hide_password=False)
        )
    raise ValueError("postgres_target must be 'primary' or 'qwen'")


def _execute_news_embedding_op(context):
    config = context.op_config
    model = str(config["embedding_model"])
    provider = str(config["embedding_provider"]).strip().casefold()
    account_id = (
        str(config.get("cloudflare_account_id", "")).strip()
        or os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        or None
    )
    base_url = (
        str(config.get("embedding_base_url", "")).strip()
        or os.getenv("QWEN_EMBEDDING_BASE_URL", "").strip()
        or None
    )
    if (
        provider == "cloudflare" or (provider == "auto" and model.startswith("@cf/"))
    ) and not account_id:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID is required when using a Cloudflare model"
        )
    if provider == "ollama" and not base_url:
        raise RuntimeError(
            "embedding_base_url or QWEN_EMBEDDING_BASE_URL is required for Ollama"
        )
    summary = run_news_embedding_window(
        start_date=str(config["start_date"]),
        end_date=str(config["end_date"]),
        embedding_model=model,
        api_key=str(config.get("api_key", "")),
        cloudflare_account_id=account_id,
        embedding_provider=provider,
        embedding_base_url=base_url,
        postgres_dsn=(
            None
            if bool(config["dry_run"])
            else _postgres_dsn_for_target(str(config["postgres_target"]))
        ),
        page_size=int(config["page_size"]),
        dry_run=bool(config["dry_run"]),
        progress=context.log.info,
    )
    context.log.info("News embedding run summary: %s", summary)


@op(
    name="preprocess_chunk_embed_news",
    out=Out(Nothing),
    config_schema=_op_config_schema(),
    description="Read licensed full-text financial news, clean, chunk, embed, and upsert idempotently to pgvector.",
)
def preprocess_chunk_embed_news(context):
    _execute_news_embedding_op(context)


@op(
    name="preprocess_chunk_embed_news_qwen",
    out=Out(Nothing),
    config_schema=_op_config_schema(
        default_start_date="2026-09-01",
        default_end_date="2026-11-23",
        default_model="qwen3-embedding:0.6b",
        default_provider="ollama",
        default_base_url="http://100.82.123.58:31434",
        default_postgres_target="qwen",
    ),
    description="Backfill licensed, financially relevant news into the isolated Qwen pgvector database via the self-hosted Ollama endpoint.",
)
def preprocess_chunk_embed_news_qwen(context):
    _execute_news_embedding_op(context)


@job(
    name="news_embedding_backfill_job",
    tags=_run_tags(),
    description="Backfill licensed financial-news chunks into the primary pgvector database; reruns skip unchanged content.",
)
def news_embedding_backfill_job():
    preprocess_chunk_embed_news()


@job(
    name="news_embedding_qwen_backfill_job",
    tags=_run_tags(),
    description="Backfill financial news into the isolated Qwen embedding database; leaves the primary embedding database untouched.",
)
def news_embedding_qwen_backfill_job():
    preprocess_chunk_embed_news_qwen()


@schedule(
    job=news_embedding_backfill_job,
    cron_schedule="0 5 * * *",
    execution_timezone="Asia/Ho_Chi_Minh",
    name="news_embedding_daily_schedule",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Daily previous-day news indexing. Enable after NEWS_EMBEDDING_API_KEY and PostgreSQL credentials are configured.",
)
def news_embedding_daily_schedule(context):
    publication_day = context.scheduled_execution_time.astimezone(
        LOCAL_TZ
    ).date() - timedelta(days=1)
    model = (
        os.getenv("NEWS_EMBEDDING_MODEL", "voyage-4-large").strip() or "voyage-4-large"
    )
    op_config = {
        "start_date": publication_day.isoformat(),
        "end_date": publication_day.isoformat(),
        "embedding_model": model,
        "api_key": {"env": "NEWS_EMBEDDING_API_KEY"},
        "page_size": 100,
        "dry_run": False,
    }
    return RunRequest(
        run_key=f"news-embedding-{model}-{publication_day.isoformat()}",
        run_config={"ops": {"preprocess_chunk_embed_news": {"config": op_config}}},
    )


defs = Definitions(
    jobs=[news_embedding_backfill_job, news_embedding_qwen_backfill_job],
    schedules=[news_embedding_daily_schedule],
)
