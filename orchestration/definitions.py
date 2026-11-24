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


@op(
    name="preprocess_chunk_embed_news",
    out=Out(Nothing),
    config_schema={
        "start_date": Field(
            String,
            default_value="2026-11-01",
            description="Inclusive Vietnam calendar date YYYY-MM-DD",
        ),
        "end_date": Field(
            String,
            default_value="2026-11-23",
            description="Inclusive Vietnam calendar date YYYY-MM-DD",
        ),
        "embedding_model": Field(
            String,
            default_value="voyage-4-large",
            description="Voyage model ID or Cloudflare @cf/... ID",
        ),
        "api_key": Field(
            StringSource,
            is_required=True,
            description="Prefer an environment source, e.g. {env: NEWS_EMBEDDING_API_KEY}; never paste a key into run config",
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
    },
    description="Read licensed full-text financial news, clean, chunk, embed, and upsert idempotently to pgvector.",
)
def preprocess_chunk_embed_news(context):
    config = context.op_config
    model = str(config["embedding_model"])
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip() or None
    if model.startswith("@cf/") and not account_id:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID is required when embedding_model starts with @cf/"
        )
    summary = run_news_embedding_window(
        start_date=str(config["start_date"]),
        end_date=str(config["end_date"]),
        embedding_model=model,
        api_key=str(config["api_key"]),
        cloudflare_account_id=account_id,
        page_size=int(config["page_size"]),
        dry_run=bool(config["dry_run"]),
        progress=context.log.info,
    )
    context.log.info("News embedding run summary: %s", summary)


@job(
    name="news_embedding_backfill_job",
    tags=_run_tags(),
    description="Backfill licensed financial-news chunks into the pgvector database; reruns skip unchanged content.",
)
def news_embedding_backfill_job():
    preprocess_chunk_embed_news()


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
    jobs=[news_embedding_backfill_job],
    schedules=[news_embedding_daily_schedule],
)
