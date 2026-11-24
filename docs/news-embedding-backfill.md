# Dagster news embedding pipeline

The Dagster code location is separate from the existing FastAPI/local RAG flow. The local Voyage integration and `app/rag/chunking/news.py` remain unchanged. Dagster runs `app/ingestion/news_embedding_pipeline.py` through `orchestration/definitions.py`.

## Backfill run config

Select `news_embedding_backfill_job` in Dagster and use this config for the requested window:

```yaml
ops:
  preprocess_chunk_embed_news:
    config:
      start_date: "2026-11-01"
      end_date: "2026-11-23"
      embedding_model: "@cf/baai/bge-m3"
      cloudflare_account_id: "YOUR_CLOUDFLARE_ACCOUNT_ID"
      api_key:
        env: NEWS_EMBEDDING_API_KEY
      page_size: 100
      dry_run: false
```

The API key is resolved inside the run pod from `NEWS_EMBEDDING_API_KEY`; never place its value in Dagster run config. `cloudflare_account_id` is an optional per-run override; if omitted, Cloudflare model IDs use `CLOUDFLARE_ACCOUNT_ID` from the environment. Omit `embedding_model` to use `voyage-4-large` (1024 dimensions). The pgvector table has one vector space at a time, so changing the model requires an intentional full re-embedding migration.

## Isolated Qwen evaluation backfill

The separate `news_embedding_qwen_backfill_job` writes to PostgreSQL database `finsights_embeddings_qwen`, leaving `finsights_embeddings` unchanged. CoreDB provisions the new database and its own `public.news_article_chunks` table with `vector(1024)`. It uses the existing restricted `finsights_embeddings_app` login, which has explicit access to both databases.

The Qwen job uses the Ollama OpenAI-compatible `/v1/embeddings` endpoint. It needs no API key. The endpoint can be overridden with `embedding_base_url` or the `QWEN_EMBEDDING_BASE_URL` environment variable. Requests contain at most four chunks with a 300-second timeout. The pipeline can issue two requests concurrently; the Ollama server must also be configured with `OLLAMA_NUM_PARALLEL=2` and restarted, otherwise it queues requests serially. The job checkpoints every ten articles by default.

```yaml
ops:
  preprocess_chunk_embed_news_qwen:
    config:
      start_date: "2026-09-01"
      end_date: "2026-11-23"
      embedding_model: "qwen3-embedding:0.6b"
      embedding_provider: "ollama"
      embedding_base_url: "http://100.82.123.58:31434"
      ollama_concurrency: 2
      postgres_target: "qwen"
      page_size: 10
      dry_run: false
```

On a Linux host running Ollama as a systemd service, configure server-side concurrency with a service override:

```ini
# sudo systemctl edit ollama
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
```

Then run `sudo systemctl daemon-reload` and `sudo systemctl restart ollama`. Confirm that the API is healthy before resuming the Dagster backfill. Two parallel contexts use more RAM than one; increase beyond two only after checking host memory and measuring throughput.

Dates remain inclusive Vietnam calendar dates. This job applies the same licensed-content, financial relevance, preprocessing, and chunking rules as the primary job. The separate database lets evaluation compare models on the same article/chunk inputs without replacing the primary vectors.

`start_date` and `end_date` are inclusive Vietnam calendar dates. The job converts them to `[00:00, next-day 00:00)` in `Asia/Ho_Chi_Minh`, then queries the MySQL Node 2 news source. A daily schedule is defined at 05:00 Vietnam time and processes the prior calendar day. It starts stopped until the Cloudflare/Voyage API key and PostgreSQL credentials are configured.

## Data-quality and idempotency behavior

Only published articles with licensed full text at both publisher and article level and an explicit financial relevance label are eligible. Rows marked irrelevant, excluded, pending, unknown, or review are skipped. The pipeline does not substitute a title or summary when the body is absent or too short.

Preprocessing extracts text from HTML if present, removes active/non-content elements and a conservative set of page chrome, decodes entities, normalizes Unicode and whitespace, preserves paragraph boundaries and financial figures, removes exact duplicated long paragraphs, and reports cleaning counters. It does not summarize or rewrite the story.

Chunking packs adjacent paragraphs, splits oversized paragraphs at sentence/word boundaries, adds a small overlap, prefixes every chunk with its title, and computes SHA-256 over the exact embedding input. There is no per-article chunk cap. The writer skips matching `(article_uid, chunk_index, content_hash, embedding_model)` rows, upserts changed chunks transactionally per page, and removes stale tail chunks. A failed provider request does not commit that page; completed pages can be safely rerun.

## Run summary

Dagster logs contain counts for rows read/skipped/eligible, source counts, cleaning changes, chunks created/unchanged/embedded/upserted, request count, and token count when the provider returns it. Cloudflare neuron consumption is authoritative in the account usage dashboard; the API response does not consistently expose a neuron counter.
