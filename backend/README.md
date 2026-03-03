# HybridRAG Backend

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#prerequisites)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](#run-with-docker-compose)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](#run-with-docker-compose)
[![MinIO](https://img.shields.io/badge/MinIO-S3%20Compatible-C72E49.svg)](#run-with-docker-compose)
[![Kafka](https://img.shields.io/badge/Kafka-Event%20Driven-231F20.svg)](#run-with-docker-compose)

Production-ready backend for a Hybrid RAG chatbot with Google sign-in, hybrid retrieval (Vector + BM25 + RRF + reranker), async ingestion, and chat session history.

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Run with Docker Compose](#run-with-docker-compose)

## Highlights

- Google login API with internal access/refresh token lifecycle.
- User-scoped chat history (`sessions` + `messages`) with ownership checks.
- Search APIs locked to server-side settings (no client override for top-k/rerank timeout knobs).
- Full `chat/answer` pipeline and SSE streaming endpoint.
- File ingestion via MinIO + Kafka event flow.
- Health endpoints for live/readiness checks.

## Architecture

<p align="center">
  <img src="./docs/images/hybridrag.jpg" alt="HybridRAG System Architecture" width="100%" />
</p>

**Core flow**
1. User question enters `Self-Reflection`.
2. Semantic router decides `chitchat` vs `retrieval` path.
3. Retrieval path executes hybrid search: vector + BM25 -> RRF -> optional reranker.
4. Top relevant chunks are used to generate final answer.
5. User and assistant messages are persisted per session.
6. Ingestion pipeline keeps BM25/FAISS stores up to date from MinIO events.

## Project Structure

```text
backend/
  Dockerfile.api             # API image build file
  docker-compose.api.yml     # API-only compose file
  build/
    kafka/                   # Kafka + Zookeeper compose
    minio/                   # MinIO compose + event setup
    postgres/                # Postgres compose + init schema
  data/
    vector_store/            # BM25 and FAISS local caches
  scripts/
  src/
    api/
      app.py                 # FastAPI app entrypoint
      auth/                  # auth internals
      core/                  # runtime/dependencies/storage
      routers/               # auth, users, search, chat, files, health
    config/
      settings.py            # runtime config from .env
    hybridrag/
      retrieval/             # BM25, vector, fusion, reranker
      rewriter/              # self-reflection / query rewrite
      router/                # keyword/semantic routing
      ingestion/
        ingestion_service/   # Kafka consumer and indexing
```

## Prerequisites

- Python `>= 3.10` (recommended `3.11`)
- Docker Engine + Docker Compose plugin
- OpenAI API key
- Google OAuth Client ID

## Quick Start

Run all commands from `backend/`.

### 1) Setup Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment

```powershell
Copy-Item .env.example .env
```

Update required fields in `.env`:

- `OPENAI_API_KEY`
- `GOOGLE_CLIENT_ID`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET_NAME`
- `KAFKA_BOOTSTRAP_SERVERS` (local default: `localhost:9092`)

## Run with Docker Compose

Create shared network once:

```powershell
docker network create utehy-network 2>$null
```

Start infrastructure and ingestion service:

```powershell
docker compose --env-file .env -f build/kafka/docker-compose.yml up -d
docker compose --env-file .env -f build/postgres/docker-compose.yml up -d
docker compose --env-file .env -f build/minio/docker-compose.yml up -d
docker compose --env-file .env -f src/hybridrag/ingestion/ingestion_service/docker-compose.yml up -d --build
```

Start API container only:

```powershell
docker compose --env-file .env -f docker-compose.api.yml up -d --build
```

Follow API logs:

```powershell
docker compose --env-file .env -f docker-compose.api.yml logs -f api
```

Stop API container:

```powershell
docker compose --env-file .env -f docker-compose.api.yml down
```

Optional checks:

```powershell
docker ps
```

Access URLs:

- Swagger UI: `http://localhost:8000/docs`
- Liveness: `http://localhost:8000/api/v1/health/live`
- Readiness: `http://localhost:8000/api/v1/health/ready`
- MinIO Console: `http://localhost:9001`
