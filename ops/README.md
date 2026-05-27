# HybridRAG Observability + Ops

Compose stack for Prometheus + Grafana + OpenTelemetry Collector, plus backup + cron eval helpers.

## Start observability stack

```bash
cd ops
docker compose -f docker-compose.observability.yml up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin — change on first login)
- OTLP gRPC: localhost:4317, HTTP: localhost:4318

## Enable telemetry from the API

In `backend/.env`:

```
METRICS_ENABLED=True
OTEL_ENABLED=True
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
RATE_LIMIT_ENABLED=True
REDIS_URL=redis://localhost:6379/0
```

OTEL is a soft dep — install the libs if you want spans exported:

```
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-psycopg2 opentelemetry-instrumentation-httpx
```

Metrics flow without any optional install (prometheus-client is in `requirements.api.txt`).

## Cron eval (Phase 6.7)

Weekly RAGAS run + regression alert.

```bash
python scripts/cron_eval.py --golden data/eval/golden_v1.jsonl --regression-threshold 0.05 \
    --slack-webhook "$SLACK_ALERT_WEBHOOK"
```

Linux crontab (every Monday 06:00):
```
0 6 * * 1 cd /path/to/backend && /usr/bin/python scripts/cron_eval.py >> /var/log/cron_eval.log 2>&1
```

Windows Task Scheduler:
```powershell
schtasks /Create /SC WEEKLY /D MON /TN "HybridRAG-Eval" /TR "python D:\my-projects\nlp\HybridRAG\backend\scripts\cron_eval.py" /ST 06:00
```

Exit codes: `0` = pass; `1` = regression detected; `2` = crash.

## Backup (Phase 6.8)

Linux/macOS:
```bash
export POSTGRES_HOST=localhost POSTGRES_PORT=5000 POSTGRES_USER=admin POSTGRES_DB=utehy POSTGRES_PASSWORD=...
export WEAVIATE_URL=http://localhost:8080
export VECTOR_STORE_DIR=/path/to/backend/data/vector_store
export MINIO_ALIAS=local  # `mc alias set local http://localhost:9000 ...`
export MINIO_BUCKET=hybridrag-backups
export RETENTION_DAYS=7
bash backend/scripts/backup.sh
```

Windows:
```powershell
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5000"
$env:POSTGRES_USER = "admin"
$env:POSTGRES_DB = "utehy"
$env:POSTGRES_PASSWORD = "..."
$env:WEAVIATE_URL = "http://localhost:8081"
$env:VECTOR_STORE_DIR = "D:\my-projects\nlp\HybridRAG\backend\data\vector_store"
$env:MINIO_ALIAS = "local"
$env:MINIO_BUCKET = "hybridrag-backups"
.\backend\scripts\backup.ps1
```

Required CLIs on PATH: `pg_dump`, `mc` (MinIO Client).

Nightly cron (Linux):
```
0 3 * * * /path/to/backend/scripts/backup.sh >> /var/log/hybridrag-backup.log 2>&1
```

## Grafana dashboard

`grafana/dashboards/hybridrag_overview.json` auto-loads via Grafana provisioning.

Panels:
1. Requests/sec by route
2. HTTP latency p95
3. LLM tokens/min by model + direction
4. LLM cost USD 24h by model
5. Retrieval p95 by backend
6. Generation p95 by model
7. Intent breakdown (1h pie)
8. Refusals/hour
9. Clarifications/hour by reason
10. Verification failures/hour
11. Tool call success rate
12. HTTP 5xx error rate
13. Active sessions (24h)

To edit + save without losing on rebuild, use Dashboard JSON model export in Grafana UI and overwrite the file.

## Linux note

`host.docker.internal` works on Docker Desktop (Windows/macOS). On Linux either:
- Run compose with `network_mode: host`, or
- Replace `host.docker.internal` in `prometheus.yml` with `172.17.0.1` (default docker0 bridge), or
- Add `extra_hosts: ["host.docker.internal:host-gateway"]` to the prometheus service.
