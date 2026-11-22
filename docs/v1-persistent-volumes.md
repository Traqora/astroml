# V1 Persistent Volumes

This document defines the persistent-storage contract for the supported v1
Docker Compose deployment. It does not cover Kubernetes or optional monitoring
and development profiles.

Downloaded ledger files are temporary v1 ingestion input and are not part of the persistence contract.
Purge them after checkpoints and derived features are durably committed.

## Supported v1 boundary

The v1 profile consists of:

- PostgreSQL
- Redis
- AstroML API
- Feature store
- Ingestion

The following are outside the v1 persistence contract:

- Prometheus and Grafana
- GPU/CPU training jobs
- Development/Jupyter containers
- Celery/Flower
- Soroban services

These optional services must not be required for database, cache, feature-store,
or ingestion recovery.

## Required named volumes

| Named volume | Container path | Required by | Purpose | Minimum estimate |
| --- | --- | --- | --- | --- |
| `postgres_data` | `/var/lib/postgresql/data` | PostgreSQL | Ledger and application state | 20 GB |
| `redis_data` | `/data` | Redis | AOF and RDB persistence for cache/job state | 5 GB |
| `feature_store_data` | `/app/feature_store` | Feature store and ingestion | Feature artifacts and metadata | 50 GB |
| `ingestion_data` | `/app/data` | Ingestion | Checkpoints and resumable ingestion state | 10 GB |

The Compose definitions declare all four volumes with the `local` driver.
The API and application logs are runtime data and are not included in the v1
state recovery contract.

## Resource expectations

Provision at least 85 GB of usable storage for the v1 profile, with at least
20% headroom. Start with:

- PostgreSQL: 0.5 CPU and 512 MiB reserved, 2 CPU and 2 GiB maximum
- Redis: 0.25 CPU and 128 MiB reserved, 1 CPU and 512 MiB maximum
- API/feature-store/ingestion: follow the documented service limits and monitor
  host memory and disk before raising sustained workload

Measure actual growth after representative backfills. Increase volumes when
disk usage exceeds 70%; never allow PostgreSQL or Redis storage to reach full.

## Create and deploy

From the repository root:

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD and required application secrets in .env.
docker compose up -d postgres redis feature-store ingestion api
docker compose ps
```

For the production override:

```bash
POSTGRES_PASSWORD='replace-me' docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  up -d postgres redis feature-store ingestion production
```

Do not use `docker compose down -v` except when intentionally deleting v1 state.
It removes named volumes.

## Backup and recovery

Back up PostgreSQL with logical dumps for portable recovery and retain volume
snapshots or filesystem backups for lower-RPO disaster recovery:

```bash
docker compose exec postgres pg_dump \
  -U astroml astroml > astroml-backup.sql
```

Back up feature-store and ingestion state as separate named volumes:

```bash
docker run --rm \
  -v astroml_feature_store_data:/data:ro \
  -v "$PWD:/backup" alpine \
  tar czf /backup/feature-store-backup.tar.gz -C /data .

docker run --rm \
  -v astroml_ingestion_data:/data:ro \
  -v "$PWD:/backup" alpine \
  tar czf /backup/ingestion-data-backup.tar.gz -C /data .
```

Redis is configured with AOF persistence. It is a cache/job-state cache, not a
system of record; resubmission or replay of PostgreSQL-backed work is the
supported recovery path.

To restore PostgreSQL:

```bash
docker compose stop api feature-store ingestion
docker compose exec -T postgres psql -U astroml astroml < astroml-backup.sql
docker compose start api feature-store ingestion
```

Restore the named volumes from their archives only on an isolated replacement
deployment, then start PostgreSQL and verify database consistency:

```bash
docker compose exec postgres pg_isready -U astroml -d astroml
docker compose ps
```

## Recovery boundaries

- PostgreSQL is authoritative for transactional application state.
- Redis recovery is best-effort and uses AOF/RDB persistence under
  `/data`.
- Feature-store state and ingestion checkpoints must be restored together only
  if an exact point-in-time application state is required.
- Application log volumes are not required to recover v1 operation.
- Rehearse restore on a throwaway database and record achieved RPO and RTO.
  The PostgreSQL runbook has detailed PITR instructions.

The supported local profile is the five-service set above; no optional services are required.
