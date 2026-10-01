# PostgreSQL Disaster-Recovery Runbook

## Scope

The PostgreSQL store (`astroml-postgres` container, database `astroml`) is
the system of record for ledgers, model metadata, and application state.
This runbook covers backup creation, restore, point-in-time recovery (PITR),
and rollback for that store. Model-deployment rollback (as opposed to
database recovery) lives in [disaster-recovery.md](../disaster-recovery.md).

Assumptions match `docker-compose.yml` and `.env.example` unless noted:

- container: `astroml-postgres`, db/user: `astroml`, port `5432`
- volume: `postgres_data` at `/var/lib/postgresql/data`
- backups produced by `scripts/docker-backup.sh` into `./backups/`

## 1. Backup creation

```bash
# Full backup (Postgres dump + Redis + configs + manifest with SHA256 sums)
./scripts/docker-backup.sh ./backups

# Keep only the compressed archive
./scripts/docker-backup.sh ./backups --compress
```

Verify before you need it:

```bash
ls -lh ./backups/astroml_backup_<TIMESTAMP>/
grep -A5 "SHA256" ./backups/astroml_backup_<TIMESTAMP>/MANIFEST.txt
# Spot-check the dump restores (see §2 against a scratch database first)
```

Schedule: nightly via cron for staging, at least daily for prod. Retain 7
daily + 4 weekly archives; the script does not prune — do that in the same
cron entry (`find ./backups -name '*.tar.gz' -mtime +30 -delete`).

## 2. Restore (full)

```bash
BACKUP=./backups/astroml_backup_<TIMESTAMP>
tar -xzf $BACKUP.tar.gz -C ./backups
gunzip -c $BACKUP/postgres.sql.gz > /tmp/restore.sql

# 1. Stop writers so no new rows land mid-restore
docker-compose stop ingestion api worker 2>/dev/null || true

# 2. Drop and recreate the database, then load the dump
docker-compose exec postgres psql -U astroml -d postgres \
  -c "DROP DATABASE astroml;" -c "CREATE DATABASE astroml;"
cat /tmp/restore.sql | docker-compose exec -T postgres \
  psql -U astroml -d astroml

# 3. Re-apply any migrations newer than the backup, then restart
alembic upgrade head
docker-compose up -d ingestion api worker
```

Verify: row counts on core tables (`ledgers`, `models`) against the numbers
recorded in `MANIFEST.txt`, then run the smoke ingestion check
(`docker-compose exec api python -m pytest tests/e2e -x -q -k smoke` or the
equivalent health probes).

## 3. Point-in-time recovery (PITR)

Plain `pg_dump` archives cannot replay to an arbitrary timestamp. For PITR:

1. Enable WAL archiving **before** the incident (it is off by default):
   ```bash
   # postgresql.conf (mount into the postgres service):
   wal_level = replica
   archive_mode = on
   archive_command = 'cp %p /var/lib/postgresql/wal/%f'
   ```
2. Keep a base backup (`pg_basebackup` or a `docker-backup.sh` archive) plus
   the WAL segment chain.
3. To recover to time T:
   ```bash
   # Restore the base backup into a fresh volume, create recovery.signal
   # with: restore_target_time = 'YYYY-MM-DD HH:MM:SS+00'
   # Start postgres; it replays WAL up to T, then promotes.
   ```
4. Point the app at the recovered instance, verify (§2), cut over.

Without WAL archiving, recovery granularity is "last backup" — say so in
the incident report instead of implying otherwise.

## 4. Rollback procedure (bad migration / bad deploy)

```bash
# 1. Identify the bad revision
alembic history | head -5

# 2. Roll the schema back one revision (repeat -1 per revision as needed)
alembic downgrade -1

# 3. Roll the code back to the matching image/tag
git log --oneline -5   # confirm the last-known-good ref
docker-compose up -d --force-recreate api worker

# 4. Re-run the migration test suite before declaring recovery
pytest tests/infrastructure/test_alembic_migrations.py -q
```

If the bad migration already destroyed data (dropped column/table), skip
the downgrade and do a full restore (§2) instead — downgrades reverse
schema, not data loss.

## 5. Escalation checklist

- [ ] Writers stopped before restore; confirmed no new rows mid-load
- [ ] Dump integrity checked (SHA256 in `MANIFEST.txt`, scratch restore)
- [ ] Migrations re-applied post-restore (`alembic upgrade head`)
- [ ] Row counts + smoke tests pass before cutover
- [ ] Incident note records RTO, RPO actually achieved, and whether WAL
      archiving was on (PITR possible or backup-granularity only)
