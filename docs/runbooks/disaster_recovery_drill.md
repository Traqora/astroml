# Disaster-Recovery Drill Runbook

## Scope

A disaster-recovery runbook that is never rehearsed is a hypothesis, not a
plan. This runbook defines the recurring drill that proves the
[PostgreSQL Disaster-Recovery Runbook](postgres_disaster_recovery.md) still
works end to end: a backup is taken, restored into a scratch database,
verified, and rolled back inside a single time-boxed session.

It complements, and does not replace, the deployment-rollback guidance in
[disaster-recovery.md](../disaster-recovery.md). The drill exercises the
*database* path; the model-rollback path is covered by
`astroml.deployment.rollback_manager`.

Assumptions match `docker-compose.yml` and `.env.example` while the
[PostgreSQL Disaster-Recovery Runbook](postgres_disaster_recovery.md)
unless noted:

- container: `astroml-postgres`, db/user: `astroml`, port `5432`
- scratch database used for the restore: `astroml_drill`
- backups produced by `scripts/docker-backup.sh` into `./backups/`

## 1. Cadence and roles

| Cadence | Owner | Output |
| --- | --- | --- |
| Quarterly (min.) | on-call DBA | drill report attached to the incident tracker |
| Before any schema cutover | change author | dry-run restore of the pre-change backup |
| After a real incident | incident commander | targeted re-drill of the step that failed |

Roles for a drill:

- **Drill lead** — runs the commands, owns the clock.
- **Scribe** — timestamps every step and records achieved RTO/RPO.
- **Observer** — independent, verifies the scribe's numbers; must not be the
  drill lead.

## 2. Pre-drill checklist

- [ ] Pick a quiet window; announce to `#incidents` with a start/end time.
- [ ] Confirm the latest backup exists and is newer than the last release:
      `ls -lh ./backups/` and `docker-compose ps`.
- [ ] Confirm headroom on the volume holding `./backups/` (a restore needs
      roughly 2x the dump size).
- [ ] Record the target metrics before starting: desired RTO and RPO.
- [ ] Start the clock and note `T0`.

```bash
# Baseline: capture the numbers the restore must reproduce
docker-compose exec -T postgres psql -U astroml -d astroml -c \
  "SELECT 'ledgers', count(*) FROM ledgers UNION ALL \
   SELECT 'models',  count(*) FROM models;"
```

## 3. Drill procedure

Run against the scratch database; never drop `astroml` during a drill.

```bash
BACKUP=./backups/astroml_backup_<TIMESTAMP>
T0=$(date -u +%s)

# 1. Verify the archive before trusting it
grep -A5 "SHA256" $BACKUP/MANIFEST.txt
tar -xzf $BACKUP.tar.gz -C ./backups
gunzip -c $BACKUP/postgres.sql.gz > /tmp/drill_restore.sql

# 2. Recreate the scratch database and load the dump
docker-compose exec -T postgres psql -U astroml -d postgres \
  -c "DROP DATABASE IF EXISTS astroml_drill;" \
  -c "CREATE DATABASE astroml_drill;"
cat /tmp/drill_restore.sql | docker-compose exec -T postgres \
  psql -U astroml -d astroml_drill

# 3. Re-apply migrations newer than the backup
alembic -x db_url=postgresql://astroml:astroml@localhost:5432/astroml_drill \
  upgrade head

# 4. Compare row counts against the §2 baseline
docker-compose exec -T postgres psql -U astroml -d astroml_drill -c \
  "SELECT 'ledgers', count(*) FROM ledgers UNION ALL \
   SELECT 'models',  count(*) FROM models;"

T1=$(date -u +%s)
echo "achieved RTO: $((T1 - T0))s (target: <TARGET_RTO>)"
```

A drill passes only if every box below is ticked:

- [ ] Dump integrity confirmed (SHA256 matches `MANIFEST.txt`).
- [ ] Scratch restore loads with no `psql` errors.
- [ ] Migrations apply cleanly on top of the restored dump.
- [ ] Row counts match the §2 baseline exactly.
- [ ] Achieved RTO is at or below the target, or a remediation item is filed.

## 4. Point-in-time recovery spot-check

At least once a year, on a postgres instance with WAL archiving enabled
(see §3 of the [PostgreSQL Disaster-Recovery
Runbook](postgres_disaster_recovery.md)), rehearse a PITR to a known
timestamp and confirm the recovered data stops where expected. If WAL
archiving is off, record that in the drill report — recovery granularity is
"last backup" and the report must say so rather than imply PITR.

## 5. Abort criteria

Stop and escalate (do **not** push through) if any of the following holds:

- the archive fails its SHA256 check, or `gunzip` errors;
- the restore drops rows that the baseline says should exist;
- migrations fail on the scratch database;
- the drill is still running at 2x the target RTO.

## 6. Post-drill

- [ ] Destroy the scratch database: `DROP DATABASE astroml_drill;`
- [ ] Delete temporary dumps (`/tmp/drill_restore.sql`).
- [ ] Post the drill report (sections 7) to the incident tracker.
- [ ] File a remediation issue for every failed checkbox, linked to this
      runbook.
- [ ] If any command in this runbook drifted from reality, update it in the
      same PR as the remediation.

## 7. Drill report template

```
Date / start-end (UTC):
Backup used (timestamp + SHA256):
Drill lead / scribe / observer:
Target RTO:        Achieved RTO:
Target RPO:        Achieved RPO:
Result:            PASS | PARTIAL | FAIL
Failed checkboxes:
Remediation issues filed:
Notes (PITR rehearsed? WAL archiving on?):
```
