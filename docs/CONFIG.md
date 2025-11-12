# Configuration and Migrations

## Environment precedence

`Settings` now loads configuration from YAML files and environment variables. Runtime
environment variables always win over values defined in `config/*.yaml`. Use this when
you need to override database credentials or swap between SQLite and PostgreSQL without
changing committed YAML files.

Example:

```bash
export DATABASE_TYPE=postgres
export POSTGRES_HOST=prod-db.internal
export POSTGRES_PORT=5433
```

With those overrides in place, `Settings()` will connect to PostgreSQL even if the YAML
configuration still lists `sqlite`.

## Database migrations

All schema changes are managed through Alembic.

```bash
alembic upgrade head
```

The upgrade path now includes:

- Renaming `ingestion_state` to `slack_ingestion_state` and adding resume columns.
- Adding `source_id` to `event_candidates` with a backfill for legacy rows.
- Creating the `pipeline_tasks` queue table with supporting indexes.

The legacy `scripts/fix_postgres_schema.py` helper has been retired. Running the Alembic
migration is sufficient for both fresh installs and upgrades.
