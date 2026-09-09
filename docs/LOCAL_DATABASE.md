# Local pgvector setup on this Mac

The development database runs directly through Homebrew. Docker is optional and
the Compose configuration remains available for other environments.

| Item | Installed/configured value |
| --- | --- |
| PostgreSQL | Homebrew `postgresql@17`, version 17.11 |
| pgvector | 0.8.6 |
| Network | `localhost:5432` only |
| Application database | `music_search` |
| Disposable integration-test database | `music_search_test` |
| Application role | `music_search`, non-superuser, owns both databases |
| Data directory | `/opt/homebrew/var/postgresql@17` |
| Runtime configuration | Ignored project `.env` |

PostgreSQL is registered as a user Homebrew service, starts at login, and retains
data across stops/restarts. TCP connections use SCRAM password authentication;
local Unix-socket administrative connections use peer authentication. Credentials
in `.env` match the existing local-development defaults in `.env.example`.

## Start, stop, and inspect

```bash
brew services start postgresql@17
brew services stop postgresql@17
brew services restart postgresql@17
brew services list
/opt/homebrew/opt/postgresql@17/bin/pg_isready -h localhost -p 5432
```

Run project commands from the repository root:

```bash
uv run music-search db-upgrade
uv run alembic current
uv run music-search ingest ./audio-small/
uv run music-search search "warm acoustic guitar with male vocals" --limit 5
```

The first two commands manage schema. Ingestion and search also need CLAP weights;
PostgreSQL alone does not perform audio understanding.

## Full test suite, including live pgvector

```bash
MUSIC_SEARCH_TEST_DATABASE_URL=postgresql+psycopg://music_search:music_search@localhost:5432/music_search_test uv run pytest -q
```

Tests create an isolated random schema in `music_search_test`, then remove only that
schema. The application catalog in `music_search` is not modified by those tests.
Both databases have the `vector` extension preinstalled by the local administrator.
The application role does not need superuser privileges for migrations or tests.

## Administrative inspection

The Homebrew PostgreSQL binaries are not added to the shell's global PATH. Use the
absolute binary path for an administrative socket connection:

```bash
/opt/homebrew/opt/postgresql@17/bin/psql -d music_search
```

Useful read-only SQL:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
SELECT version_num FROM alembic_version;
SELECT count(*) FROM tracks;
SELECT count(*) FROM segments;
```

Do not run the Compose database simultaneously on port 5432. Stop the native service
before switching. Docker volumes and this Homebrew data directory are separate
catalogs; switching runtimes does not transfer data.

For fresh Homebrew environments, install `postgresql@17` and `pgvector`, initialize
the local role/databases, and install `vector` in each database as the administrator
before running application migrations. See the supported
[pgvector installation instructions](https://github.com/pgvector/pgvector#homebrew).
