"""Factory for creating database repository instances."""

from typing import cast

from src.adapters.sqlite_repository import SQLiteRepository

try:
    from src.adapters.postgres_repository import PostgresRepository
except ImportError:
    PostgresRepository = None  # type: ignore[misc,assignment]
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.domain.protocols import RepositoryProtocol

logger = get_logger(__name__)


def create_repository(settings: Settings) -> RepositoryProtocol:
    """Create appropriate repository based on settings.

    Args:
        settings: Application settings

    Returns:
        Repository instance (SQLite or PostgreSQL)

    Raises:
        ValueError: If database_type is not supported
        RepositoryError: On connection errors
    """
    if settings.database_type == "sqlite":
        logger.info("repository_sqlite_selected", path=settings.db_path)
        return cast(
            RepositoryProtocol,
            SQLiteRepository(
                db_path=settings.db_path,
                chunk_size=settings.bulk_upsert_chunk_size,
            ),
        )

    elif settings.database_type == "postgres":
        if PostgresRepository is None:
            raise ImportError(
                "PostgreSQL repository not available. Install psycopg2 or psycopg2-binary."
            )

        # Validate that password is provided
        if not settings.postgres_password:
            raise ValueError(
                "POSTGRES_PASSWORD environment variable must be set when using PostgreSQL"
            )

        logger.info(
            "repository_postgres_selected",
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_database,
            user=settings.postgres_user,
        )
        return cast(
            RepositoryProtocol,
            PostgresRepository(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_database,
                user=settings.postgres_user,
                password=settings.postgres_password.get_secret_value(),
                settings=settings,
            ),
        )

    else:
        raise ValueError(
            f"Unsupported database type: {settings.database_type}. "
            f"Must be 'sqlite' or 'postgres'"
        )
