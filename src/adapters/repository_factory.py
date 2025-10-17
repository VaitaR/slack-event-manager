"""Factory for creating database repository instances.

Supports SQLite (development) and PostgreSQL (production).
"""

from src.adapters.postgres_repository import PostgresRepository
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.protocols import RepositoryProtocol


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
        print(f"📁 SQLite database mode: {settings.db_path}")
        return SQLiteRepository(db_path=settings.db_path)

    elif settings.database_type == "postgres":
        # Validate that password is provided
        if not settings.postgres_password:
            raise ValueError(
                "POSTGRES_PASSWORD environment variable must be set when using PostgreSQL"
            )

        print(
            f"🐘 PostgreSQL database mode: {settings.postgres_user}@"
            f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_database}"
        )
        return PostgresRepository(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_database,
            user=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
        )

    else:
        raise ValueError(
            f"Unsupported database type: {settings.database_type}. "
            f"Must be 'sqlite' or 'postgres'"
        )
