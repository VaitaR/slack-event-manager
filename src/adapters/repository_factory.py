"""Repository factory for creating database adapters based on configuration.

This factory implements the Abstract Factory pattern to provide
a clean interface for creating repository instances.
"""

from src.config.settings import Settings
from src.domain.protocols import RepositoryProtocol


def create_repository(settings: Settings) -> RepositoryProtocol:
    """Create repository based on configuration.

    Args:
        settings: Application settings

    Returns:
        Repository instance (SQLite or PostgreSQL)

    Raises:
        ValueError: If database type is not supported
        RepositoryError: On connection errors

    Example:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings()
        >>> repo = create_repository(settings)
        >>> # Use repo for all database operations
    """
    if settings.database_type == "postgres":
        from src.adapters.postgres_repository import PostgresRepository

        if settings.postgres_password is None:
            raise ValueError(
                "POSTGRES_PASSWORD environment variable must be set "
                "when using PostgreSQL database"
            )

        return PostgresRepository(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_database,
            user=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
        )
    elif settings.database_type == "sqlite":
        from src.adapters.sqlite_repository import SQLiteRepository

        return SQLiteRepository(db_path=settings.db_path)
    else:
        raise ValueError(
            f"Unsupported database type: {settings.database_type}. "
            "Supported types: sqlite, postgres"
        )
