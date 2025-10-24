"""Helpers for correlation identifiers in logs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from src.config.logging_config import bind_context, unbind_context

CORRELATION_ID_KEY = "correlation_id"


@contextmanager
def correlation_scope(existing_id: str | None = None) -> Iterator[str]:
    """Bind a correlation identifier for the lifetime of the context."""

    correlation_id = existing_id or str(uuid4())
    bind_context(**{CORRELATION_ID_KEY: correlation_id})
    try:
        yield correlation_id
    finally:
        unbind_context(CORRELATION_ID_KEY)


__all__ = ["CORRELATION_ID_KEY", "correlation_scope"]
