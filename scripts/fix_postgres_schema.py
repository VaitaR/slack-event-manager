#!/usr/bin/env python3
"""Deprecated manual schema fixer."""

from __future__ import annotations

from textwrap import dedent


def main() -> None:
    """Exit with guidance to use Alembic migrations instead."""

    message = dedent(
        """
        The manual schema fixer has been retired.

        Slack Event Manager now manages PostgreSQL schema changes via Alembic.
        Run the standard migration workflow instead:

            alembic upgrade head

        See docs/CONFIG.md for the latest deployment instructions.
        """
    ).strip()
    raise SystemExit(message)


if __name__ == "__main__":
    main()
