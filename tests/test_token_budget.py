"""Tests for token budget heuristics."""

from __future__ import annotations

from src.services import token_budget


def test_token_budget_truncate_deterministic() -> None:
    """Long text should truncate deterministically within budget."""

    text = " ".join(["alpha"] * 100)
    budget = 40

    first = token_budget.truncate_or_chunk(text, budget)
    second = token_budget.truncate_or_chunk(text, budget)

    assert first == second
    assert len(first) > 1
    assert len(first[0]) <= budget
    assert first[0].endswith(token_budget.ELLIPSIS)


def test_token_budget_chunking_deterministic() -> None:
    """Large text should be chunked deterministically into stable pieces."""

    text = " ".join(f"section-{index:02d}" for index in range(30))
    budget = 50

    first = token_budget.truncate_or_chunk(text, budget)
    second = token_budget.truncate_or_chunk(text, budget)

    assert first == second
    assert len(first) > 1
    assert all(len(chunk) <= budget for chunk in first)

    # Ensure join reconstructs normalized text without ellipsis artifacts
    reconstructed = " ".join(
        chunk.removesuffix(token_budget.ELLIPSIS).strip() for chunk in first
    )
    assert reconstructed.startswith("section-00")
