"""Utilities for estimating token usage and chunking large prompts."""

from __future__ import annotations

from math import ceil
from typing import Final

DEFAULT_CHAR_PER_TOKEN: Final[int] = 4
DEFAULT_PROMPT_BUDGET: Final[int] = 3000
ELLIPSIS: Final[str] = " …"

MODEL_CHAR_PER_TOKEN: dict[str, int] = {
    "gpt-5-nano": 3,
    "gpt-4o-mini": 4,
}

MODEL_PROMPT_BUDGET: dict[str, int] = {
    "gpt-5-nano": 3500,
    "gpt-4o-mini": 6000,
}


def _char_per_token_for(model: str) -> int:
    return MODEL_CHAR_PER_TOKEN.get(model, DEFAULT_CHAR_PER_TOKEN)


def prompt_budget_for_model(model: str) -> int:
    """Return the default prompt budget (tokens) for a model."""

    return MODEL_PROMPT_BUDGET.get(model, DEFAULT_PROMPT_BUDGET)


def characters_for_tokens(tokens: int, model: str) -> int:
    """Convert a token budget to an approximate character budget."""

    if tokens <= 0:
        raise ValueError("token budget must be positive")
    per_token = _char_per_token_for(model)
    return max(1, tokens * per_token)


def estimate_tokens(text: str, model: str) -> int:
    """Estimate token usage for text based on model heuristics."""

    if not text:
        return 0
    per_token = _char_per_token_for(model)
    return max(1, ceil(len(text) / per_token))


def truncate_or_chunk(
    text: str, budget: int, strategy: str = "truncate_or_chunk"
) -> list[str]:
    """Deterministically split text into chunks within a character budget."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    if strategy != "truncate_or_chunk":
        raise ValueError(f"unsupported strategy: {strategy}")
    stripped = text.strip()
    if not stripped:
        return [stripped]

    max_chars = max(1, budget)
    segments: list[str] = []
    remaining = stripped

    while remaining:
        if len(remaining) <= max_chars:
            segments.append(remaining)
            break

        chunk, remainder = _split_once(remaining, max_chars)

        if remainder:
            ellipsis_budget = max_chars - len(ELLIPSIS)
            if len(chunk) + len(ELLIPSIS) > max_chars and ellipsis_budget > 0:
                tighter_chunk, tighter_remainder = _split_once(
                    remaining, ellipsis_budget
                )
                if tighter_chunk:
                    chunk, remainder = tighter_chunk, tighter_remainder

            if len(chunk) + len(ELLIPSIS) <= max_chars:
                segments.append(f"{chunk}{ELLIPSIS}")
            else:
                segments.append(chunk)
        else:
            segments.append(chunk)

        remaining = remainder

    return segments


def _split_once(text: str, limit: int) -> tuple[str, str]:
    if limit <= 0:
        return "", text
    if len(text) <= limit:
        return text, ""

    cut = text.rfind(" ", 0, limit)
    if cut == -1 or cut < limit // 2:
        cut = limit

    chunk = text[:cut].rstrip()
    remainder = text[cut:].lstrip()

    if not chunk:
        chunk = text[:limit].rstrip()
        remainder = text[len(chunk) :].lstrip()

    return chunk, remainder


__all__ = [
    "DEFAULT_PROMPT_BUDGET",
    "ELLIPSIS",
    "characters_for_tokens",
    "estimate_tokens",
    "prompt_budget_for_model",
    "truncate_or_chunk",
]
