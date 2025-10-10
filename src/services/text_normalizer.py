"""Text normalization service for Slack messages.

Handles:
- Code block removal
- URL removal (after link extraction)
- Whitespace normalization
- Block Kit text extraction
"""

import json
import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize message text for analysis.

    Args:
        text: Raw message text

    Returns:
        Normalized text (lowercase, no URLs, collapsed whitespace)

    Example:
        >>> normalize_text("Check out https://example.com for details!")
        'check out for details!'
    """
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove URLs (they're extracted separately)
    text = re.sub(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove Slack user/channel mentions formatting
    text = re.sub(r"<@[A-Z0-9]+>", " ", text)
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"\1", text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def extract_blocks_text(blocks: list[dict[str, Any]] | None) -> str:
    """Extract text content from Slack Block Kit blocks.

    Args:
        blocks: Slack blocks array or None

    Returns:
        Concatenated text from all blocks

    Example:
        >>> blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}]
        >>> extract_blocks_text(blocks)
        'Hello'
    """
    if not blocks:
        return ""

    texts: list[str] = []

    def extract_text_recursive(obj: Any) -> None:
        """Recursively extract text fields."""
        if isinstance(obj, dict):
            if "text" in obj and isinstance(obj["text"], str):
                texts.append(obj["text"])
            for value in obj.values():
                extract_text_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_text_recursive(item)

    extract_text_recursive(blocks)
    return " ".join(texts)


def extract_blocks_from_json(blocks_json: str) -> str:
    """Parse blocks JSON and extract text.

    Args:
        blocks_json: JSON string of blocks

    Returns:
        Extracted text or empty string on error

    Example:
        >>> extract_blocks_from_json('[{"text": {"text": "Hi"}}]')
        'Hi'
    """
    if not blocks_json or blocks_json == "[]":
        return ""

    try:
        blocks = json.loads(blocks_json)
        return extract_blocks_text(blocks)
    except (json.JSONDecodeError, TypeError):
        return ""


def combine_text_sources(text: str, blocks_text: str) -> str:
    """Combine text from message and blocks, avoiding duplication.

    Args:
        text: Main message text
        blocks_text: Text extracted from blocks

    Returns:
        Combined text with duplicates removed

    Example:
        >>> combine_text_sources("Hello world", "Hello world Extra")
        'Hello world Extra'
    """
    if not blocks_text or blocks_text == text:
        return text

    # If blocks_text contains text, prefer blocks (often richer)
    if text and text in blocks_text:
        return blocks_text

    # Otherwise concatenate
    return f"{text} {blocks_text}".strip()

