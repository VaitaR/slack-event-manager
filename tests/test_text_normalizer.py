"""Tests for text normalization service."""

from src.services import text_normalizer


def test_normalize_text_basic() -> None:
    """Test basic text normalization."""
    text = "Check out HTTPS://EXAMPLE.COM for details!"
    result = text_normalizer.normalize_text(text)
    assert result == "check out for details!"


def test_normalize_text_removes_code_blocks() -> None:
    """Test code block removal."""
    text = "Here is code: ```python\nprint('hello')\n``` and more text"
    result = text_normalizer.normalize_text(text)
    assert "```" not in result
    assert "print" not in result
    assert "here is code" in result
    assert "and more text" in result


def test_normalize_text_removes_inline_code() -> None:
    """Test inline code removal."""
    text = "Use the `function_name` in your code"
    result = text_normalizer.normalize_text(text)
    assert "`" not in result
    assert "function_name" not in result
    assert "use the in your code" in result


def test_normalize_text_removes_urls() -> None:
    """Test URL removal."""
    text = "Visit https://example.com and www.test.com for info"
    result = text_normalizer.normalize_text(text)
    assert "example.com" not in result
    assert "test.com" not in result
    assert "visit and for info" in result


def test_normalize_text_removes_slack_mentions() -> None:
    """Test Slack mention removal."""
    text = "Hey <@U123456> check <#C123456|general>"
    result = text_normalizer.normalize_text(text)
    assert "<@" not in result
    assert "general" in result  # Channel name preserved


def test_normalize_text_collapses_whitespace() -> None:
    """Test whitespace collapsing."""
    text = "Too    many     spaces\n\n\nand\nnewlines"
    result = text_normalizer.normalize_text(text)
    assert "  " not in result  # No double spaces
    assert result == "too many spaces and newlines"


def test_extract_blocks_text_empty() -> None:
    """Test blocks extraction with empty input."""
    assert text_normalizer.extract_blocks_text(None) == ""
    assert text_normalizer.extract_blocks_text([]) == ""


def test_extract_blocks_text_simple() -> None:
    """Test blocks extraction with simple blocks."""
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}},
        {"type": "section", "text": {"type": "plain_text", "text": "World"}},
    ]
    result = text_normalizer.extract_blocks_text(blocks)
    assert "Hello" in result
    assert "World" in result


def test_extract_blocks_text_nested() -> None:
    """Test blocks extraction with nested structure."""
    blocks = [
        {
            "type": "section",
            "text": {"text": "First"},
            "fields": [{"text": "Field 1"}, {"text": "Field 2"}],
        }
    ]
    result = text_normalizer.extract_blocks_text(blocks)
    assert "First" in result
    assert "Field 1" in result
    assert "Field 2" in result


def test_extract_blocks_from_json_valid() -> None:
    """Test JSON blocks extraction."""
    blocks_json = '[{"text": {"text": "Test"}}]'
    result = text_normalizer.extract_blocks_from_json(blocks_json)
    assert "Test" in result


def test_extract_blocks_from_json_invalid() -> None:
    """Test JSON blocks extraction with invalid JSON."""
    result = text_normalizer.extract_blocks_from_json("invalid json")
    assert result == ""


def test_extract_blocks_from_json_empty() -> None:
    """Test JSON blocks extraction with empty."""
    assert text_normalizer.extract_blocks_from_json("") == ""
    assert text_normalizer.extract_blocks_from_json("[]") == ""


def test_combine_text_sources_identical() -> None:
    """Test combining identical text sources."""
    text = "Same text"
    blocks_text = "Same text"
    result = text_normalizer.combine_text_sources(text, blocks_text)
    assert result == "Same text"


def test_combine_text_sources_blocks_contains_text() -> None:
    """Test combining when blocks contain text."""
    text = "Short"
    blocks_text = "Short and longer version"
    result = text_normalizer.combine_text_sources(text, blocks_text)
    assert result == "Short and longer version"


def test_combine_text_sources_different() -> None:
    """Test combining different text sources."""
    text = "Text part"
    blocks_text = "Blocks part"
    result = text_normalizer.combine_text_sources(text, blocks_text)
    assert "Text part" in result
    assert "Blocks part" in result
