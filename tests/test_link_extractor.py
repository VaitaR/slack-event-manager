"""Tests for link and anchor extraction service."""

from src.services import link_extractor


def test_extract_urls_basic() -> None:
    """Test basic URL extraction."""
    text = "Check https://example.com and www.test.com"
    urls = link_extractor.extract_urls(text)
    assert len(urls) == 2
    assert "https://example.com" in urls
    assert "www.test.com" in urls


def test_extract_urls_with_paths() -> None:
    """Test URL extraction with paths and params."""
    text = "Visit https://example.com/path/to/page?id=123&name=test"
    urls = link_extractor.extract_urls(text)
    assert len(urls) == 1
    assert "https://example.com/path/to/page?id=123&name=test" in urls


def test_extract_urls_removes_trailing_punctuation() -> None:
    """Test URL extraction removes trailing punctuation."""
    text = "See https://example.com, and https://test.com."
    urls = link_extractor.extract_urls(text)
    assert len(urls) == 2
    # Check that trailing punctuation is removed, but dots in domains are kept
    assert all("," not in url for url in urls)  # Remove commas
    assert all(not url.endswith(".") for url in urls)  # Remove trailing dots


def test_normalize_url_removes_utm_params() -> None:
    """Test URL normalization removes UTM parameters."""
    url = "https://example.com/page?utm_source=slack&utm_campaign=test&id=123"
    normalized = link_extractor.normalize_url(url)
    assert "utm_source" not in normalized
    assert "utm_campaign" not in normalized
    assert "id=123" in normalized


def test_normalize_url_removes_fragment() -> None:
    """Test URL normalization removes fragments."""
    url = "https://example.com/page#section"
    normalized = link_extractor.normalize_url(url)
    assert "#section" not in normalized


def test_normalize_url_adds_scheme() -> None:
    """Test URL normalization adds scheme to www URLs."""
    url = "www.example.com/page"
    normalized = link_extractor.normalize_url(url)
    assert normalized.startswith("https://")


def test_normalize_url_removes_trailing_slash() -> None:
    """Test URL normalization removes trailing slash."""
    url = "https://example.com/page/"
    normalized = link_extractor.normalize_url(url)
    assert not normalized.endswith("/")


def test_normalize_url_keeps_root_slash() -> None:
    """Test URL normalization keeps root slash."""
    url = "https://example.com/"
    normalized = link_extractor.normalize_url(url)
    assert normalized == "https://example.com/"


def test_extract_jira_keys() -> None:
    """Test Jira key extraction."""
    text = "Fixed PROJ-123 and TEAM-456 issues"
    keys = link_extractor.extract_jira_keys(text)
    assert len(keys) == 2
    assert "PROJ-123" in keys
    assert "TEAM-456" in keys


def test_extract_jira_keys_none() -> None:
    """Test Jira key extraction with no keys."""
    text = "No Jira keys here"
    keys = link_extractor.extract_jira_keys(text)
    assert len(keys) == 0


def test_extract_github_issues() -> None:
    """Test GitHub issue extraction."""
    text = "See github.com/owner/repo/issues/42 and github.com/org/project#123"
    issues = link_extractor.extract_github_issues(text)
    assert len(issues) >= 1
    assert any("owner/repo#42" in issue for issue in issues)


def test_extract_gitlab_issues() -> None:
    """Test GitLab issue extraction."""
    text = "Check gitlab.com/group/project/-/issues/10"
    issues = link_extractor.extract_gitlab_issues(text)
    assert len(issues) >= 1
    assert any("#10" in issue for issue in issues)


def test_extract_meeting_links_zoom() -> None:
    """Test Zoom meeting link extraction."""
    text = "Join zoom.us/j/123456789"
    meetings = link_extractor.extract_meeting_links(text)
    assert len(meetings) >= 1
    assert any("zoom:" in m for m in meetings)


def test_extract_meeting_links_google_meet() -> None:
    """Test Google Meet link extraction."""
    text = "Join meet.google.com/abc-defg-hij"
    meetings = link_extractor.extract_meeting_links(text)
    assert len(meetings) >= 1
    assert any("meet:" in m for m in meetings)


def test_extract_document_anchors_google_docs() -> None:
    """Test Google Docs anchor extraction."""
    text = "See docs.google.com/document/d/1234567890abcdefghijklmnopq"
    anchors = link_extractor.extract_document_anchors(text)
    assert len(anchors) >= 1
    assert any("gdoc:" in a for a in anchors)


def test_extract_all_anchors_mixed() -> None:
    """Test extracting all anchor types."""
    text = """
    Fixed PROJ-123 in github.com/org/repo/issues/42
    See docs.google.com/document/d/abc123def456ghi789jkl012mno345
    Join meet.google.com/abc-defg-hij
    """
    anchors = link_extractor.extract_all_anchors(text)

    # Should have Jira, GitHub, Google Doc, and Meet anchors
    assert len(anchors) >= 4
    assert any("PROJ-123" in a for a in anchors)
    assert any("org/repo#42" in a for a in anchors)
    assert any("gdoc:" in a for a in anchors)
    assert any("meet:" in a for a in anchors)


def test_extract_all_anchors_deduplicated() -> None:
    """Test anchor deduplication."""
    text = "PROJ-123 and PROJ-123 again"
    anchors = link_extractor.extract_all_anchors(text)
    assert len([a for a in anchors if a == "PROJ-123"]) == 1


def test_normalize_links() -> None:
    """Test normalizing list of links."""
    urls = [
        "https://example.com?utm_source=x",
        "www.test.com",
        "https://site.com/path/",
    ]
    normalized = link_extractor.normalize_links(urls)

    assert len(normalized) == 3
    assert "utm_source" not in normalized[0]
    assert normalized[1].startswith("https://")
    assert not normalized[2].endswith("/")
