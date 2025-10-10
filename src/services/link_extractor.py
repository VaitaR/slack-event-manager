"""Link and anchor extraction service.

Extracts URLs and identifies anchors (Jira, GitHub, meeting links, etc.)
"""

import re
from typing import Final
from urllib.parse import urlparse, urlunparse

# Anchor patterns for issue trackers
JIRA_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{2,10}-\d+)\b")
"""Pattern to match Jira ticket IDs (e.g., PROJ-123, ABC-4567)."""

GITHUB_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)(?:/issues/|/pull/|#)(\d+)"
)
"""Pattern to match GitHub issues and PRs (e.g., org/repo#123, org/repo/issues/456)."""

GITLAB_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"gitlab\.com/([a-zA-Z0-9_/-]+)(?:/issues/|/merge_requests/|-/issues/|-/merge_requests/)(\d+)"
)
"""Pattern to match GitLab issues and MRs."""

# Meeting link patterns
ZOOM_PATTERN: Final[re.Pattern[str]] = re.compile(r"zoom\.us/[^\s<>\"']+")
"""Pattern to match Zoom meeting URLs."""

MEET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"meet\.google\.com/[a-z]{3}-[a-z]{4}-[a-z]{3}"
)
"""Pattern to match Google Meet links (e.g., meet.google.com/abc-defg-hij)."""

TEAMS_PATTERN: Final[re.Pattern[str]] = re.compile(r"teams\.microsoft\.com/[^\s<>\"']+")
"""Pattern to match Microsoft Teams meeting links."""

# Document link patterns
CONFLUENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"confluence\.[a-z0-9.-]+/(?:pages/viewpage\.action\?pageId=|display/[^/]+/)([0-9]+)"
)
"""Pattern to match Confluence page links."""

NOTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"notion\.so/[^\s<>\"']*-([a-f0-9]{32})"
)
"""Pattern to match Notion page links (with 32-char hex ID)."""

GDOCS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"docs\.google\.com/[^\s<>\"']+/d/([a-zA-Z0-9_-]{25,})"
)
"""Pattern to match Google Docs links."""

# Generic URL pattern
URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", flags=re.IGNORECASE
)
"""Pattern to match HTTP/HTTPS URLs and www. domains."""


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text.

    Args:
        text: Message text

    Returns:
        List of URLs found

    Example:
        >>> extract_urls("Check https://example.com and www.test.com")
        ['https://example.com', 'www.test.com']
    """
    urls = URL_PATTERN.findall(text)

    # Remove trailing punctuation
    cleaned_urls = []
    for url in urls:
        url = url.rstrip(",.;:!?)")
        # Handle Slack's <url|text> format
        if "|" in url:
            url = url.split("|")[0].lstrip("<")
        cleaned_urls.append(url)

    return cleaned_urls


def normalize_url(url: str) -> str:
    """Normalize URL by removing utm parameters and fragments.

    Args:
        url: Raw URL

    Returns:
        Normalized URL (scheme + host + path)

    Example:
        >>> normalize_url("https://example.com/page?utm_source=x&id=1#section")
        'https://example.com/page?id=1'
    """
    try:
        # Add scheme if missing
        if url.startswith("www."):
            url = f"https://{url}"

        parsed = urlparse(url)

        # Remove utm_* parameters
        if parsed.query:
            params = [
                param
                for param in parsed.query.split("&")
                if not param.startswith("utm_")
            ]
            clean_query = "&".join(params)
        else:
            clean_query = ""

        # Reconstruct without fragment
        normalized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                "",  # No fragment
            )
        )

        # Remove trailing slash unless it's the root
        if normalized.endswith("/") and parsed.path != "/":
            normalized = normalized.rstrip("/")

        return normalized
    except Exception:
        # Return original on parsing error
        return url


def extract_jira_keys(text: str) -> list[str]:
    """Extract Jira issue keys.

    Args:
        text: Message text

    Returns:
        List of Jira keys (e.g., ["PROJ-123"])

    Example:
        >>> extract_jira_keys("Fixed PROJ-123 and TEAM-456")
        ['PROJ-123', 'TEAM-456']
    """
    return JIRA_PATTERN.findall(text)


def extract_github_issues(text: str) -> list[str]:
    """Extract GitHub issue/PR references.

    Args:
        text: Message text

    Returns:
        List of GitHub references (e.g., ["owner/repo#123"])

    Example:
        >>> extract_github_issues("See github.com/org/project/issues/42")
        ['org/project#42']
    """
    matches = GITHUB_PATTERN.findall(text)
    return [f"{repo}#{issue}" for repo, issue in matches]


def extract_gitlab_issues(text: str) -> list[str]:
    """Extract GitLab issue/MR references.

    Args:
        text: Message text

    Returns:
        List of GitLab references

    Example:
        >>> extract_gitlab_issues("gitlab.com/group/proj/-/issues/10")
        ['group/proj#10']
    """
    matches = GITLAB_PATTERN.findall(text)
    return [f"{repo}#{issue}" for repo, issue in matches]


def extract_meeting_links(text: str) -> list[str]:
    """Extract meeting link anchors.

    Args:
        text: Message text

    Returns:
        List of meeting identifiers

    Example:
        >>> extract_meeting_links("Join meet.google.com/abc-defg-hij")
        ['meet:abc-defg-hij']
    """
    anchors = []

    # Zoom
    zoom_matches = ZOOM_PATTERN.findall(text)
    for match in zoom_matches:
        anchors.append(f"zoom:{match}")

    # Google Meet
    meet_matches = MEET_PATTERN.findall(text)
    for match in meet_matches:
        # Extract meeting code
        code = match.replace("meet.google.com/", "")
        anchors.append(f"meet:{code}")

    # Teams
    teams_matches = TEAMS_PATTERN.findall(text)
    if teams_matches:
        anchors.append("teams:meeting")

    return anchors


def extract_document_anchors(text: str) -> list[str]:
    """Extract document ID anchors (Confluence, Notion, Google Docs).

    Args:
        text: Message text

    Returns:
        List of document anchors

    Example:
        >>> extract_document_anchors("docs.google.com/document/d/abc123def")
        ['gdoc:abc123def']
    """
    anchors = []

    # Confluence page IDs
    conf_matches = CONFLUENCE_PATTERN.findall(text)
    for page_id in conf_matches:
        anchors.append(f"confluence:{page_id}")

    # Notion page IDs
    notion_matches = NOTION_PATTERN.findall(text)
    for page_id in notion_matches:
        anchors.append(f"notion:{page_id}")

    # Google Docs IDs
    gdoc_matches = GDOCS_PATTERN.findall(text)
    for doc_id in gdoc_matches:
        anchors.append(f"gdoc:{doc_id}")

    return anchors


def extract_all_anchors(text: str) -> list[str]:
    """Extract all anchor types from text.

    Args:
        text: Message text

    Returns:
        Deduplicated list of all anchors

    Example:
        >>> extract_all_anchors("Fixed PROJ-123, see github.com/org/repo/issues/5")
        ['PROJ-123', 'org/repo#5']
    """
    anchors: list[str] = []

    # Jira keys
    anchors.extend(extract_jira_keys(text))

    # GitHub issues
    anchors.extend(extract_github_issues(text))

    # GitLab issues
    anchors.extend(extract_gitlab_issues(text))

    # Meeting links
    anchors.extend(extract_meeting_links(text))

    # Document IDs
    anchors.extend(extract_document_anchors(text))

    # Deduplicate while preserving order
    seen = set()
    unique_anchors = []
    for anchor in anchors:
        if anchor not in seen:
            seen.add(anchor)
            unique_anchors.append(anchor)

    return unique_anchors


def normalize_links(urls: list[str]) -> list[str]:
    """Normalize a list of URLs.

    Args:
        urls: List of raw URLs

    Returns:
        List of normalized URLs

    Example:
        >>> normalize_links(["https://example.com?utm_source=x", "www.test.com"])
        ['https://example.com', 'https://www.test.com']
    """
    return [normalize_url(url) for url in urls]
