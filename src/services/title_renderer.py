"""Title renderer service for generating canonical event titles.

System is single source of truth - titles are rendered from slots, never stored.
"""

import re
from typing import Final

from src.domain.models import Event, EventCategory, Severity
from src.domain.validation_constants import MAX_QUALIFIERS

# Title format: <Action>: <Object> — <Qualifier>[, <Stroke>] (<Anchor>)
# Max length constraints
MAX_TITLE_LENGTH: Final[int] = 140
TARGET_TITLE_LENGTH: Final[int] = 110

# Patterns to detect forbidden elements
URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://", re.IGNORECASE)
DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
)
EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF]"
)


class TitleRenderer:
    """Renders canonical titles from event slots."""

    def render_canonical_title(
        self, event: Event, format_style: str = "canonical"
    ) -> str:
        """Render canonical title from event slots.

        Format styles:
        - canonical: <Action>: <Object> — <Qualifier>[, <Stroke>] (<Anchor>)
        - object_first: <Object>: <Action> — <Qualifier>[, <Stroke>] (<Anchor>)
        - severity_first: SEV-X: <event> — <object>[, stroke]

        Args:
            event: Event with title slots
            format_style: Title format style

        Returns:
            Rendered canonical title

        Example:
            >>> renderer = TitleRenderer()
            >>> event = Event(action=ActionType.LAUNCH, ...)
            >>> renderer.render_canonical_title(event)
            'Launch: Stocks & ETFs — alpha, Wallet team (INV-1024)'
        """
        if format_style == "severity_first" and event.category == EventCategory.RISK:
            return self._render_severity_first(event)
        elif format_style == "object_first":
            return self._render_object_first(event)
        else:
            return self._render_canonical(event)

    def _render_canonical(self, event: Event) -> str:
        """Render canonical format: <Action>: <Object> — <Qualifier>[, <Stroke>] (<Anchor>)."""
        parts: list[str] = []

        # Action: Object
        parts.append(f"{event.action.value}: {event.object_name_raw}")

        # — Qualifiers
        if event.qualifiers:
            qualifiers_str = ", ".join(event.qualifiers[:2])  # Max 2
            parts.append(f" — {qualifiers_str}")

        # , Stroke
        if event.stroke:
            parts.append(f", {event.stroke}")

        # (Anchor)
        if event.anchor:
            parts.append(f" ({event.anchor})")

        title = "".join(parts)

        # Truncate if too long
        if len(title) > MAX_TITLE_LENGTH:
            title = self._truncate_title(event)

        return title

    def _render_object_first(self, event: Event) -> str:
        """Render object-first format: <Object>: <Action> — <Qualifier>[, <Stroke>] (<Anchor>)."""
        parts: list[str] = []

        # Object: Action
        parts.append(f"{event.object_name_raw}: {event.action.value}")

        # — Qualifiers
        if event.qualifiers:
            qualifiers_str = ", ".join(event.qualifiers[:2])
            parts.append(f" — {qualifiers_str}")

        # , Stroke
        if event.stroke:
            parts.append(f", {event.stroke}")

        # (Anchor)
        if event.anchor:
            parts.append(f" ({event.anchor})")

        title = "".join(parts)

        if len(title) > MAX_TITLE_LENGTH:
            title = self._truncate_title(event, object_first=True)

        return title

    def _render_severity_first(self, event: Event) -> str:
        """Render severity-first for risk: SEV-X: <event> — <object>[, stroke]."""
        parts: list[str] = []

        # SEV-X
        if event.severity and event.severity != Severity.UNKNOWN:
            sev_label = event.severity.value.upper()
            parts.append(f"{sev_label}: ")
        else:
            parts.append("INCIDENT: ")

        # Event (action)
        parts.append(event.action.value.lower())

        # — Object
        parts.append(f" — {event.object_name_raw}")

        # , Stroke
        if event.stroke:
            parts.append(f", {event.stroke}")

        title = "".join(parts)

        if len(title) > MAX_TITLE_LENGTH:
            # For severity format, drop stroke if too long
            title = f"{parts[0]}{event.action.value.lower()} — {event.object_name_raw}"

        return title

    def _truncate_title(self, event: Event, object_first: bool = False) -> str:
        """Truncate title by dropping elements in priority order.

        Priority: stroke → 2nd qualifier → anchor

        Args:
            event: Event with title slots
            object_first: Use object-first format

        Returns:
            Truncated title <= 140 chars
        """
        # Try without stroke
        parts: list[str] = []

        if object_first:
            parts.append(f"{event.object_name_raw}: {event.action.value}")
        else:
            parts.append(f"{event.action.value}: {event.object_name_raw}")

        if event.qualifiers:
            qualifiers_str = ", ".join(event.qualifiers[:2])
            parts.append(f" — {qualifiers_str}")

        if event.anchor:
            parts.append(f" ({event.anchor})")

        title = "".join(parts)
        if len(title) <= MAX_TITLE_LENGTH:
            return title

        # Try without 2nd qualifier
        parts = []
        if object_first:
            parts.append(f"{event.object_name_raw}: {event.action.value}")
        else:
            parts.append(f"{event.action.value}: {event.object_name_raw}")

        if event.qualifiers:
            parts.append(f" — {event.qualifiers[0]}")  # Only first qualifier

        if event.anchor:
            parts.append(f" ({event.anchor})")

        title = "".join(parts)
        if len(title) <= MAX_TITLE_LENGTH:
            return title

        # Try without anchor
        parts = []
        if object_first:
            parts.append(f"{event.object_name_raw}: {event.action.value}")
        else:
            parts.append(f"{event.action.value}: {event.object_name_raw}")

        if event.qualifiers:
            parts.append(f" — {event.qualifiers[0]}")

        title = "".join(parts)
        if len(title) <= MAX_TITLE_LENGTH:
            return title

        # Last resort: just action + object, truncate object if needed
        if object_first:
            base = f"{event.object_name_raw}: {event.action.value}"
        else:
            base = f"{event.action.value}: {event.object_name_raw}"

        if len(base) > MAX_TITLE_LENGTH:
            # Truncate object name
            max_object_len = MAX_TITLE_LENGTH - len(event.action.value) - 4
            if object_first:
                return (
                    f"{event.object_name_raw[:max_object_len]}...: {event.action.value}"
                )
            else:
                return (
                    f"{event.action.value}: {event.object_name_raw[:max_object_len]}..."
                )

        return base

    def validate_title_structure(self, event: Event) -> list[str]:
        """Validate event title structure against PRD rules.

        Checks:
        - Action in enum
        - Max 2 qualifiers
        - Max 1 stroke
        - Max 1 anchor
        - Length <= 140
        - No dates/emojis/URLs in slots

        Args:
            event: Event to validate

        Returns:
            List of validation errors (empty if valid)

        Example:
            >>> renderer = TitleRenderer()
            >>> errors = renderer.validate_title_structure(event)
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
        """
        errors: list[str] = []

        # Max qualifiers
        if len(event.qualifiers) > MAX_QUALIFIERS:
            errors.append(
                f"Too many qualifiers: {len(event.qualifiers)} (max {MAX_QUALIFIERS})"
            )

        # Check for forbidden elements in slots
        all_text = " ".join(
            [
                event.object_name_raw,
                *event.qualifiers,
                event.stroke or "",
                event.anchor or "",
            ]
        )

        if URL_PATTERN.search(all_text):
            errors.append("URLs not allowed in title slots")

        if DATE_PATTERN.search(all_text):
            errors.append("Dates not allowed in title slots")

        if EMOJI_PATTERN.search(all_text):
            errors.append("Emojis not allowed in title slots")

        # Render and check length
        rendered = self.render_canonical_title(event)
        if len(rendered) > MAX_TITLE_LENGTH:
            errors.append(
                f"Rendered title too long: {len(rendered)} chars (max {MAX_TITLE_LENGTH})"
            )

        return errors
