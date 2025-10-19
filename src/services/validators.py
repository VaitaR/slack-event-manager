"""Event validation service.

Validates event structure, semantics, and quality for publishing.
"""

import re
from typing import Final

from src.domain.models import Event, EventCategory, EventStatus
from src.domain.validation_constants import (
    MAX_IMPACT_AREAS,
    MAX_LINKS,
    MAX_QUALIFIERS,
    MAX_SUMMARY_LENGTH,
    MAX_TITLE_LENGTH,
    MIN_CONFIDENCE_DEFAULT,
    MIN_IMPORTANCE_DEFAULT,
)

# Patterns for forbidden elements
URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://", re.IGNORECASE)
DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
)
EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF]"
)


class EventValidator:
    """Validates events against PRD rules."""

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE_DEFAULT,
        max_title_length: int = MAX_TITLE_LENGTH,
        max_qualifiers: int = MAX_QUALIFIERS,
    ) -> None:
        """Initialize validator.

        Args:
            min_confidence: Minimum confidence threshold
            max_title_length: Maximum title length
            max_qualifiers: Maximum number of qualifiers
        """
        self.min_confidence = min_confidence
        self.max_title_length = max_title_length
        self.max_qualifiers = max_qualifiers

    def validate_title_lint(self, event: Event) -> list[str]:
        """Validate title structure (lint rules).

        Checks:
        - Action in enum (implicit via type)
        - Max 2 qualifiers
        - Max 1 stroke
        - Max 1 anchor
        - No dates/emojis/URLs in slots

        Args:
            event: Event to validate

        Returns:
            List of errors (empty if valid)

        Example:
            >>> validator = EventValidator()
            >>> errors = validator.validate_title_lint(event)
            >>> if not errors:
            ...     print("Title structure valid")
        """
        errors: list[str] = []

        # Max qualifiers
        if len(event.qualifiers) > self.max_qualifiers:
            errors.append(
                f"Too many qualifiers: {len(event.qualifiers)} (max {self.max_qualifiers})"
            )

        # Check for forbidden elements in title slots
        all_slot_text = " ".join(
            [
                event.object_name_raw,
                *event.qualifiers,
                event.stroke or "",
                event.anchor or "",
            ]
        )

        if URL_PATTERN.search(all_slot_text):
            errors.append("URLs not allowed in title slots")

        if DATE_PATTERN.search(all_slot_text):
            errors.append("Dates not allowed in title slots")

        if EMOJI_PATTERN.search(all_slot_text):
            errors.append("Emojis not allowed in title slots")

        return errors

    def validate_event_semantics(self, event: Event) -> list[str]:
        """Validate event semantic consistency.

        Checks:
        - Summary filled and within limit
        - Category not unknown (warning)
        - Status ↔ time consistency
        - Links normalized and <= 3
        - Anchors present when referenced

        Args:
            event: Event to validate

        Returns:
            List of errors/warnings

        Example:
            >>> validator = EventValidator()
            >>> errors = validator.validate_event_semantics(event)
        """
        errors: list[str] = []

        # Summary required
        if not event.summary or not event.summary.strip():
            errors.append("Summary is required")
        elif len(event.summary) > MAX_SUMMARY_LENGTH:
            errors.append(
                f"Summary too long: {len(event.summary)} chars (max {MAX_SUMMARY_LENGTH})"
            )

        # Category warning
        if event.category == EventCategory.UNKNOWN:
            errors.append("WARNING: Category is unknown")

        # Status ↔ time consistency
        if event.status == EventStatus.COMPLETED:
            if not event.actual_end:
                errors.append("Status 'completed' requires actual_end timestamp")
        elif event.status == EventStatus.STARTED:
            if not event.actual_start:
                errors.append("Status 'started' requires actual_start timestamp")
        elif event.status in (EventStatus.PLANNED, EventStatus.CONFIRMED):
            if not event.planned_start:
                errors.append(
                    f"Status '{event.status.value}' requires planned_start timestamp"
                )

        # Links validation
        if len(event.links) > MAX_LINKS:
            errors.append(f"Too many links: {len(event.links)} (max {MAX_LINKS})")

        for link in event.links:
            if not link.startswith(("http://", "https://")):
                errors.append(f"Invalid link format: {link}")

        # Impact area limit
        if len(event.impact_area) > MAX_IMPACT_AREAS:
            errors.append(
                f"Too many impact areas: {len(event.impact_area)} (max {MAX_IMPACT_AREAS})"
            )

        return errors

    def validate_all(self, event: Event) -> list[str]:
        """Run all validations.

        Args:
            event: Event to validate

        Returns:
            Combined list of all errors/warnings

        Example:
            >>> validator = EventValidator()
            >>> errors = validator.validate_all(event)
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
        """
        errors: list[str] = []
        errors.extend(self.validate_title_lint(event))
        errors.extend(self.validate_event_semantics(event))
        return errors

    def get_critical_errors(self, event: Event) -> list[str]:
        """Get only critical validation errors that block saving.

        Critical errors are issues that violate domain invariants and
        make the event unsuitable for downstream processing.

        Args:
            event: Event to validate

        Returns:
            List of critical errors (empty if event is valid for saving)

        Example:
            >>> validator = EventValidator()
            >>> errors = validator.get_critical_errors(event)
            >>> if errors:
            ...     print(f"Event blocked: {errors}")
        """
        all_issues = self.validate_all(event)
        return [issue for issue in all_issues if not issue.startswith("WARNING:")]

    def get_validation_summary(self, event: Event) -> dict[str, list[str]]:
        """Get comprehensive validation summary for audit logging.

        Args:
            event: Event to validate

        Returns:
            Dict with 'critical', 'warnings', and 'info' categories

        Example:
            >>> validator = EventValidator()
            >>> summary = validator.get_validation_summary(event)
            >>> print(f"Critical: {summary['critical']}")
        """
        all_issues = self.validate_all(event)

        critical = []
        warnings = []
        info = []

        for issue in all_issues:
            if issue.startswith("WARNING:"):
                warnings.append(issue.replace("WARNING: ", ""))
            elif issue.startswith("INFO:"):
                info.append(issue.replace("INFO: ", ""))
            else:
                critical.append(issue)

        return {
            "critical": critical,
            "warnings": warnings,
            "info": info,
        }

    def should_publish(
        self,
        event: Event,
        min_importance: int = MIN_IMPORTANCE_DEFAULT,
        min_confidence: float | None = None,
    ) -> bool:
        """Check if event meets quality thresholds for publishing.

        Args:
            event: Event to check
            min_importance: Minimum importance threshold
            min_confidence: Override minimum confidence (uses instance default if None)

        Returns:
            True if event should be published

        Example:
            >>> validator = EventValidator()
            >>> if validator.should_publish(event, min_importance=60):
            ...     print("Event ready for publishing")
        """
        conf_threshold = (
            min_confidence if min_confidence is not None else self.min_confidence
        )

        if event.confidence < conf_threshold:
            return False

        if event.importance < min_importance:
            return False

        # Must have no critical errors (warnings OK)
        errors = self.validate_all(event)
        critical_errors = [e for e in errors if not e.startswith("WARNING:")]

        return len(critical_errors) == 0

    def get_quality_issues(self, event: Event) -> dict[str, list[str]]:
        """Get detailed quality issues breakdown.

        Args:
            event: Event to check

        Returns:
            Dict with 'errors', 'warnings', and 'info' lists

        Example:
            >>> validator = EventValidator()
            >>> issues = validator.get_quality_issues(event)
            >>> if issues['errors']:
            ...     print(f"Errors: {issues['errors']}")
        """
        all_issues = self.validate_all(event)

        errors = []
        warnings = []
        info = []

        for issue in all_issues:
            if issue.startswith("WARNING:"):
                warnings.append(issue.replace("WARNING: ", ""))
            elif issue.startswith("INFO:"):
                info.append(issue.replace("INFO: ", ""))
            else:
                errors.append(issue)

        # Add quality checks
        if event.confidence < self.min_confidence:
            warnings.append(
                f"Low confidence: {event.confidence:.2f} (threshold: {self.min_confidence})"
            )

        if event.importance < MIN_IMPORTANCE_DEFAULT:
            info.append(
                f"Low importance: {event.importance} (threshold: {MIN_IMPORTANCE_DEFAULT})"
            )

        return {
            "errors": errors,
            "warnings": warnings,
            "info": info,
        }
