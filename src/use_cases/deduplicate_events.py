"""Deduplicate events use case.

Applies deduplication rules to merge similar events.
"""

from datetime import datetime, timedelta
from time import perf_counter

import pytz

from src.adapters.query_builders import EventQueryCriteria
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.domain.models import DeduplicationResult, Event, MessageSource
from src.domain.protocols import RepositoryProtocol
from src.observability.metrics import PIPELINE_STAGE_DURATION_SECONDS
from src.observability.tracing import correlation_scope
from src.services import deduplicator
from src.services.title_renderer import TitleRenderer
from src.services.validators import EventValidator

_TITLE_RENDERER = TitleRenderer()
_EVENT_VALIDATOR = EventValidator()
logger = get_logger(__name__)


def deduplicate_events_use_case(
    repository: RepositoryProtocol,
    settings: Settings,
    lookback_days: int = 7,
    source_id: MessageSource | None = None,
    *,
    correlation_id: str | None = None,
) -> DeduplicationResult:
    """Deduplicate events within lookback window.

    1. Fetch all events from lookback window (optionally filtered by source)
    2. For each event:
       a. Check: same message_id with other events? → no merge (Rule 1)
       b. Find merge candidates: anchor/link overlap + date Δ
       c. Fuzzy match title (rapidfuzz ≥0.8)
       d. If merge: combine attributes, increment version
       e. Generate dedup_key
       f. Upsert to events table
    3. Return counts

    Args:
        repository: Repository protocol implementation
        settings: Application settings
        lookback_days: Days to look back for deduplication
        source_id: Optional source filter for strict isolation (prevents cross-source merging)

    Returns:
        DeduplicationResult with counts

    Example:
        >>> # Deduplicate all events
        >>> result = deduplicate_events_use_case(repo, settings)
        >>> result.merged_events
        3

        >>> # Deduplicate only Slack events (strict isolation)
        >>> result = deduplicate_events_use_case(repo, settings, source_id=MessageSource.SLACK)
    """
    with correlation_scope(correlation_id) as bound_correlation_id:
        stage_start = perf_counter()
        result: DeduplicationResult | None = None
        try:
            now = datetime.utcnow().replace(tzinfo=pytz.UTC)
            extracted_after = now - timedelta(days=lookback_days)

            criteria = EventQueryCriteria(
                extracted_after=extracted_after,
                source_id=source_id.value if source_id else None,
                order_by="extracted_at",
                order_desc=False,
            )

            all_events = repository.query_events(criteria)
            initial_count = len(all_events)

            logger.info(
                "deduplication_started",
                correlation_id=bound_correlation_id,
                event_count=initial_count,
                source=source_id.value if source_id else None,
                lookback_days=lookback_days,
                title_similarity=settings.dedup_title_similarity,
                date_window_hours=settings.dedup_date_window_hours,
            )

            if not all_events:
                result = DeduplicationResult(
                    new_events=0, merged_events=0, total_events=0
                )
                return result

            deduplicated_events = deduplicator.deduplicate_event_list(
                all_events,
                date_window_hours=settings.dedup_date_window_hours,
                title_similarity_threshold=settings.dedup_title_similarity,
                title_renderer=_TITLE_RENDERER,
            )

            final_count = len(deduplicated_events)
            merged_count = initial_count - final_count

            if deduplicated_events:
                validated_events: list[Event] = []

                for event in deduplicated_events:
                    issues = _EVENT_VALIDATOR.validate_all(event)

                    if issues:
                        logger.warning(
                            "deduplication_validation_issue",
                            correlation_id=bound_correlation_id,
                            message_id=event.message_id,
                            issues=issues,
                        )

                    validated_events.append(event)

                repository.save_events(validated_events)

            result = DeduplicationResult(
                new_events=final_count,
                merged_events=merged_count,
                total_events=final_count,
            )
            return result
        finally:
            duration = perf_counter() - stage_start
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="dedup").observe(duration)
            if result is not None:
                logger.info(
                    "deduplication_finished",
                    correlation_id=bound_correlation_id,
                    duration_seconds=duration,
                    new_events=result.new_events,
                    merged_events=result.merged_events,
                    total_events=result.total_events,
                )
