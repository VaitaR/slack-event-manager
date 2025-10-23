"""Extract events use case.

Uses LLM to extract structured events from candidate messages.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from src.adapters.llm_client import LLMClient
from src.adapters.query_builders import CandidateQueryCriteria
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.domain.exceptions import BudgetExceededError, LLMAPIError, ValidationError
from src.domain.models import (
    ActionType,
    CandidateStatus,
    ChangeType,
    Environment,
    Event,
    EventStatus,
    ExtractionResult,
    MessageSource,
    Severity,
    TimeSource,
)
from src.domain.protocols import RepositoryProtocol
from src.domain.validation_constants import (
    MAX_IMPACT_AREAS,
    MAX_LINKS,
    MAX_QUALIFIERS,
)
from src.services import deduplicator
from src.services.importance_scorer import ImportanceScorer
from src.services.object_registry import ObjectRegistry
from src.services.validators import EventValidator

logger = get_logger(__name__)

# Initialize services (singleton-style)
_object_registry: ObjectRegistry | None = None
_importance_scorer: ImportanceScorer | None = None
_event_validator: EventValidator | None = None


def _get_object_registry() -> ObjectRegistry:
    """Get or create ObjectRegistry instance."""
    global _object_registry
    if _object_registry is None:
        from src.config.settings import get_settings

        settings = get_settings()
        registry_path = Path(settings.object_registry_path)

        if not registry_path.exists():
            # Graceful fallback if file doesn't exist
            logger.warning(
                "object_registry_missing",
                path=str(registry_path),
            )
            # Try fallback to example file
            fallback_path = Path("config/defaults/object_registry.example.yaml")
            if fallback_path.exists():
                registry_path = fallback_path
                logger.info("object_registry_fallback", path=str(fallback_path))
            else:
                # Create minimal empty registry
                logger.warning("object_registry_unavailable")

        _object_registry = ObjectRegistry(registry_path)
    return _object_registry


def _get_importance_scorer() -> ImportanceScorer:
    """Get or create ImportanceScorer instance."""
    global _importance_scorer
    if _importance_scorer is None:
        _importance_scorer = ImportanceScorer()
    return _importance_scorer


def _get_event_validator() -> EventValidator:
    """Get or create EventValidator instance."""
    global _event_validator
    if _event_validator is None:
        _event_validator = EventValidator()
    return _event_validator


def _parse_datetime(
    dt_str: str | None, fallback: datetime | None = None
) -> datetime | None:
    """Parse ISO8601 datetime string with fallback."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except (ValueError, AttributeError):
        return fallback


def convert_llm_event_to_domain(
    llm_event: Any,
    message_id: str,
    message_ts_dt: datetime,
    channel_name: str,
    reaction_count: int = 0,
    mention_count: int = 0,
) -> Event:
    """Convert LLM event to domain Event model with new comprehensive structure.

    Uses ObjectRegistry, ImportanceScorer, and deduplicator services.

    Args:
        llm_event: LLM event object
        message_id: Source message ID
        message_ts_dt: Message timestamp for date fallback
        channel_name: Channel name
        reaction_count: Reaction count from message (for importance)
        mention_count: Mention count from message (for importance)

    Returns:
        Domain Event object with all new fields
    """
    # Get services
    object_registry = _get_object_registry()
    importance_scorer = _get_importance_scorer()

    # Parse all time fields
    planned_start = _parse_datetime(llm_event.planned_start, None)
    planned_end = _parse_datetime(llm_event.planned_end, None)
    actual_start = _parse_datetime(llm_event.actual_start, None)
    actual_end = _parse_datetime(llm_event.actual_end, None)

    # Canonicalize object_id using registry
    object_id = object_registry.canonicalize_object(llm_event.object_name_raw)

    # Parse enums
    try:
        action = ActionType(llm_event.action)
    except ValueError:
        action = ActionType.OTHER

    try:
        status = EventStatus(llm_event.status)
    except ValueError:
        status = EventStatus.UPDATED

    try:
        change_type = ChangeType(llm_event.change_type)
    except ValueError:
        change_type = ChangeType.OTHER

    try:
        environment = Environment(llm_event.environment)
    except ValueError:
        environment = Environment.UNKNOWN

    severity = None
    if llm_event.severity:
        try:
            severity = Severity(llm_event.severity)
        except ValueError:
            pass

    try:
        time_source = TimeSource(llm_event.time_source)
    except ValueError:
        time_source = TimeSource.TS_FALLBACK

    # Create event (importance will be calculated below)
    event = Event(
        # Identification
        message_id=message_id,
        source_channels=[channel_name] if channel_name else [],
        # Title slots
        action=action,
        object_id=object_id,
        object_name_raw=llm_event.object_name_raw,
        qualifiers=llm_event.qualifiers[:MAX_QUALIFIERS],
        stroke=llm_event.stroke,
        anchor=llm_event.anchor,
        # Classification
        category=llm_event.category,
        status=status,
        change_type=change_type,
        environment=environment,
        severity=severity,
        # Time fields
        planned_start=planned_start,
        planned_end=planned_end,
        actual_start=actual_start,
        actual_end=actual_end,
        time_source=time_source,
        time_confidence=llm_event.time_confidence,
        # Content
        summary=llm_event.summary,
        why_it_matters=llm_event.why_it_matters,
        links=llm_event.links[:MAX_LINKS],
        anchors=llm_event.anchors,
        impact_area=llm_event.impact_area[:MAX_IMPACT_AREAS],
        impact_type=llm_event.impact_type,
        # Quality (importance calculated below)
        confidence=llm_event.confidence,
        importance=0,  # Calculated below
        # Clustering (generated below)
        cluster_key="",
        dedup_key="",
        relations=[],
    )

    # Generate cluster_key and dedup_key
    event.cluster_key = deduplicator.generate_cluster_key(event)
    event.dedup_key = deduplicator.generate_dedup_key(event)

    # Calculate importance score
    importance_result = importance_scorer.calculate_importance(
        event,
        llm_score=None,  # TODO: Add LLM importance scoring if needed
        reaction_count=reaction_count,
        mention_count=mention_count,
        is_duplicate=False,
    )
    event.importance = importance_result.final_score

    return event


def extract_events_use_case(
    llm_client: LLMClient,
    repository: RepositoryProtocol,
    settings: Settings,
    source_id: MessageSource | None = None,
    batch_size: int | None = 50,
    check_budget: bool = True,
) -> ExtractionResult:
    """Extract events from candidates using LLM.

    1. Fetch candidates with status='new', order by score DESC
    2. Check LLM budget remaining
    3. If budget low, filter to P90+ score only
    4. For each candidate:
       a. Build prompt (text + top 3 links)
       b. Call LLM (with cache check)
       c. Validate JSON response
       d. Parse events array (0 to K)
       e. Resolve dates/times to UTC
       f. Save to staging with (message_id, idx)
       g. Update status = 'llm_ok' | 'llm_fail'
    5. Save LLM call metadata

    Args:
        llm_client: LLM client
        repository: Repository protocol implementation
        settings: Application settings
        source_id: Filter candidates by message source (None = all sources)
        batch_size: Max candidates to process
        check_budget: Whether to enforce budget limits

    Returns:
        ExtractionResult with counts and cost

    Example:
        >>> result = extract_events_use_case(llm, repo, settings)
        >>> result.events_extracted
        45
    """
    # Check budget
    min_score = None
    if check_budget:
        today = datetime.utcnow().replace(tzinfo=pytz.UTC)
        daily_cost = repository.get_daily_llm_cost(today)

        if daily_cost >= settings.llm_daily_budget_usd:
            raise BudgetExceededError(
                f"Daily budget ${settings.llm_daily_budget_usd} exceeded: ${daily_cost:.2f}"
            )

        remaining_budget = settings.llm_daily_budget_usd - daily_cost

        # If budget is low (< 20%), only process high-score candidates
        if remaining_budget < settings.llm_daily_budget_usd * 0.2:
            # Calculate P90 score using CandidateQueryCriteria
            criteria = CandidateQueryCriteria(
                status="new",
                order_by="score",
                order_desc=True,
                limit=1000,
            )
            all_candidates = repository.query_candidates(criteria)
            if len(all_candidates) >= 10:
                scores = [c.score for c in all_candidates]
                scores.sort(reverse=True)
                p90_idx = int(len(scores) * 0.1)
                min_score = scores[p90_idx]

    # Fetch candidates using Criteria pattern with source isolation
    candidates = repository.get_candidates_for_extraction(
        batch_size=batch_size, min_score=min_score, source_id=source_id
    )

    logger.info(
        "event_extraction_started",
        candidate_count=len(candidates),
        source=source_id.value if source_id else None,
        batch_size=batch_size,
        min_score=min_score,
        check_budget=check_budget,
    )

    if not candidates:
        result = ExtractionResult(
            events_extracted=0,
            candidates_processed=0,
            llm_calls=0,
            cache_hits=0,
            total_cost_usd=0.0,
            errors=[],
        )
        logger.info(
            "event_extraction_finished",
            events_extracted=result.events_extracted,
            candidates_processed=result.candidates_processed,
            llm_calls=result.llm_calls,
            cache_hits=result.cache_hits,
            total_cost_usd=result.total_cost_usd,
        )
        return result

    events_extracted = 0
    candidates_processed = 0
    llm_calls = 0
    cache_hits = 0
    total_cost = 0.0
    errors: list[str] = []

    for candidate in candidates:
        candidates_processed += 1

        # Get source-aware channel config using unified interface
        candidate_source = getattr(candidate, "source_id", MessageSource.SLACK)
        channel_config = settings.get_scoring_config(
            candidate_source, candidate.channel
        )
        channel_name = (
            channel_config.channel_name if channel_config else candidate.channel
        )

        logger.info(
            "processing_candidate",
            candidate_num=candidates_processed,
            total_candidates=len(candidates),
            message_id=candidate.message_id[:8],
            source=candidate_source.value,
            channel=channel_name,
        )

        try:
            logger.debug(
                "calling_llm_api",
                message_id=candidate.message_id[:8],
                text_length=len(candidate.text_norm),
                links_count=len(candidate.links_norm[:MAX_LINKS]),
            )

            llm_response = llm_client.extract_events_with_retry(
                text=candidate.text_norm,
                links=candidate.links_norm[:MAX_LINKS],
                message_ts_dt=candidate.ts_dt,
                channel_name=channel_name,
            )

            logger.info(
                "llm_response_received",
                message_id=candidate.message_id[:8],
                is_event=llm_response.is_event,
                events_count=len(llm_response.events) if llm_response.events else 0,
            )

            llm_calls += 1

            # Get call metadata
            call_metadata = llm_client.get_call_metadata()
            call_metadata.message_id = candidate.message_id
            total_cost += call_metadata.cost_usd

            # Save call metadata
            repository.save_llm_call(call_metadata)

            # Process events
            if llm_response.is_event and llm_response.events:
                events_to_save: list[Event] = []
                validation_errors: list[str] = []

                # Get reaction and mention counts from candidate features
                reaction_count = candidate.features.reaction_count
                mention_count = 1 if candidate.features.has_mention else 0

                # Get validator for event validation
                validator = _get_event_validator()

                for llm_event in llm_response.events:
                    domain_event = convert_llm_event_to_domain(
                        llm_event,
                        candidate.message_id,
                        candidate.ts_dt,
                        channel_name,
                        reaction_count=reaction_count,
                        mention_count=mention_count,
                    )

                    # Validate event before saving - check for critical errors
                    critical_errors = validator.get_critical_errors(domain_event)
                    validation_summary = validator.get_validation_summary(domain_event)

                    if critical_errors:
                        # Critical errors block saving - log for audit and skip
                        validation_errors.extend(
                            [
                                f"Event {llm_event.object_name_raw}: {error}"
                                for error in critical_errors
                            ]
                        )
                        logger.warning(
                            "event_validation_failed",
                            event_object=llm_event.object_name_raw,
                            critical_errors=critical_errors,
                            warnings_count=len(validation_summary["warnings"]),
                            info_count=len(validation_summary["info"]),
                            reason="domain_rule_violations",
                        )

                        # Skip this event - don't save events with critical errors
                        continue

                    # Event passed critical validation - save it
                    events_to_save.append(domain_event)

                    # Log warnings if present (non-blocking)
                    if validation_summary["warnings"]:
                        logger.info(
                            "event_validation_warnings",
                            event_object=llm_event.object_name_raw,
                            warnings=validation_summary["warnings"],
                        )

                # Save events (without deduplication yet)
                if events_to_save:
                    repository.save_events(events_to_save)
                    events_extracted += len(events_to_save)

                # Log validation summary with audit trail
                total_events_processed = (
                    len(llm_response.events) if llm_response.events else 0
                )
                blocked_events = total_events_processed - len(events_to_save)
                saved_events = len(events_to_save)

                logger.info(
                    "validation_audit",
                    message_id=candidate.message_id[:8],
                    saved_events=saved_events,
                    blocked_events=blocked_events,
                    total_issues=len(validation_errors),
                )

                if validation_errors:
                    critical_issues = len(
                        [
                            e
                            for e in validation_errors
                            if "critical" in e.lower()
                            or "required" in e.lower()
                            or "missing" in e.lower()
                        ]
                    )
                    logger.warning(
                        "validation_errors_detected",
                        message_id=candidate.message_id[:8],
                        blocked_events=blocked_events,
                        critical_issues=critical_issues,
                        errors=validation_errors[:5],  # Log first 5 errors
                    )

            # Update candidate status
            repository.update_candidate_status(
                candidate.message_id, CandidateStatus.LLM_OK.value
            )

        except (LLMAPIError, ValidationError) as e:
            errors.append(f"Message {candidate.message_id}: {str(e)}")
            repository.update_candidate_status(
                candidate.message_id, CandidateStatus.LLM_FAIL.value
            )
        except BudgetExceededError:
            # Stop processing on budget exceeded
            errors.append("Budget exceeded during processing")
            break
        except Exception as e:
            errors.append(f"Unexpected error for {candidate.message_id}: {str(e)}")

    result = ExtractionResult(
        events_extracted=events_extracted,
        candidates_processed=candidates_processed,
        llm_calls=llm_calls,
        cache_hits=cache_hits,
        total_cost_usd=total_cost,
        errors=errors,
    )

    logger.info(
        "event_extraction_finished",
        events_extracted=result.events_extracted,
        candidates_processed=result.candidates_processed,
        llm_calls=result.llm_calls,
        cache_hits=result.cache_hits,
        total_cost_usd=result.total_cost_usd,
        errors=len(result.errors),
    )

    return result
