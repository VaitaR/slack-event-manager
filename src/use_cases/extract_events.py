"""Extract events use case.

Uses LLM to extract structured events from candidate messages.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from src.adapters.llm_client import LLMClient
from src.adapters.query_builders import CandidateQueryCriteria
from src.adapters.sqlite_repository import SQLiteRepository
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
    Severity,
    TimeSource,
)
from src.services import deduplicator
from src.services.importance_scorer import ImportanceScorer
from src.services.object_registry import ObjectRegistry

# Initialize services (singleton-style)
_object_registry: ObjectRegistry | None = None
_importance_scorer: ImportanceScorer | None = None


def _get_object_registry() -> ObjectRegistry:
    """Get or create ObjectRegistry instance."""
    global _object_registry
    if _object_registry is None:
        registry_path = (
            Path(__file__).parent.parent.parent / "config" / "object_registry.yaml"
        )
        _object_registry = ObjectRegistry(registry_path)
    return _object_registry


def _get_importance_scorer() -> ImportanceScorer:
    """Get or create ImportanceScorer instance."""
    global _importance_scorer
    if _importance_scorer is None:
        _importance_scorer = ImportanceScorer()
    return _importance_scorer


def _parse_datetime(dt_str: str | None, fallback: datetime) -> datetime | None:
    """Parse ISO8601 datetime string with fallback."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except (ValueError, AttributeError):
        return fallback if dt_str else None


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
        qualifiers=llm_event.qualifiers[:2],  # Max 2
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
        links=llm_event.links[:3],  # Max 3
        anchors=llm_event.anchors,
        impact_area=llm_event.impact_area[:3],  # Max 3
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
    repository: SQLiteRepository,
    settings: Settings,
    batch_size: int = 50,
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
        repository: Data repository
        settings: Application settings
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

    # Fetch candidates using Criteria pattern
    candidates = repository.get_candidates_for_extraction(
        batch_size=batch_size, min_score=min_score
    )

    if not candidates:
        return ExtractionResult(
            events_extracted=0,
            candidates_processed=0,
            llm_calls=0,
            cache_hits=0,
            total_cost_usd=0.0,
            errors=[],
        )

    events_extracted = 0
    candidates_processed = 0
    llm_calls = 0
    cache_hits = 0
    total_cost = 0.0
    errors: list[str] = []

    for candidate in candidates:
        candidates_processed += 1

        # Debug output
        print(
            f"🔄 Processing candidate {candidates_processed}/{len(candidates)}: {candidate.message_id[:8]}..."
        )
        import sys

        sys.stdout.flush()

        try:
            # Get channel config for context
            channel_config = settings.get_channel_config(candidate.channel)
            channel_name = (
                channel_config.channel_name if channel_config else candidate.channel
            )

            # Call LLM
            print("   Calling LLM API...")
            sys.stdout.flush()

            llm_response = llm_client.extract_events_with_retry(
                text=candidate.text_norm,
                links=candidate.links_norm[:3],
                message_ts_dt=candidate.ts_dt,
                channel_name=channel_name,
            )

            print("   ✅ LLM responded")
            sys.stdout.flush()

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

                # Get reaction and mention counts from candidate features
                reaction_count = candidate.features.reaction_count
                mention_count = 1 if candidate.features.has_mention else 0

                for llm_event in llm_response.events:
                    domain_event = convert_llm_event_to_domain(
                        llm_event,
                        candidate.message_id,
                        candidate.ts_dt,
                        channel_name,
                        reaction_count=reaction_count,
                        mention_count=mention_count,
                    )
                    events_to_save.append(domain_event)

                # Save events (without deduplication yet)
                if events_to_save:
                    repository.save_events(events_to_save)
                    events_extracted += len(events_to_save)

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

    return ExtractionResult(
        events_extracted=events_extracted,
        candidates_processed=candidates_processed,
        llm_calls=llm_calls,
        cache_hits=cache_hits,
        total_cost_usd=total_cost,
        errors=errors,
    )
