"""Extract events use case.

Uses LLM to extract structured events from candidate messages.
"""

from datetime import datetime
from typing import Any

import pytz

from src.adapters.llm_client import LLMClient
from src.adapters.query_builders import CandidateQueryCriteria
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.exceptions import BudgetExceededError, LLMAPIError, ValidationError
from src.domain.models import CandidateStatus, Event, ExtractionResult
from src.services import deduplicator


def convert_llm_event_to_domain(
    llm_event: Any,
    message_id: str,
    event_idx: int,
    message_ts_dt: datetime,
    channel_name: str,
) -> Event:
    """Convert LLM event to domain Event model.

    Args:
        llm_event: LLM event object
        message_id: Source message ID
        event_idx: Index within message (0-4)
        message_ts_dt: Message timestamp for date fallback
        channel_name: Channel name

    Returns:
        Domain Event object
    """
    # Parse dates
    try:
        event_date_str = llm_event.event_date
        event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
        if event_date.tzinfo is None:
            event_date = event_date.replace(tzinfo=pytz.UTC)
    except (ValueError, AttributeError):
        # Fall back to message timestamp
        event_date = message_ts_dt

    event_end = None
    if llm_event.event_end:
        try:
            event_end_str = llm_event.event_end
            event_end = datetime.fromisoformat(event_end_str.replace("Z", "+00:00"))
            if event_end.tzinfo is None:
                event_end = event_end.replace(tzinfo=pytz.UTC)
        except (ValueError, AttributeError):
            pass

    # Create domain event
    event = Event(
        message_id=message_id,
        source_msg_event_idx=event_idx,
        dedup_key="",  # Will be generated
        event_date=event_date,
        event_end=event_end,
        category=llm_event.category,
        title=llm_event.title,
        summary=llm_event.summary,
        impact_area=llm_event.impact_area,
        tags=llm_event.tags,
        links=llm_event.links[:3],  # Max 3
        anchors=[],  # Will be populated from links
        confidence=llm_event.confidence,
        source_channels=[channel_name] if channel_name else [],
    )

    # Generate dedup key
    event.dedup_key = deduplicator.generate_dedup_key(event)

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

                for idx, llm_event in enumerate(llm_response.events):
                    domain_event = convert_llm_event_to_domain(
                        llm_event,
                        candidate.message_id,
                        idx,
                        candidate.ts_dt,
                        channel_name,
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
