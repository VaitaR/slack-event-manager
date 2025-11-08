"""Extract events use case.

Uses LLM to extract structured events from candidate messages.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import pytz

from src.adapters.llm_client import LLMClient
from src.adapters.query_builders import CandidateQueryCriteria
from src.config.logging_config import get_logger
from src.config.settings import LLM_CACHE_TTL_DAYS_DEFAULT, Settings
from src.domain.exceptions import BudgetExceededError, LLMAPIError, ValidationError
from src.domain.models import (
    ActionType,
    CandidateStatus,
    ChangeType,
    Environment,
    Event,
    EventCandidate,
    EventStatus,
    ExtractionResult,
    LLMCallMetadata,
    LLMEvent,
    LLMResponse,
    MessageSource,
    Severity,
    TimeSource,
)
from src.domain.protocols import RepositoryProtocol
from src.domain.task_queue import TaskCreate, TaskType
from src.domain.validation_constants import (
    MAX_IMPACT_AREAS,
    MAX_LINKS,
    MAX_QUALIFIERS,
)
from src.observability.metrics import PIPELINE_STAGE_DURATION_SECONDS
from src.observability.tracing import correlation_scope
from src.ports.task_queue import TaskQueuePort
from src.services import deduplicator, token_budget
from src.services.importance_scorer import ImportanceScorer
from src.services.object_registry import ObjectRegistry
from src.services.validators import EventValidator

logger = get_logger(__name__)

_event_validator: EventValidator | None = None


def build_object_registry(settings: Settings) -> ObjectRegistry:
    """Build an object registry using application settings."""

    registry_path = Path(settings.object_registry_path)

    if not registry_path.exists():
        logger.warning("object_registry_missing", path=str(registry_path))
        fallback_path = Path("config/defaults/object_registry.example.yaml")
        if fallback_path.exists():
            registry_path = fallback_path
            logger.info("object_registry_fallback", path=str(fallback_path))
        else:
            logger.warning("object_registry_unavailable")

    return ObjectRegistry(registry_path)


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


def _normalize_to_utc(dt: datetime) -> datetime:
    """Ensure datetime has UTC timezone information."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=pytz.UTC)
    return dt.astimezone(pytz.UTC)


def _compute_prompt_hash(
    llm_client: LLMClient,
    chunk_text: str,
    links: list[str],
    message_ts_dt: datetime,
    channel_name: str,
    chunk_index: int,
) -> str:
    """Compute deterministic prompt hash for caching."""

    payload = {
        "text": chunk_text,
        "links": links,
        "ts": _normalize_to_utc(message_ts_dt).isoformat(),
        "channel": channel_name,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    normalized_content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    template_version = llm_client.prompt_version or "inline"
    prompt_fingerprint = llm_client.system_prompt_hash
    hash_input = "|".join(
        [
            llm_client.model,
            template_version,
            prompt_fingerprint,
            str(chunk_index),
            normalized_content_hash,
        ]
    )
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CandidateExtractionMetrics:
    """Metrics produced while processing a single candidate."""

    events_extracted: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    total_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    dedup_required: bool = False


@dataclass(slots=True)
class LLMTaskScheduleResult:
    """Summary of scheduled LLM extraction jobs."""

    total_candidates: int = 0
    candidates_enqueued: int = 0


def _resolve_cache_ttl(settings: Settings) -> timedelta | None:
    """Resolve cache TTL configuration to a ``timedelta``."""

    cache_ttl_raw = getattr(settings, "llm_cache_ttl_days", None)
    if cache_ttl_raw is None:
        return timedelta(days=LLM_CACHE_TTL_DAYS_DEFAULT)

    try:
        cache_ttl_days = int(cache_ttl_raw)
    except (TypeError, ValueError):
        logger.warning("invalid_cache_ttl_config", value=cache_ttl_raw)
        return timedelta(days=LLM_CACHE_TTL_DAYS_DEFAULT)

    if cache_ttl_days <= 0:
        return None

    return timedelta(days=cache_ttl_days)


def _ensure_budget_allows_call(
    *, repository: RepositoryProtocol, settings: Settings
) -> None:
    """Raise :class:`BudgetExceededError` if the daily budget is exhausted."""

    today = datetime.now(tz=pytz.UTC)
    daily_cost = repository.get_daily_llm_cost(today)
    if daily_cost >= settings.llm_daily_budget_usd:
        raise BudgetExceededError(
            f"Daily budget ${settings.llm_daily_budget_usd} exceeded: ${daily_cost:.2f}"
        )


def convert_llm_event_to_domain(
    llm_event: Any,
    *,
    message_id: str,
    message_ts_dt: datetime,
    channel_name: str,
    source_id: MessageSource,
    object_registry: ObjectRegistry,
) -> Event:
    """Convert LLM event to domain Event model with new comprehensive structure.

    Uses ObjectRegistry, ImportanceScorer, and deduplicator services.

    Args:
        llm_event: LLM event object
        message_id: Source message ID
        message_ts_dt: Message timestamp for date fallback
        channel_name: Channel name
        source_id: Source identifier for provenance
        object_registry: Registry for canonicalizing object names

    Returns:
        Domain Event object with all new fields
    """
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
        source_id=source_id,
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

    return event


def _process_candidate_with_llm(
    *,
    candidate: EventCandidate,
    llm_client: LLMClient,
    repository: RepositoryProtocol,
    settings: Settings,
    cache_ttl: timedelta | None,
    object_registry: ObjectRegistry,
    importance_scorer: ImportanceScorer,
    validator: EventValidator,
    correlation_id: str | None,
    position: int | None,
    total_candidates: int | None,
    enforce_budget: bool,
) -> CandidateExtractionMetrics:
    metrics = CandidateExtractionMetrics()
    candidate_source = getattr(candidate, "source_id", MessageSource.SLACK)
    if not isinstance(candidate_source, MessageSource):
        candidate_source = MessageSource.SLACK
    channel_config = settings.get_scoring_config(candidate_source, candidate.channel)
    feature_channel_name = getattr(candidate.features, "channel_name", None)
    if not isinstance(feature_channel_name, str):
        feature_channel_name = None
    channel_name = feature_channel_name or (
        channel_config.channel_name if channel_config else candidate.channel
    )

    logger.info(
        "processing_candidate",
        correlation_id=correlation_id,
        candidate_num=position,
        total_candidates=total_candidates,
        message_id=candidate.message_id[:8],
        source=candidate_source.value,
        channel=channel_name,
    )

    limited_links = candidate.links_norm[:MAX_LINKS]
    char_budget = token_budget.characters_for_tokens(
        llm_client.prompt_token_budget, llm_client.model
    )
    text_chunks = token_budget.truncate_or_chunk(candidate.text_norm, char_budget)

    if len(text_chunks) > 1:
        logger.info(
            "llm_candidate_chunked",
            correlation_id=correlation_id,
            message_id=candidate.message_id[:8],
            chunk_count=len(text_chunks),
            channel=channel_name,
        )

    chunk_events: list[LLMEvent] = []
    chunk_is_event = False
    budget_checked = False

    try:
        for chunk_index, chunk_text in enumerate(text_chunks):
            prompt_hash = _compute_prompt_hash(
                llm_client=llm_client,
                chunk_text=chunk_text,
                links=limited_links,
                message_ts_dt=candidate.ts_dt,
                channel_name=channel_name,
                chunk_index=chunk_index,
            )

            llm_response: LLMResponse | None = None
            if cache_ttl is not None:
                cached_payload = repository.get_cached_llm_response(
                    prompt_hash, max_age=cache_ttl
                )
                if isinstance(cached_payload, str) and cached_payload:
                    try:
                        llm_response = LLMResponse.model_validate_json(cached_payload)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "llm_cache_deserialization_failed",
                            prompt_hash=prompt_hash,
                            error=str(exc),
                        )
                        repository.invalidate_llm_cache_entry(prompt_hash)
                    else:
                        metrics.cache_hits += 1
                        logger.info(
                            "llm_cache_hit",
                            correlation_id=correlation_id,
                            message_id=candidate.message_id[:8],
                            prompt_hash=prompt_hash[:12],
                            chunk_index=chunk_index,
                            events_count=len(llm_response.events)
                            if llm_response.events
                            else 0,
                        )
                        repository.save_llm_call(
                            LLMCallMetadata(
                                message_id=candidate.message_id,
                                prompt_hash=prompt_hash,
                                model=llm_client.model,
                                tokens_in=0,
                                tokens_out=0,
                                cost_usd=0.0,
                                latency_ms=0,
                                cached=True,
                            )
                        )

            if llm_response is None:
                if enforce_budget and not budget_checked:
                    _ensure_budget_allows_call(repository=repository, settings=settings)
                    budget_checked = True

                logger.debug(
                    "calling_llm_api",
                    correlation_id=correlation_id,
                    message_id=candidate.message_id[:8],
                    text_length=len(chunk_text),
                    links_count=len(limited_links),
                    chunk_index=chunk_index,
                )

                llm_response = llm_client.extract_events_with_retry(
                    text=chunk_text,
                    links=limited_links,
                    message_ts_dt=candidate.ts_dt,
                    channel_name=channel_name,
                    chunk_index=chunk_index,
                )

                logger.info(
                    "llm_response_received",
                    correlation_id=correlation_id,
                    message_id=candidate.message_id[:8],
                    is_event=llm_response.is_event,
                    events_count=len(llm_response.events) if llm_response.events else 0,
                    chunk_index=chunk_index,
                )

                metrics.llm_calls += 1

                call_metadata = llm_client.get_call_metadata()
                call_metadata.message_id = candidate.message_id
                call_metadata.prompt_hash = prompt_hash
                call_metadata.cached = False
                metrics.total_cost_usd += call_metadata.cost_usd

                repository.save_llm_call(call_metadata)
                repository.save_llm_response(
                    prompt_hash, llm_response.model_dump_json()
                )

            if llm_response.events:
                chunk_events.extend(llm_response.events)
            chunk_is_event = chunk_is_event or llm_response.is_event

        events_source = chunk_events
        max_events_raw = getattr(settings, "llm_max_events_per_msg", 5)
        try:
            max_events = int(max_events_raw) if max_events_raw is not None else None
        except (TypeError, ValueError):
            max_events = 5

        llm_events = (
            events_source[:max_events] if max_events is not None else events_source
        )

        if len(events_source) > len(llm_events):
            logger.info(
                "llm_response_truncated",
                correlation_id=correlation_id,
                message_id=candidate.message_id[:8],
                original_count=len(events_source),
                max_events=max_events,
            )

        if chunk_is_event and llm_events:
            events_to_save: list[Event] = []
            validation_errors: list[str] = []

            reaction_count = candidate.features.reaction_count
            mention_count = 1 if candidate.features.has_mention else 0

            for llm_event in llm_events:
                domain_event = convert_llm_event_to_domain(
                    llm_event,
                    message_id=candidate.message_id,
                    message_ts_dt=candidate.ts_dt,
                    channel_name=channel_name,
                    source_id=candidate_source,
                    object_registry=object_registry,
                )

                importance_result = importance_scorer.calculate_importance(
                    domain_event,
                    llm_score=None,
                    reaction_count=reaction_count,
                    mention_count=mention_count,
                    is_duplicate=False,
                )
                domain_event.importance = importance_result.final_score

                critical_errors = validator.get_critical_errors(domain_event)
                validation_summary = validator.get_validation_summary(domain_event)

                if critical_errors:
                    validation_errors.extend(
                        [
                            f"Event {llm_event.object_name_raw}: {error}"
                            for error in critical_errors
                        ]
                    )
                    logger.warning(
                        "event_validation_failed",
                        correlation_id=correlation_id,
                        event_object=llm_event.object_name_raw,
                        critical_errors=critical_errors,
                        warnings_count=len(validation_summary["warnings"]),
                        info_count=len(validation_summary["info"]),
                        reason="domain_rule_violations",
                    )
                    continue

                events_to_save.append(domain_event)

                if validation_summary["warnings"]:
                    logger.info(
                        "event_validation_warnings",
                        correlation_id=correlation_id,
                        event_object=llm_event.object_name_raw,
                        warnings=validation_summary["warnings"],
                    )

            if events_to_save:
                repository.save_events(events_to_save)
                metrics.events_extracted += len(events_to_save)
                metrics.dedup_required = True

            total_events_processed = len(events_source)
            blocked_events = total_events_processed - len(events_to_save)
            saved_events = len(events_to_save)

            logger.info(
                "validation_audit",
                correlation_id=correlation_id,
                message_id=candidate.message_id[:8],
                saved_events=saved_events,
                blocked_events=blocked_events,
                total_issues=len(validation_errors),
            )

            if validation_errors:
                critical_issues = len(
                    [
                        error
                        for error in validation_errors
                        if "critical" in error.lower()
                        or "required" in error.lower()
                        or "missing" in error.lower()
                    ]
                )
                logger.warning(
                    "validation_errors_detected",
                    correlation_id=correlation_id,
                    message_id=candidate.message_id[:8],
                    blocked_events=blocked_events,
                    critical_issues=critical_issues,
                    errors=validation_errors[:5],
                )

        repository.update_candidate_status(
            candidate.message_id, CandidateStatus.LLM_OK.value
        )
        return metrics

    except BudgetExceededError:
        metrics.errors.append("Budget exceeded during processing")
        metrics.budget_exhausted = True
        repository.update_candidate_status(
            candidate.message_id, CandidateStatus.NEW.value
        )
        return metrics
    except (LLMAPIError, ValidationError) as exc:
        metrics.errors.append(f"Message {candidate.message_id}: {exc}")
        repository.update_candidate_status(
            candidate.message_id, CandidateStatus.LLM_FAIL.value
        )
        return metrics
    except Exception as exc:  # noqa: BLE001
        metrics.errors.append(f"Unexpected error for {candidate.message_id}: {exc}")
        repository.update_candidate_status(
            candidate.message_id, CandidateStatus.LLM_FAIL.value
        )
        return metrics


def extract_events_use_case(
    llm_client: LLMClient,
    repository: RepositoryProtocol,
    settings: Settings,
    source_id: MessageSource | None = None,
    batch_size: int | None = 50,
    check_budget: bool = True,
    *,
    object_registry: ObjectRegistry,
    importance_scorer: ImportanceScorer,
    event_validator: EventValidator | None = None,
    correlation_id: str | None = None,
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
        >>> registry = build_object_registry(settings)
        >>> scorer = ImportanceScorer()
        >>> result = extract_events_use_case(
        ...     llm,
        ...     repo,
        ...     settings,
        ...     object_registry=registry,
        ...     importance_scorer=scorer,
        ... )
        >>> result.events_extracted
        45
    """
    with correlation_scope(correlation_id) as bound_correlation_id:
        stage_start = perf_counter()
        final_result: ExtractionResult | None = None
        final_log_payload: dict[str, object] = {}
        logger.info(
            "event_extraction_started",
            correlation_id=bound_correlation_id,
            source=source_id.value if source_id else None,
            batch_size=batch_size,
            check_budget=check_budget,
        )

        try:
            min_score = None
            if check_budget:
                today = datetime.now(tz=pytz.UTC)
                daily_cost = repository.get_daily_llm_cost(today)

                if daily_cost >= settings.llm_daily_budget_usd:
                    raise BudgetExceededError(
                        f"Daily budget ${settings.llm_daily_budget_usd} exceeded: ${daily_cost:.2f}"
                    )

                remaining_budget = settings.llm_daily_budget_usd - daily_cost

                if remaining_budget < settings.llm_daily_budget_usd * 0.2:
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

            candidates = repository.get_candidates_for_extraction(
                batch_size=batch_size, min_score=min_score, source_id=source_id
            )

            logger.info(
                "candidate_batch_ready",
                correlation_id=bound_correlation_id,
                candidate_count=len(candidates),
                min_score=min_score,
            )

            if not candidates:
                final_result = ExtractionResult(
                    events_extracted=0,
                    candidates_processed=0,
                    llm_calls=0,
                    cache_hits=0,
                    total_cost_usd=0.0,
                    errors=[],
                )
                final_log_payload = {
                    "events_extracted": final_result.events_extracted,
                    "candidates_processed": final_result.candidates_processed,
                    "llm_calls": final_result.llm_calls,
                    "cache_hits": final_result.cache_hits,
                    "total_cost_usd": final_result.total_cost_usd,
                    "errors": len(final_result.errors),
                }
                return final_result

            events_extracted = 0
            candidates_processed = 0
            llm_calls = 0
            cache_hits = 0
            total_cost = 0.0
            errors: list[str] = []
            cache_ttl = _resolve_cache_ttl(settings)
            validator = event_validator or _get_event_validator()

            for index, candidate in enumerate(candidates, start=1):
                candidates_processed += 1

                metrics = _process_candidate_with_llm(
                    candidate=candidate,
                    llm_client=llm_client,
                    repository=repository,
                    settings=settings,
                    cache_ttl=cache_ttl,
                    object_registry=object_registry,
                    importance_scorer=importance_scorer,
                    validator=validator,
                    correlation_id=bound_correlation_id,
                    position=index,
                    total_candidates=len(candidates),
                    enforce_budget=check_budget,
                )

                events_extracted += metrics.events_extracted
                llm_calls += metrics.llm_calls
                cache_hits += metrics.cache_hits
                total_cost += metrics.total_cost_usd
                errors.extend(metrics.errors)

                if metrics.budget_exhausted:
                    logger.warning(
                        "llm_budget_exhausted_mid_batch",
                        correlation_id=bound_correlation_id,
                        message_id=candidate.message_id[:8],
                    )
                    break

            final_result = ExtractionResult(
                events_extracted=events_extracted,
                candidates_processed=candidates_processed,
                llm_calls=llm_calls,
                cache_hits=cache_hits,
                total_cost_usd=total_cost,
                errors=errors,
            )
            final_log_payload = {
                "events_extracted": final_result.events_extracted,
                "candidates_processed": final_result.candidates_processed,
                "llm_calls": final_result.llm_calls,
                "cache_hits": final_result.cache_hits,
                "total_cost_usd": final_result.total_cost_usd,
                "errors": len(final_result.errors),
            }
            return final_result
        finally:
            duration = perf_counter() - stage_start
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="extract").observe(duration)
            if final_result is not None:
                logger.info(
                    "event_extraction_finished",
                    correlation_id=bound_correlation_id,
                    duration_seconds=duration,
                    **final_log_payload,
                )


def schedule_llm_extraction_tasks_use_case(
    *,
    repository: RepositoryProtocol,
    task_queue: TaskQueuePort,
    settings: Settings,
    priority: int,
    batch_size: int | None = 50,
    batch_hint: int | None = None,
    source_id: MessageSource | None = None,
    correlation_id: str | None = None,
) -> LLMTaskScheduleResult:
    """Lease candidates and enqueue LLM extraction tasks."""

    with correlation_scope(correlation_id) as bound_correlation_id:
        effective_batch_size = (
            batch_hint if isinstance(batch_hint, int) and batch_hint > 0 else batch_size
        )

        candidates = repository.get_candidates_for_extraction(
            batch_size=effective_batch_size,
            source_id=source_id,
        )

        logger.info(
            "llm_schedule_candidate_batch",
            correlation_id=bound_correlation_id,
            total_available=len(candidates),
            batch_size=effective_batch_size,
            batch_hint=batch_hint,
            source=source_id.value if source_id else None,
        )

        if not candidates:
            return LLMTaskScheduleResult()

        tasks = [
            TaskCreate(
                task_type=TaskType.LLM_EXTRACTION,
                payload={
                    "message_id": candidate.message_id,
                    "source_id": candidate.source_id.value,
                    "correlation_id": bound_correlation_id,
                },
                priority=priority,
                idempotency_key=f"llm:{candidate.message_id}",
            )
            for candidate in candidates
        ]

        enqueued = task_queue.enqueue_many(tasks)

        logger.info(
            "llm_tasks_enqueued",
            correlation_id=bound_correlation_id,
            total_candidates=len(candidates),
            enqueued=len(enqueued),
        )

        return LLMTaskScheduleResult(
            total_candidates=len(candidates),
            candidates_enqueued=len(enqueued),
        )


def process_llm_candidate_task_use_case(
    *,
    message_id: str,
    llm_client: LLMClient,
    repository: RepositoryProtocol,
    settings: Settings,
    object_registry: ObjectRegistry,
    importance_scorer: ImportanceScorer,
    event_validator: EventValidator | None = None,
    correlation_id: str | None = None,
) -> CandidateExtractionMetrics:
    """Process a single candidate through the LLM pipeline."""

    with correlation_scope(correlation_id) as bound_correlation_id:
        stage_start = perf_counter()
        validator = event_validator or _get_event_validator()
        cache_ttl = _resolve_cache_ttl(settings)

        candidate = repository.get_candidate_by_message_id(message_id)
        if candidate is None:
            duration = perf_counter() - stage_start
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="extract").observe(duration)
            logger.info(
                "llm_candidate_missing",
                correlation_id=bound_correlation_id,
                message_id=message_id[:8],
            )
            return CandidateExtractionMetrics()

        metrics = _process_candidate_with_llm(
            candidate=candidate,
            llm_client=llm_client,
            repository=repository,
            settings=settings,
            cache_ttl=cache_ttl,
            object_registry=object_registry,
            importance_scorer=importance_scorer,
            validator=validator,
            correlation_id=bound_correlation_id,
            position=None,
            total_candidates=None,
            enforce_budget=True,
        )

        duration = perf_counter() - stage_start
        PIPELINE_STAGE_DURATION_SECONDS.labels(stage="extract").observe(duration)
        logger.info(
            "llm_candidate_processed",
            correlation_id=bound_correlation_id,
            message_id=message_id[:8],
            events_extracted=metrics.events_extracted,
            llm_calls=metrics.llm_calls,
            cache_hits=metrics.cache_hits,
            duration_seconds=duration,
            dedup_required=metrics.dedup_required,
            budget_exhausted=metrics.budget_exhausted,
            errors=len(metrics.errors),
        )

        if metrics.errors:
            logger.warning(
                "llm_candidate_errors",
                correlation_id=bound_correlation_id,
                message_id=message_id[:8],
                error_count=len(metrics.errors),
            )

        return metrics
