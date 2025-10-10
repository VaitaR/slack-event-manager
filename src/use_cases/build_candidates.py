"""Build candidates use case.

Scores messages and selects candidates for LLM extraction.
"""

from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.models import CandidateResult, CandidateStatus, EventCandidate
from src.services import scoring_engine


def build_candidates_use_case(
    repository: SQLiteRepository,
    settings: Settings,
) -> CandidateResult:
    """Build event candidates from new messages.

    1. Fetch new messages from raw (not in candidates)
    2. For each message, calculate score via scoring_engine
    3. If score >= THRESHOLD, create candidate
    4. Save features_json snapshot
    5. Insert into event_candidates

    Args:
        repository: Data repository
        settings: Application settings

    Returns:
        CandidateResult with counts

    Example:
        >>> result = build_candidates_use_case(repo, settings)
        >>> result.candidates_created
        15
    """
    # Get messages not yet scored
    new_messages = repository.get_new_messages_for_candidates()

    if not new_messages:
        return CandidateResult(
            candidates_created=0, messages_processed=0, average_score=0.0
        )

    candidates_to_save: list[EventCandidate] = []
    scores: list[float] = []
    messages_processed = 0

    for message in new_messages:
        messages_processed += 1

        # Get channel config
        channel_config = settings.get_channel_config(message.channel)

        if not channel_config:
            # Channel not in whitelist, skip
            continue

        # Score message
        score, features = scoring_engine.score_message(message, channel_config)
        scores.append(score)

        # Check if meets threshold (using Specification pattern for filtering)
        # Could also use: ScoreAboveThresholdSpec(threshold).is_satisfied_by(candidate)
        if scoring_engine.is_candidate(score, channel_config.threshold_score):
            candidate = EventCandidate(
                message_id=message.message_id,
                channel=message.channel,
                ts_dt=message.ts_dt,
                text_norm=message.text_norm,
                links_norm=message.links_norm,
                anchors=message.anchors,
                score=score,
                status=CandidateStatus.NEW,
                features=features,
            )
            candidates_to_save.append(candidate)

    # Save candidates
    candidates_created = 0
    if candidates_to_save:
        candidates_created = repository.save_candidates(candidates_to_save)

    # Calculate statistics
    average_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0

    return CandidateResult(
        candidates_created=candidates_created,
        messages_processed=messages_processed,
        average_score=average_score,
        max_score=max_score,
        min_score=min_score,
    )

