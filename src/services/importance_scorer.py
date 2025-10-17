"""Importance scoring service for events.

Combines heuristic scoring (H) and LLM scoring (S) into final importance.
Final = 0.6 * H + 0.4 * (S * 100), with decay and penalties.
"""

from datetime import datetime
from typing import Final

import pytz

from src.domain.models import (
    Environment,
    Event,
    EventCategory,
    EventStatus,
    ImportanceScore,
    Severity,
)

# Category base scores
CATEGORY_BASE_SCORES: Final[dict[EventCategory, int]] = {
    EventCategory.PRODUCT: 30,
    EventCategory.RISK: 35,
    EventCategory.PROCESS: 20,
    EventCategory.MARKETING: 15,
    EventCategory.ORG: 25,
    EventCategory.UNKNOWN: 10,
}

# Environment multipliers
ENV_MULTIPLIERS: Final[dict[Environment, float]] = {
    Environment.PROD: 1.2,
    Environment.MULTI: 1.15,
    Environment.STAGING: 0.8,
    Environment.DEV: 0.6,
    Environment.UNKNOWN: 1.0,
}

# Severity bonuses
SEVERITY_BONUSES: Final[dict[Severity, int]] = {
    Severity.SEV1: 20,
    Severity.SEV2: 15,
    Severity.SEV3: 10,
    Severity.INFO: 5,
    Severity.UNKNOWN: 0,
}

# Status bonuses
STATUS_BONUSES: Final[dict[EventStatus, int]] = {
    EventStatus.STARTED: 10,
    EventStatus.COMPLETED: 8,
    EventStatus.ROLLED_BACK: 15,
    EventStatus.CANCELED: 5,
    EventStatus.POSTPONED: 3,
    EventStatus.PLANNED: 0,
    EventStatus.CONFIRMED: 2,
    EventStatus.UPDATED: 5,
}

# Per-anchor bonus
ANCHOR_BONUS: Final[int] = 5

# Critical subsystems (bonus if in impact_area)
CRITICAL_SUBSYSTEMS: Final[set[str]] = {
    "authentication",
    "payment",
    "trading",
    "wallet",
    "database",
    "api-gateway",
}

CRITICAL_SUBSYSTEM_BONUS: Final[int] = 10

# Age decay (importance decreases over time)
AGE_DECAY_DAYS: Final[int] = 7
AGE_DECAY_FACTOR: Final[float] = 0.9  # 10% reduction per week

# Duplicate penalty
DUPLICATE_PENALTY: Final[int] = 20


class ImportanceScorer:
    """Calculate importance scores for events."""

    def __init__(
        self,
        category_scores: dict[str, int] | None = None,
        critical_subsystems: set[str] | None = None,
    ) -> None:
        """Initialize importance scorer.

        Args:
            category_scores: Override default category base scores
            critical_subsystems: Override default critical subsystems
        """
        self.category_scores = category_scores or {
            k.value: v for k, v in CATEGORY_BASE_SCORES.items()
        }
        self.critical_subsystems = critical_subsystems or CRITICAL_SUBSYSTEMS

    def calculate_heuristic_score(
        self,
        event: Event,
        reaction_count: int = 0,
        mention_count: int = 0,
    ) -> int:
        """Calculate heuristic score (H: 0-100).

        Components:
        - Category base score
        - Scope multiplier (impact_area count, environment)
        - Urgency (severity, status)
        - Signals (reactions, mentions)
        - Anchors
        - Critical subsystems

        Args:
            event: Event to score
            reaction_count: Number of reactions on source message
            mention_count: Number of mentions (@channel, @here)

        Returns:
            Heuristic score (0-100)

        Example:
            >>> scorer = ImportanceScorer()
            >>> score = scorer.calculate_heuristic_score(event, reaction_count=5)
            >>> score
            75
        """
        # Base score from category
        base = self.category_scores.get(event.category.value, 10)

        # Scope multiplier
        scope_mult = 1.0

        # Impact area count
        if len(event.impact_area) >= 3:
            scope_mult *= 1.3
        elif len(event.impact_area) >= 2:
            scope_mult *= 1.15
        elif len(event.impact_area) >= 1:
            scope_mult *= 1.0
        else:
            scope_mult *= 0.9

        # Environment multiplier
        env_mult = ENV_MULTIPLIERS.get(event.environment, 1.0)
        scope_mult *= env_mult

        score = int(base * scope_mult)

        # Urgency bonuses
        if event.severity:
            score += SEVERITY_BONUSES.get(event.severity, 0)

        score += STATUS_BONUSES.get(event.status, 0)

        # Signals
        if reaction_count > 0:
            score += min(reaction_count, 10)  # Max +10 from reactions

        if mention_count > 0:
            score += min(mention_count * 3, 9)  # Max +9 from mentions

        # Anchors
        score += min(len(event.anchors) * ANCHOR_BONUS, 15)  # Max +15 from anchors

        # Critical subsystems
        if any(
            subsystem.lower() in self.critical_subsystems
            for subsystem in event.impact_area
        ):
            score += CRITICAL_SUBSYSTEM_BONUS

        # Cap at 100
        return min(score, 100)

    def calculate_importance(
        self,
        event: Event,
        llm_score: float | None = None,
        reaction_count: int = 0,
        mention_count: int = 0,
        is_duplicate: bool = False,
    ) -> ImportanceScore:
        """Calculate final importance score.

        Formula: importance = round(0.6 * H + 0.4 * (S * 100))
        With age decay and duplicate penalty.

        Args:
            event: Event to score
            llm_score: LLM importance score (0-1), None to skip
            reaction_count: Reactions on source message
            mention_count: Mentions in source message
            is_duplicate: If event is duplicate/merged

        Returns:
            ImportanceScore with breakdown

        Example:
            >>> scorer = ImportanceScorer()
            >>> result = scorer.calculate_importance(event, llm_score=0.8)
            >>> result.final_score
            72
        """
        # Heuristic score (H: 0-100)
        heuristic = self.calculate_heuristic_score(event, reaction_count, mention_count)

        # LLM score (S: 0-1) - use default if not provided
        llm_score_value = llm_score if llm_score is not None else 0.5

        # Weighted combination
        importance = round(0.6 * heuristic + 0.4 * (llm_score_value * 100))

        # Age decay
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)

        # Find most recent time (actual > planned)
        event_time = (
            event.actual_start
            or event.actual_end
            or event.planned_start
            or event.planned_end
        )

        if event_time:
            age_days = (now - event_time).total_seconds() / 86400
            if age_days > AGE_DECAY_DAYS:
                weeks_old = age_days / 7
                decay = AGE_DECAY_FACTOR**weeks_old
                importance = int(importance * decay)

        # Duplicate penalty
        if is_duplicate:
            importance = max(importance - DUPLICATE_PENALTY, 0)

        # Cap at 0-100
        importance = max(0, min(importance, 100))

        return ImportanceScore(
            heuristic_score=heuristic,
            llm_score=llm_score_value,
            final_score=importance,
        )

    def get_importance_label(self, importance: int) -> str:
        """Get human-readable label for importance score.

        Args:
            importance: Importance score (0-100)

        Returns:
            Label: "critical", "high", "medium", "low"

        Example:
            >>> scorer = ImportanceScorer()
            >>> scorer.get_importance_label(85)
            'high'
        """
        if importance >= 80:
            return "critical"
        elif importance >= 60:
            return "high"
        elif importance >= 40:
            return "medium"
        else:
            return "low"
