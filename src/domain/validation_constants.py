"""Validation constants for event structure.

All values are based on PRD requirements.
"""

from typing import Final

# Title structure limits (PRD Section 3.2)
MAX_QUALIFIERS: Final[int] = 2
MAX_LINKS: Final[int] = 3
MAX_IMPACT_AREAS: Final[int] = 3

# Content limits (PRD Section 3.5)
MAX_SUMMARY_LENGTH: Final[int] = 320
MAX_TITLE_LENGTH: Final[int] = 140

# Importance thresholds (PRD Section 3.6)
IMPORTANCE_CRITICAL: Final[int] = 80
IMPORTANCE_HIGH: Final[int] = 60
IMPORTANCE_MEDIUM: Final[int] = 40
IMPORTANCE_LOW: Final[int] = 0

# Quality thresholds
MIN_CONFIDENCE_DEFAULT: Final[float] = 0.6
MIN_IMPORTANCE_DEFAULT: Final[int] = 60
