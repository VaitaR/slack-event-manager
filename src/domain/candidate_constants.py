"""Domain constants for candidate processing."""

from datetime import timedelta
from typing import Final

CANDIDATE_LEASE_TIMEOUT: Final[timedelta] = timedelta(minutes=15)

__all__ = ["CANDIDATE_LEASE_TIMEOUT"]
