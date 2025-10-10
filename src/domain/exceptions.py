"""Custom exception hierarchy for the Slack Event Manager.

Following error taxonomy: retryable, non-retryable, validation, rate-limit.
"""


class SlackEventManagerError(Exception):
    """Base exception for all application errors."""

    pass


class RetryableError(SlackEventManagerError):
    """Errors that can be retried (network issues, temporary failures)."""

    pass


class NonRetryableError(SlackEventManagerError):
    """Errors that should not be retried (validation, auth, logic errors)."""

    pass


class ValidationError(NonRetryableError):
    """Data validation errors."""

    pass


class RateLimitError(RetryableError):
    """API rate limit exceeded."""

    def __init__(self, retry_after: int | None = None) -> None:
        """Initialize with optional retry_after seconds."""
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after: {retry_after}s")


class SlackAPIError(RetryableError):
    """Slack API communication errors."""

    pass


class LLMAPIError(RetryableError):
    """LLM API communication errors."""

    pass


class BudgetExceededError(NonRetryableError):
    """LLM budget has been exceeded."""

    pass


class RepositoryError(RetryableError):
    """Database/storage errors."""

    pass
