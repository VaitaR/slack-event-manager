"""Slack API client adapter."""

import time
from typing import Any, Final, cast

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.config.logging_config import get_logger
from src.domain.exceptions import RateLimitError, SlackAPIError

logger = get_logger(__name__)


DEFAULT_SLACK_PAGE_SIZE: Final[int] = 200
DEFAULT_SLACK_PAGE_DELAY_SECONDS: Final[float] = 0.5
DEFAULT_SLACK_MAX_RETRIES: Final[int] = 3


class SlackClient:
    """Slack API client with rate limiting and caching."""

    def __init__(
        self,
        bot_token: str,
        *,
        page_size: int | None = None,
        max_total_messages: int | None = None,
        page_delay_seconds: float = DEFAULT_SLACK_PAGE_DELAY_SECONDS,
        max_retries: int = DEFAULT_SLACK_MAX_RETRIES,
    ) -> None:
        """Initialize Slack client.

        Args:
            bot_token: Slack bot user OAuth token
            page_size: Optional override for per-page size (default 200)
            max_total_messages: Optional maximum messages per fetch (default unlimited)
            page_delay_seconds: Delay between paginated requests
            max_retries: Maximum retry attempts for transient errors
        """
        self.client = WebClient(token=bot_token)
        self._user_cache: dict[str, dict[str, Any]] = {}
        self._page_size = page_size or DEFAULT_SLACK_PAGE_SIZE
        if self._page_size <= 0:
            raise ValueError("Slack page_size must be positive")
        self._max_total_messages = max_total_messages
        self._page_delay_seconds = max(page_delay_seconds, 0.0)
        self._max_retries = max(max_retries, 1)

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Slack channel (root messages only).

        Implements pagination and filters for thread_ts == ts.

        Args:
            channel_id: Slack channel ID
            oldest_ts: Oldest timestamp (inclusive)
            latest_ts: Latest timestamp (inclusive)
            limit: Maximum total messages to fetch (None = unlimited)
            page_size: Optional override for messages per page

        Returns:
            List of raw Slack message dictionaries

        Raises:
            SlackAPIError: On API communication errors
            RateLimitError: On rate limit exceeded
        """
        aggregated: list[dict[str, Any]] = []
        cursor: str | None = None
        effective_page_size = page_size or self._page_size
        if effective_page_size <= 0:
            raise ValueError("Slack page_size must be positive")

        effective_limit = limit if limit is not None else self._max_total_messages

        while True:
            remaining = (
                effective_limit - len(aggregated)
                if effective_limit is not None
                else None
            )
            if remaining is not None and remaining <= 0:
                break

            page_limit = effective_page_size
            if remaining is not None:
                page_limit = min(page_limit, remaining)

            params: dict[str, Any] = {
                "channel": channel_id,
                "limit": page_limit,
            }

            if oldest_ts:
                params["oldest"] = oldest_ts
            if latest_ts:
                params["latest"] = latest_ts
            if cursor:
                params["cursor"] = cursor

            response = self._fetch_page_with_retries(channel_id, params)

            if not response.get("ok"):
                raise SlackAPIError(f"Slack API error: {response.get('error')}")

            messages: list[dict[str, Any]] = response.get("messages", [])

            root_messages = [
                msg
                for msg in messages
                if msg.get("thread_ts") is None or msg.get("thread_ts") == msg.get("ts")
            ]

            aggregated.extend(root_messages)

            cursor = response.get("response_metadata", {}).get("next_cursor")  # type: ignore[call-overload]
            if not cursor:
                break

            if self._page_delay_seconds > 0:
                time.sleep(self._page_delay_seconds)

        if effective_limit is not None and len(aggregated) > effective_limit:
            return aggregated[:effective_limit]

        return aggregated

    def _fetch_page_with_retries(
        self, channel_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute conversations.history with retry handling."""

        attempt = 0
        while True:
            try:
                return cast(dict[str, Any], self.client.conversations_history(**params))
            except SlackApiError as error:
                if error.response.get("error") == "ratelimited":
                    retry_after = int(error.response.headers.get("Retry-After", 10))
                    logger.warning(
                        "slack_rate_limited",
                        channel_id=channel_id,
                        retry_after_seconds=retry_after,
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                    )
                    time.sleep(retry_after)
                    attempt += 1
                    if attempt >= self._max_retries:
                        raise RateLimitError(retry_after=retry_after)
                    continue

                attempt += 1
                if attempt >= self._max_retries:
                    raise SlackAPIError(
                        f"Failed after {self._max_retries} retries: {error}"
                    )

                backoff_seconds = 2**attempt
                logger.warning(
                    "slack_api_retry",
                    error=str(error),
                    attempt=attempt,
                    max_retries=self._max_retries,
                    backoff_seconds=backoff_seconds,
                    channel_id=channel_id,
                )
                time.sleep(backoff_seconds)

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get user information by ID (with in-memory caching).

        Args:
            user_id: Slack user ID

        Returns:
            User info dictionary

        Raises:
            SlackAPIError: On API communication errors
        """
        if not user_id:
            return {"real_name": "Unknown", "name": "unknown"}

        # Check cache
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            response = self.client.users_info(user=user_id)

            if not response["ok"]:
                return {"real_name": "Unknown", "name": "unknown"}

            user_data = response["user"]
            self._user_cache[user_id] = user_data
            return user_data

        except SlackApiError as e:
            raise SlackAPIError(f"Failed to fetch user info: {e}")

    def post_message(
        self, channel_id: str, blocks: list[dict[str, Any]], text: str = ""
    ) -> str:
        """Post message with Block Kit to channel.

        Args:
            channel_id: Target channel ID
            blocks: Slack Block Kit blocks
            text: Fallback text for notifications

        Returns:
            Message timestamp

        Raises:
            SlackAPIError: On API communication errors
        """
        try:
            response = self.client.chat_postMessage(
                channel=channel_id, blocks=blocks, text=text or "Event Digest"
            )

            if not response["ok"]:
                raise SlackAPIError(f"Failed to post message: {response.get('error')}")

            return response["ts"]

        except SlackApiError as e:
            if e.response.get("error") == "ratelimited":
                retry_after = int(e.response.headers.get("Retry-After", 60))
                raise RateLimitError(retry_after=retry_after)

            raise SlackAPIError(f"Failed to post message: {e}")

    def get_channel_name(self, channel_id: str) -> str:
        """Get channel name by ID.

        Args:
            channel_id: Channel ID

        Returns:
            Channel name or ID if lookup fails
        """
        try:
            response = self.client.conversations_info(channel=channel_id)
            if response["ok"]:
                return response["channel"]["name"]
        except SlackApiError:
            pass

        return channel_id

    def get_permalink(self, channel_id: str, message_ts: str) -> str | None:
        """Get permalink for a message.

        Args:
            channel_id: Channel ID
            message_ts: Message timestamp

        Returns:
            Permalink URL or None if lookup fails
        """
        try:
            response = self.client.chat_getPermalink(
                channel=channel_id, message_ts=message_ts
            )
            if response["ok"]:
                return response["permalink"]
        except SlackApiError:
            pass

        return None
