"""Slack API client adapter.

Implements SlackClientProtocol for Slack API interactions.
"""

import time
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.domain.exceptions import RateLimitError, SlackAPIError


class SlackClient:
    """Slack API client with rate limiting and caching."""

    def __init__(self, bot_token: str) -> None:
        """Initialize Slack client.

        Args:
            bot_token: Slack bot user OAuth token
        """
        self.client = WebClient(token=bot_token)
        self._user_cache: dict[str, dict[str, Any]] = {}

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Slack channel (root messages only).

        Implements pagination and filters for thread_ts == ts.

        Args:
            channel_id: Slack channel ID
            oldest_ts: Oldest timestamp (inclusive)
            latest_ts: Latest timestamp (inclusive)
            limit: Messages per page

        Returns:
            List of raw Slack message dictionaries

        Raises:
            SlackAPIError: On API communication errors
            RateLimitError: On rate limit exceeded
        """
        all_messages: list[dict[str, Any]] = []
        cursor: str | None = None
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                params: dict[str, Any] = {
                    "channel": channel_id,
                    "limit": limit,
                }

                if oldest_ts:
                    params["oldest"] = oldest_ts
                if latest_ts:
                    params["latest"] = latest_ts
                if cursor:
                    params["cursor"] = cursor

                response = self.client.conversations_history(**params)

                if not response["ok"]:
                    raise SlackAPIError(f"Slack API error: {response.get('error')}")

                messages = response.get("messages", [])

                # Filter for root messages only (thread_ts == ts or no thread_ts)
                root_messages = [
                    msg
                    for msg in messages
                    if msg.get("thread_ts") is None
                    or msg.get("thread_ts") == msg.get("ts")
                ]

                all_messages.extend(root_messages)

                # Stop if we've reached the requested limit
                if len(all_messages) >= limit:
                    all_messages = all_messages[:limit]
                    break

                # Check for more pages
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

                # Rate limit courtesy delay
                time.sleep(0.5)

            except SlackApiError as e:
                if e.response.get("error") == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 10))
                    import sys

                    print(
                        f"⚠️ Rate limited. Waiting {retry_after}s before retry (attempt {retry_count + 1}/{max_retries})..."
                    )
                    sys.stdout.flush()
                    time.sleep(retry_after)
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise RateLimitError(retry_after=retry_after)
                    continue

                retry_count += 1
                if retry_count >= max_retries:
                    raise SlackAPIError(f"Failed after {max_retries} retries: {e}")

                # Exponential backoff
                time.sleep(2**retry_count)

        return all_messages

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
