"""Ingest messages use case.

Fetches messages from Slack channels and stores them with normalization.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import pytz

from src.adapters.slack_client import SlackClient
from src.adapters.slack_state_store import GetConnectionCallable, SlackStateStore
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.domain.exceptions import SlackAPIError
from src.domain.models import IngestResult, SlackMessage
from src.domain.protocols import RepositoryProtocol
from src.services import link_extractor, text_normalizer

if TYPE_CHECKING:
    from src.adapters.sqlite_repository import SQLiteRepository

try:
    from src.adapters.sqlite_repository import (
        SQLiteRepository as SQLiteRepositoryRuntime,
    )
except ImportError:  # pragma: no cover
    SQLiteRepositoryRuntime = None  # type: ignore[misc,assignment]

try:
    from src.adapters.postgres_repository import (
        PostgresRepository as PostgresRepositoryRuntime,
    )
except ImportError:  # pragma: no cover
    PostgresRepositoryRuntime = None  # type: ignore[misc,assignment]

logger = get_logger(__name__)

DEFAULT_STATE_SQLITE_DSN: Final[str] = "data/app.db"


@contextmanager
def _sqlite_repository_connection(
    repository: SQLiteRepository,
) -> Iterator[sqlite3.Connection]:
    conn = repository._get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _fallback_sqlite_connection(dsn: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(dsn)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _resolve_state_connection_factory(
    repository: RepositoryProtocol,
) -> GetConnectionCallable:
    if SQLiteRepositoryRuntime is not None and isinstance(
        repository, SQLiteRepositoryRuntime
    ):

        def _get_conn() -> AbstractContextManager[sqlite3.Connection]:
            return _sqlite_repository_connection(repository)

        return _get_conn

    if PostgresRepositoryRuntime is not None and isinstance(
        repository, PostgresRepositoryRuntime
    ):
        return repository._get_connection

    raw_get_conn = getattr(repository, "_get_connection", None)
    if callable(raw_get_conn):

        def _generic_conn() -> AbstractContextManager[Any]:
            @contextmanager
            def _ctx() -> Iterator[Any]:
                candidate: Any = raw_get_conn()
                enter = getattr(candidate, "__enter__", None)
                exit_ = getattr(candidate, "__exit__", None)
                if callable(enter) and callable(exit_):
                    with candidate as conn:
                        yield conn
                else:
                    try:
                        yield candidate
                    finally:
                        close = getattr(candidate, "close", None)
                        if callable(close):
                            close()

            return _ctx()

        return _generic_conn

    dsn = os.getenv("SQLITE_DSN", DEFAULT_STATE_SQLITE_DSN)
    logger.warning("slack_state_store_fallback_sqlite", dsn=dsn)

    def _fallback_get_conn() -> AbstractContextManager[sqlite3.Connection]:
        return _fallback_sqlite_connection(dsn)

    return _fallback_get_conn


def _fetch_slack_messages(
    slack_client: SlackClient,
    channel_id: str,
    *,
    oldest_ts: str | None,
    cursor: str | None,
    limit: int | None,
    page_size: int,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    aggregated: list[dict[str, Any]] = []
    current_cursor = cursor
    next_cursor: str | None = None

    while True:
        remaining = None if limit is None else limit - len(aggregated)
        if remaining is not None and remaining <= 0:
            next_cursor = current_cursor
            break

        page_limit = (
            page_size if remaining is None else max(1, min(page_size, remaining))
        )

        params: dict[str, Any] = {"channel": channel_id, "limit": page_limit}
        if oldest_ts:
            params["oldest"] = oldest_ts
        if current_cursor:
            params["cursor"] = current_cursor

        response = slack_client._fetch_page_with_retries(channel_id, params)

        if not response.get("ok"):
            raise SlackAPIError(f"Slack API error: {response.get('error')}")

        messages: list[dict[str, Any]] = response.get("messages", [])
        root_messages = [
            msg
            for msg in messages
            if msg.get("thread_ts") is None or msg.get("thread_ts") == msg.get("ts")
        ]
        aggregated.extend(root_messages)

        metadata = response.get("response_metadata") or {}
        next_cursor_value = metadata.get("next_cursor")
        if not next_cursor_value:
            next_cursor = None
            current_cursor = None
        else:
            current_cursor = str(next_cursor_value)
            next_cursor = current_cursor

        if limit is not None and len(aggregated) >= limit:
            break
        if not next_cursor_value:
            break

    has_more = next_cursor is not None
    return aggregated, next_cursor, has_more


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=pytz.UTC)
    return dt.astimezone(pytz.UTC)


def generate_message_id(channel: str, ts: str) -> str:
    """Generate deterministic message ID.

    Args:
        channel: Channel ID
        ts: Message timestamp

    Returns:
        SHA1 hash of channel|ts

    Example:
        >>> generate_message_id("C123", "1234567890.123456")
        'a3f2b1c0...'
    """
    key_material = f"{channel}|{ts}"
    return hashlib.sha1(key_material.encode("utf-8")).hexdigest()


def _parse_ts(raw: Any) -> float:
    """Convert Slack timestamp representations to float."""

    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str) and raw:
        try:
            return float(raw)
        except ValueError:
            return 0.0
    return 0.0


def process_slack_message(
    raw_msg: dict[str, Any],
    channel_id: str,
    user_info: dict[str, Any] | None = None,
    permalink: str | None = None,
) -> SlackMessage:
    """Process raw Slack message into domain model.

    Args:
        raw_msg: Raw Slack message dictionary
        channel_id: Channel ID
        user_info: Optional user info from Slack API
        permalink: Optional permalink URL

    Returns:
        Processed SlackMessage
    """
    # Extract basic fields
    ts = raw_msg.get("ts", "")
    ts_dt = (
        datetime.fromtimestamp(float(ts), tz=pytz.UTC)
        if ts
        else datetime.utcnow().replace(tzinfo=pytz.UTC)
    )
    user = raw_msg.get("user")
    bot_id = raw_msg.get("bot_id")
    is_bot = bool(bot_id or raw_msg.get("subtype") == "bot_message")
    subtype = raw_msg.get("subtype")
    text = raw_msg.get("text", "")

    # Extract user information
    user_real_name: str | None = None
    user_display_name: str | None = None
    user_email: str | None = None
    user_profile_image: str | None = None

    if user_info:
        profile = user_info.get("profile", {})
        user_real_name = user_info.get("real_name") or profile.get("real_name")
        user_display_name = profile.get("display_name") or user_info.get("name")
        user_email = profile.get("email")
        # Get largest available profile image
        user_profile_image = (
            profile.get("image_512")
            or profile.get("image_192")
            or profile.get("image_72")
            or profile.get("image_48")
        )

    # Extract blocks text
    blocks = raw_msg.get("blocks", [])
    blocks_text = text_normalizer.extract_blocks_text(blocks)

    # Combine text sources
    combined_text = text_normalizer.combine_text_sources(text, blocks_text)

    # Extract links and anchors
    links_raw = link_extractor.extract_urls(combined_text)
    links_norm = link_extractor.normalize_links(links_raw)
    anchors = link_extractor.extract_all_anchors(combined_text)

    # Normalize text
    text_norm = text_normalizer.normalize_text(combined_text)

    # Extract attachments and files count
    attachments_count = len(raw_msg.get("attachments", []))
    files_count = len(raw_msg.get("files", []))

    # Extract reactions
    reactions: dict[str, int] = {}
    total_reactions = 0
    for reaction in raw_msg.get("reactions", []):
        emoji = reaction.get("name", "")
        count = reaction.get("count", 0)
        if emoji:
            reactions[emoji] = count
            total_reactions += count

    # Reply count
    reply_count = raw_msg.get("reply_count", 0)

    # Extract edit information
    edited_info = raw_msg.get("edited")
    edited_ts: str | None = None
    edited_user: str | None = None
    if edited_info:
        edited_ts = edited_info.get("ts")
        edited_user = edited_info.get("user")

    # Generate message ID
    message_id = generate_message_id(channel_id, ts)

    return SlackMessage(
        message_id=message_id,
        channel=channel_id,
        ts=ts,
        ts_dt=ts_dt,
        user=user,
        user_real_name=user_real_name,
        user_display_name=user_display_name,
        user_email=user_email,
        user_profile_image=user_profile_image,
        is_bot=is_bot,
        bot_id=bot_id,
        subtype=subtype,
        text=text,
        blocks_text=blocks_text,
        text_norm=text_norm,
        links_raw=links_raw,
        links_norm=links_norm,
        anchors=anchors,
        attachments_count=attachments_count,
        files_count=files_count,
        reactions=reactions,
        total_reactions=total_reactions,
        reply_count=reply_count,
        permalink=permalink,
        edited_ts=edited_ts,
        edited_user=edited_user,
        ingested_at=datetime.utcnow().replace(tzinfo=pytz.UTC),
    )


def ingest_messages_use_case(
    slack_client: SlackClient,
    repository: RepositoryProtocol,
    settings: Settings,
    lookback_hours: int | None = None,
    backfill_from_date: datetime | None = None,
) -> IngestResult:
    """Ingest messages from configured Slack channels.

    For each channel:
    1. Get ingestion state (last_ts)
    2. Determine oldest timestamp:
       - If state exists: use last_ts (incremental)
       - If no state and backfill_from_date: use that date
       - If no state and no backfill: use 30 days ago (default)
    3. Fetch messages from oldest to now
    4. Normalize text
    5. Extract links & anchors
    6. Generate message_id
    7. Save to raw_slack_messages
    8. Update ingestion_state with max timestamp

    Args:
        slack_client: Slack client
        repository: Repository protocol implementation
        settings: Application settings
        lookback_hours: Override default lookback (deprecated, use backfill_from_date)
        backfill_from_date: Optional date to start backfill from (first run only)

    Returns:
        IngestResult with counts and errors

    Example:
        >>> result = ingest_messages_use_case(client, repo, settings)
        >>> result.messages_saved
        42
    """
    if lookback_hours is None:
        lookback_hours = settings.lookback_hours_default

    if lookback_hours < 0:
        raise ValueError("lookback_hours must be non-negative")

    total_fetched = 0
    total_saved = 0
    channels_processed: list[str] = []
    errors: list[str] = []

    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    now_ts = now.timestamp()
    state_store = SlackStateStore(_resolve_state_connection_factory(repository))

    for channel_config in settings.slack_channels:
        channel_id = channel_config.channel_id

        try:
            state = state_store.get(channel_id)
            max_processed_raw = state.get("max_processed_ts", 0.0)
            max_processed = (
                float(max_processed_raw)
                if isinstance(max_processed_raw, int | float)
                else 0.0
            )
            resume_cursor_value = state.get("resume_cursor")
            resume_cursor = (
                resume_cursor_value
                if isinstance(resume_cursor_value, str) and resume_cursor_value
                else None
            )
            resume_min_raw = state.get("resume_min_ts")
            resume_min_ts = (
                float(resume_min_raw)
                if isinstance(resume_min_raw, int | float)
                else None
            )

            if max_processed == 0.0 and resume_cursor is None:
                legacy_ts = repository.get_last_processed_ts(channel_id)
                if legacy_ts is not None:
                    max_processed = float(legacy_ts)

            baseline_ts: float | None = None
            oldest_ts: str | None

            if resume_cursor:
                baseline_ts = (
                    resume_min_ts
                    if resume_min_ts is not None
                    else (max_processed if max_processed > 0 else None)
                )
                oldest_ts = str(baseline_ts) if baseline_ts is not None else None
                logger.info(
                    "slack_ingestion_resume",
                    channel_id=channel_id,
                    resume_cursor=resume_cursor,
                    oldest_ts=oldest_ts,
                )
            elif max_processed > 0:
                baseline_ts = max_processed
                oldest_ts = str(baseline_ts)
                logger.info(
                    "slack_ingestion_incremental",
                    channel_id=channel_id,
                    oldest_ts=oldest_ts,
                    strategy="incremental",
                )
            else:
                lookback_dt = now - timedelta(hours=lookback_hours)
                candidates: list[tuple[datetime, dict[str, object]]] = [
                    (
                        lookback_dt,
                        {"strategy": "lookback", "lookback_hours": lookback_hours},
                    )
                ]

                if backfill_from_date:
                    backfill_dt = _ensure_utc(backfill_from_date)
                    candidates.append(
                        (
                            backfill_dt,
                            {
                                "strategy": "explicit_backfill",
                                "backfill_date": backfill_dt.isoformat(),
                                "lookback_hours": lookback_hours,
                            },
                        )
                    )

                oldest_dt, log_context = min(candidates, key=lambda item: item[0])
                baseline_ts = oldest_dt.timestamp()
                oldest_ts = str(baseline_ts)
                event_name = (
                    "slack_ingestion_first_run"
                    if log_context["strategy"] == "lookback"
                    else "slack_ingestion_backfill"
                )
                logger.info(
                    event_name,
                    channel_id=channel_id,
                    oldest_ts=oldest_ts,
                    **log_context,
                )

            raw_messages, next_cursor, has_more = _fetch_slack_messages(
                slack_client,
                channel_id,
                oldest_ts=oldest_ts,
                cursor=resume_cursor,
                limit=settings.slack_max_messages_per_run,
                page_size=settings.slack_page_size,
            )

            total_fetched += len(raw_messages)

            if resume_cursor is None:
                threshold = max_processed
                raw_messages = [
                    msg for msg in raw_messages if _parse_ts(msg.get("ts")) > threshold
                ]

            if not raw_messages:
                if resume_cursor:
                    state_store.upsert(
                        channel_id,
                        max_processed_ts=max_processed,
                        resume_cursor=None,
                        resume_min_ts=None,
                    )
                elif max_processed <= 0:
                    state_store.upsert(
                        channel_id,
                        max_processed_ts=now_ts,
                        resume_cursor=None,
                        resume_min_ts=None,
                    )
                    logger.info(
                        "slack_ingestion_state_initialized",
                        channel_id=channel_id,
                        reason="no_messages",
                    )
                channels_processed.append(channel_id)
                continue

            raw_messages.sort(key=lambda item: _parse_ts(item.get("ts")))

            processed_messages: list[SlackMessage] = []
            for raw_msg in raw_messages:
                user_info = None
                user_id = raw_msg.get("user")
                if user_id and not raw_msg.get("bot_id"):
                    try:
                        user_info = slack_client.get_user_info(user_id)
                    except Exception:
                        pass

                permalink = None
                msg_ts = raw_msg.get("ts")
                if msg_ts:
                    try:
                        permalink = slack_client.get_permalink(channel_id, msg_ts)
                    except Exception:
                        pass

                processed_msg = process_slack_message(
                    raw_msg, channel_id, user_info=user_info, permalink=permalink
                )
                processed_messages.append(processed_msg)

            saved_count = repository.save_messages(processed_messages)
            total_saved += saved_count

            limit_value = settings.slack_max_messages_per_run
            limit_hit = False
            if limit_value is not None and next_cursor is not None and has_more:
                limit_hit = len(processed_messages) >= limit_value

            resume_min_value = (
                baseline_ts
                if baseline_ts is not None
                else (max_processed if max_processed > 0 else None)
            )

            if limit_hit:
                state_store.upsert(
                    channel_id,
                    max_processed_ts=max_processed,
                    resume_cursor=next_cursor,
                    resume_min_ts=resume_min_value,
                )
                logger.info(
                    "slack_ingestion_resume_checkpoint",
                    channel_id=channel_id,
                    resume_cursor=next_cursor,
                    processed=len(processed_messages),
                )
            else:
                latest_ts_float = max(_parse_ts(msg.ts) for msg in processed_messages)
                new_max = max(max_processed, latest_ts_float)
                state_store.upsert(
                    channel_id,
                    max_processed_ts=new_max,
                    resume_cursor=None,
                    resume_min_ts=None,
                )
                logger.info(
                    "slack_ingestion_state_updated",
                    channel_id=channel_id,
                    latest_ts=f"{latest_ts_float:.6f}",
                    messages_saved=saved_count,
                )

            channels_processed.append(channel_id)

        except Exception as e:
            error_msg = f"Channel {channel_id}: {str(e)}"
            errors.append(error_msg)

    return IngestResult(
        messages_fetched=total_fetched,
        messages_saved=total_saved,
        channels_processed=channels_processed,
        errors=errors,
    )
