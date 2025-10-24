#!/usr/bin/env python3
"""
Streamlit Demo App for Slack Event Manager.

This app provides a visual interface for the Slack Event Manager pipeline,
allowing users to configure settings, run the pipeline, and visualize results.
"""

import hmac
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from src.adapters.repository_factory import create_repository
from src.config.settings import get_settings
from src.domain.protocols import RepositoryProtocol
from src.observability.metrics import ensure_metrics_exporter
from src.presentation.streamlit_orchestration import (
    RateLimitExceededError,
    job_result,
    job_status,
    submit_ingest_extract_job,
)
from src.use_cases.dashboard_queries import (
    fetch_recent_candidates,
    fetch_recent_events,
    fetch_recent_messages,
)
from src.use_cases.pipeline_orchestrator import PipelineParams

# UI Constants
MAX_MESSAGE_LENGTH: Final[int] = 150
MAX_CANDIDATE_TEXT_LENGTH: Final[int] = 200
AUTH_TOKEN_ENV: Final[str] = "STREAMLIT_AUTH_TOKEN"
SESSION_AUTH_KEY: Final[str] = "auth_verified"
SESSION_ID_KEY: Final[str] = "session_id"
SESSION_JOB_ID_KEY: Final[str] = "active_job_id"
SESSION_JOB_RESULT_KEY: Final[str] = "last_job_result"
SESSION_STATUS_MESSAGE_KEY: Final[str] = "job_status_message"
POLL_INTERVAL_MS: Final[int] = 1500


def get_repository() -> RepositoryProtocol:
    """Get repository instance based on settings.

    Returns:
        Repository instance (SQLite or PostgreSQL)
    """
    settings = get_settings()
    return create_repository(settings)


def _ensure_session_defaults() -> None:
    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = str(uuid4())
    st.session_state.setdefault(SESSION_JOB_ID_KEY, None)
    st.session_state.setdefault(SESSION_JOB_RESULT_KEY, None)
    st.session_state.setdefault(SESSION_STATUS_MESSAGE_KEY, "")


def _require_auth() -> str:
    expected = os.getenv(AUTH_TOKEN_ENV)
    if not expected:
        st.error("Access is disabled until STREAMLIT_AUTH_TOKEN is configured.")
        st.stop()

    if st.session_state.get(SESSION_AUTH_KEY):
        return str(st.session_state[SESSION_ID_KEY])

    st.warning("Authentication required to run the pipeline.")
    with st.form("auth_form", clear_on_submit=False):
        provided = st.text_input("Access Token", type="password")
        submit = st.form_submit_button("Sign In")

    if not submit:
        st.stop()

    if not provided or not hmac.compare_digest(provided, expected):
        st.error("Invalid access token.")
        st.stop()

    st.session_state[SESSION_AUTH_KEY] = True
    return str(st.session_state[SESSION_ID_KEY])


def _submit_pipeline_job(message_limit: int, channels: list[str], user_id: str) -> str:
    params = PipelineParams(message_limit=message_limit, channel_ids=channels)
    job_id = submit_ingest_extract_job(params, user_id)
    st.session_state[SESSION_JOB_ID_KEY] = job_id
    st.session_state[SESSION_JOB_RESULT_KEY] = None
    st.session_state[SESSION_STATUS_MESSAGE_KEY] = "Job submitted"
    return job_id


def _render_job_status(job_id: str) -> None:
    status = job_status(job_id)
    progress_value = float(status.get("progress", 0.0))
    st.progress(progress_value)
    status_raw = str(status.get("status", "unknown")).lower()
    status_text = status_raw.capitalize()
    message = status.get("message") or ""
    st.info(f"Status: {status_text} {message}")

    if status_raw in {"succeeded", "failed"}:
        final = job_result(job_id)
        st.session_state[SESSION_JOB_RESULT_KEY] = final
        st.session_state[SESSION_JOB_ID_KEY] = None
        st.session_state[SESSION_STATUS_MESSAGE_KEY] = status_text
        if status_raw == "failed":
            error_msg = status.get("error") or "Pipeline failed"
            st.error(error_msg)
        else:
            st.success("Pipeline completed successfully")


def _render_job_summary(result: dict[str, object]) -> None:
    st.subheader("Latest Pipeline Run")
    correlation_id = result.get("correlation_id")
    if correlation_id:
        st.caption(f"Correlation ID: {correlation_id}")

    ingest = result.get("ingest", {})
    extract = result.get("extract", {})
    dedup = result.get("dedup", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Messages Saved",
            ingest.get("messages_saved", 0),
            help="Total messages persisted during ingestion",
        )
    with col2:
        st.metric(
            "Events Extracted",
            extract.get("events_extracted", 0),
            help="Events produced by the extraction stage",
        )
    with col3:
        st.metric(
            "Final Events",
            dedup.get("total_events", 0),
            help="Events remaining after deduplication",
        )


# Page configuration
st.set_page_config(
    page_title="Slack Event Manager",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main application function."""

    ensure_metrics_exporter()
    _ensure_session_defaults()
    user_id = _require_auth()

    # Header
    st.markdown(
        '<h1 class="main-header">📅 Slack Event Manager</h1>', unsafe_allow_html=True
    )
    st.markdown(
        "Visual interface for processing Slack messages and extracting structured events."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Load settings
        settings = get_settings()

        # Basic settings
        message_limit = st.slider(
            "Message Limit",
            min_value=5,
            max_value=100,
            value=20,
            help="Number of recent messages to fetch from each channel",
        )

        # Channel selection from config
        # Create mapping: channel_name -> channel_id
        channel_options = {
            f"{ch.channel_name} ({ch.channel_id})": ch.channel_id
            for ch in settings.slack_channels
        }

        # Use channel names as display options
        selected_channel_names = st.multiselect(
            "Channels",
            options=list(channel_options.keys()),
            default=list(channel_options.keys()),  # All channels by default
            help="Select Slack channels to process",
        )

        # Convert back to channel IDs
        channels = [channel_options[name] for name in selected_channel_names]

        # Show database info
        if settings.database_type == "postgres":
            st.info(
                "🐘 PostgreSQL: "
                f"{settings.postgres_user}@{settings.postgres_host}:"
                f"{settings.postgres_port}/{settings.postgres_database}"
            )
        else:
            st.info(f"📁 SQLite: {settings.db_path}")

        # Run pipeline button
        run_pipeline = st.button(
            "🚀 Run Pipeline", type="primary", use_container_width=True
        )

        if run_pipeline:
            try:
                job_id = _submit_pipeline_job(message_limit, channels, user_id)
                st.success(f"Pipeline job submitted (ID: {job_id[:8]}…)")
            except RateLimitExceededError as exc:
                st.error(str(exc))

    active_job_id = st.session_state.get(SESSION_JOB_ID_KEY)
    if active_job_id:
        st.autorefresh(interval=POLL_INTERVAL_MS, key=f"poll_{active_job_id}")
        _render_job_status(active_job_id)
    else:
        latest_result = st.session_state.get(SESSION_JOB_RESULT_KEY)
        if isinstance(latest_result, dict):
            _render_job_summary(latest_result)
        elif st.session_state.get(SESSION_STATUS_MESSAGE_KEY):
            st.info(st.session_state[SESSION_STATUS_MESSAGE_KEY])

    show_database_inspection()


def fetch_channel_messages(
    slack_client: Any, *, channels: list[str], limit: int | None
) -> list[tuple[str, dict[str, Any]]]:
    """Fetch messages for each channel while preserving attribution."""

    channel_messages: list[tuple[str, dict[str, Any]]] = []
    for channel in channels:
        messages = slack_client.fetch_messages(channel_id=channel, limit=limit)
        channel_messages.extend((channel, message) for message in messages)
    return channel_messages


def show_pipeline_results():
    """Show detailed results from the pipeline."""

    st.header("📊 Pipeline Results")

    # Database inspection using repository
    try:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📨 Messages", "🎯 Candidates", "📝 Events", "📈 Timeline"]
        )

        with tab1:
            show_messages_table()

        with tab2:
            show_candidates_table()

        with tab3:
            show_events_table()

        with tab4:
            show_gantt_chart()

    except Exception as e:
        st.error(f"Error reading database: {str(e)}")


def show_database_inspection():
    """Show database inspection when pipeline hasn't been run."""

    st.header("🔍 Database Inspection")

    try:
        # Use repository for inspection
        show_pipeline_results()
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")


def show_messages_table():
    """Display messages table."""

    st.subheader("📨 Raw Messages")

    try:
        repo = get_repository()
        settings = get_settings()

        if settings.database_type == "postgres":
            st.caption(
                "🐘 Source: PostgreSQL ("
                f"{settings.postgres_host}:{settings.postgres_port}/"
                f"{settings.postgres_database})"
            )
        else:
            st.caption(f"📁 Source: SQLite ({settings.db_path})")

        messages = fetch_recent_messages(repository=repo, limit=100)

        if not messages:
            st.info("No messages found.")
            return

        # Convert to DataFrame
        messages_data = []
        for msg in messages:
            messages_data.append(
                {
                    "message_id": msg.message_id,
                    "text": msg.text[:MAX_MESSAGE_LENGTH] + "..."
                    if len(msg.text) > MAX_MESSAGE_LENGTH
                    else msg.text,
                    "ts": msg.ts_dt,
                    "user_real_name": msg.user_real_name or "",
                    "user_email": msg.user_email or "",
                    "total_reactions": msg.total_reactions or 0,
                    "reply_count": msg.reply_count or 0,
                    "attachments_count": msg.attachments_count or 0,
                    "files_count": msg.files_count or 0,
                    "permalink": msg.permalink or "",
                    "edited_ts": msg.edited_ts,
                    "edited": msg.edited_ts is not None,
                }
            )

        messages_df = pd.DataFrame(messages_data)

        if messages_df.empty:
            st.info("No messages found.")
            return

        st.dataframe(
            messages_df,
            use_container_width=True,
            column_config={
                "message_id": st.column_config.TextColumn("Message ID", width="medium"),
                "text": st.column_config.TextColumn("Text", width="large"),
                "ts": st.column_config.DatetimeColumn(
                    "Timestamp", format="YYYY-MM-DD HH:mm:ss"
                ),
                "user_real_name": st.column_config.TextColumn("User", width="medium"),
                "user_email": st.column_config.TextColumn("Email", width="medium"),
                "total_reactions": st.column_config.NumberColumn(
                    "👍 Reactions", width="small"
                ),
                "reply_count": st.column_config.NumberColumn(
                    "💬 Replies", width="small"
                ),
                "attachments_count": st.column_config.NumberColumn(
                    "📎 Files", width="small"
                ),
                "files_count": st.column_config.NumberColumn("📄 Docs", width="small"),
                "permalink": st.column_config.LinkColumn("🔗 Link", width="small"),
                "edited": st.column_config.CheckboxColumn("✏️ Edited", width="small"),
            },
            hide_index=True,
        )

        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", len(messages_df))
        with col2:
            st.metric("Total Reactions", int(messages_df["total_reactions"].sum()))
        with col3:
            st.metric("Total Replies", int(messages_df["reply_count"].sum()))
        with col4:
            edited_count = (
                messages_df["edited_ts"].notna().sum()
                if "edited_ts" in messages_df.columns
                else 0
            )
            st.metric("Edited Messages", int(edited_count))

    except Exception as e:
        st.error(f"Error loading messages: {str(e)}")


def show_candidates_table():
    """Display candidates table."""

    st.subheader("🎯 Event Candidates")

    try:
        repo = get_repository()
        candidates = fetch_recent_candidates(repository=repo, limit=100)

        if not candidates:
            st.info("No candidates found.")
            return

        # Convert to DataFrame
        candidates_data = []
        for cand in candidates:
            candidates_data.append(
                {
                    "message_id": cand.message_id,
                    "text_norm": cand.text_norm[:MAX_CANDIDATE_TEXT_LENGTH] + "..."
                    if len(cand.text_norm) > MAX_CANDIDATE_TEXT_LENGTH
                    else cand.text_norm,
                    "score": cand.score,
                    "status": cand.status.value,
                    "features_json": str(
                        cand.features.model_dump() if cand.features else {}
                    ),
                }
            )

        candidates_df = pd.DataFrame(candidates_data)

        if candidates_df.empty:
            st.info("No candidates found.")
            return

        st.dataframe(
            candidates_df,
            use_container_width=True,
            column_config={
                "message_id": st.column_config.TextColumn("Message ID", width="medium"),
                "text_norm": st.column_config.TextColumn("Text", width="large"),
                "score": st.column_config.NumberColumn("Score", format="%.2f"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "features_json": st.column_config.TextColumn(
                    "Features", width="medium"
                ),
            },
        )

        st.caption(f"Total candidates: {len(candidates_df)}")

    except Exception as e:
        st.error(f"Error loading candidates: {str(e)}")


def show_events_table():
    """Display events table."""

    st.subheader("📝 Extracted Events")

    try:
        repo = get_repository()
        events = fetch_recent_events(repository=repo, limit=100)

        if not events:
            st.info("No events found.")
            return

        # Convert to DataFrame with new event structure
        events_data = []
        for evt in events:
            events_data.append(
                {
                    "event_id": evt.event_id,
                    "message_id": evt.message_id,
                    "title": evt.title,  # Property that renders from slots
                    "category": evt.category.value,
                    "status": evt.status.value,
                    "event_date": evt.event_date,  # Property that returns first non-None time
                    "confidence": evt.confidence,
                    "importance": evt.importance,
                    "cluster_key": evt.cluster_key,
                    "dedup_key": evt.dedup_key or "",
                }
            )

        events_df = pd.DataFrame(events_data)

        if events_df.empty:
            st.info("No events found.")
            return

        st.dataframe(
            events_df,
            use_container_width=True,
            column_config={
                "event_id": st.column_config.TextColumn("Event ID", width="medium"),
                "title": st.column_config.TextColumn("Title", width="large"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "event_date": st.column_config.DatetimeColumn(
                    "Date", format="YYYY-MM-DD HH:mm"
                ),
                "confidence": st.column_config.NumberColumn(
                    "Confidence", format="%.2f"
                ),
                "importance": st.column_config.NumberColumn(
                    "Importance", width="small"
                ),
                "message_id": st.column_config.TextColumn(
                    "Source Message", width="medium"
                ),
                "cluster_key": st.column_config.TextColumn(
                    "Cluster Key", width="small"
                ),
                "dedup_key": st.column_config.TextColumn("Dedup Key", width="small"),
            },
            hide_index=True,
        )

        st.caption(f"Total events: {len(events_df)}")

    except Exception as e:
        st.error(f"Error loading events: {str(e)}")


def show_gantt_chart():
    """Display Gantt chart visualization."""

    st.subheader("📈 Events Timeline (Gantt Chart)")

    try:
        # Get repository and fetch events with dates
        repo = get_repository()

        start_date = datetime.now(UTC) - timedelta(days=90)
        end_date = datetime.now(UTC) + timedelta(days=365)
        events = repo.get_events_in_window(start_date, end_date)

        if not events:
            st.info("No events with dates found for timeline.")
            return

        # Convert to DataFrame - only events with dates
        events_data = []
        for evt in events:
            if evt.event_date:
                events_data.append(
                    {
                        "title": evt.title,  # Property that renders from slots
                        "category": evt.category.value,
                        "event_date": evt.event_date,  # Property that returns first non-None time
                    }
                )

        events_df = pd.DataFrame(events_data)

        if events_df.empty:
            st.info("No events with dates found for timeline.")
            return

        # Create Gantt chart
        fig = px.timeline(
            events_df,
            x_start=events_df["event_date"],
            x_end=events_df["event_date"] + pd.Timedelta(days=1),  # Events span 1 day
            y="title",
            color="category",
            title="Events Timeline",
            labels={"event_date": "Date"},
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Timeline",
            yaxis_title="Events",
            showlegend=True,
            height=max(
                400, len(events_df) * 30
            ),  # Dynamic height based on number of events
        )

        # Update y-axis to show full titles
        fig.update_yaxes(automargin=True)

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", len(events_df))
        with col2:
            st.metric("Categories", events_df["category"].nunique())
        with col3:
            date_range = (
                events_df["event_date"].max() - events_df["event_date"].min()
            ).days
            st.metric("Date Range", f"{date_range} days")

    except Exception as e:
        st.error(f"Error creating Gantt chart: {str(e)}")


if __name__ == "__main__":
    main()
