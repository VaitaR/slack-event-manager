#!/usr/bin/env python3
"""
Streamlit Demo App for Slack Event Manager.

This app provides a visual interface for the Slack Event Manager pipeline,
allowing users to configure settings, run the pipeline, and visualize results.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import UTC

from adapters.llm_client import LLMClient
from adapters.repository_factory import create_repository
from adapters.slack_client import SlackClient
from config.settings import get_settings
from use_cases.build_candidates import build_candidates_use_case
from use_cases.deduplicate_events import deduplicate_events_use_case
from use_cases.extract_events import extract_events_use_case
from use_cases.ingest_messages import process_slack_message

# UI Constants
MAX_MESSAGE_LENGTH = 150
MAX_CANDIDATE_TEXT_LENGTH = 200


def get_repository():
    """Get repository instance based on settings.

    Returns:
        Repository instance (SQLite or PostgreSQL)
    """
    settings = get_settings()
    return create_repository(settings)


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
                f"🐘 PostgreSQL: {settings.postgres_user}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_database}"
            )
        else:
            st.info(f"📁 SQLite: {settings.db_path}")

        # Run pipeline button
        run_pipeline = st.button(
            "🚀 Run Pipeline", type="primary", use_container_width=True
        )

    # Main content
    if run_pipeline:
        run_full_pipeline(message_limit, channels)
    else:
        show_database_inspection()


def run_full_pipeline(message_limit: int, channels: list):
    """Run the complete pipeline and show results."""

    with st.spinner("Running pipeline... This may take a few minutes."):
        try:
            # Initialize components
            settings = get_settings()
            slack_client = SlackClient(
                bot_token=settings.slack_bot_token.get_secret_value()
            )

            # gpt-5-nano requires temperature=1.0 (cannot be changed)
            temperature = (
                1.0 if settings.llm_model == "gpt-5-nano" else settings.llm_temperature
            )

            llm_client = LLMClient(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.llm_model,
                temperature=temperature,
                timeout=30,
                verbose=False,  # Disable verbose for demo
            )

            # Create repository (works for both SQLite and PostgreSQL)
            repo = create_repository(settings)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Fetch messages
            status_text.text("📨 Fetching messages from Slack...")
            progress_bar.progress(10)

            all_messages = []
            for channel in channels:
                messages = slack_client.fetch_messages(
                    channel_id=channel, limit=message_limit
                )
                all_messages.extend(messages)

            # Step 2: Process and save messages
            status_text.text("💾 Processing and saving messages...")
            progress_bar.progress(30)

            processed_messages = []
            for idx, msg in enumerate(all_messages):
                channel = channels[idx % len(channels)]

                # Get user info if available
                user_info = None
                user_id = msg.get("user")
                if user_id and not msg.get("bot_id"):
                    try:
                        user_info = slack_client.get_user_info(user_id)
                    except Exception:
                        pass  # Continue without user info

                # Get permalink
                permalink = None
                msg_ts = msg.get("ts")
                if msg_ts:
                    try:
                        permalink = slack_client.get_permalink(channel, msg_ts)
                    except Exception:
                        pass  # Continue without permalink

                processed_msg = process_slack_message(
                    msg, channel, user_info=user_info, permalink=permalink
                )
                processed_messages.append(processed_msg)

            saved_count = repo.save_messages(processed_messages)

            # Step 3: Build candidates
            status_text.text("🎯 Building event candidates...")
            progress_bar.progress(50)

            candidate_result = build_candidates_use_case(
                repository=repo,
                settings=settings,
            )

            # Step 4: Extract events with LLM
            status_text.text("🤖 Extracting events with AI...")
            progress_bar.progress(70)

            extraction_result = extract_events_use_case(
                llm_client=llm_client,
                repository=repo,
                settings=settings,
                batch_size=None,  # Process all candidates
                check_budget=False,
            )

            # Step 5: Deduplicate events
            status_text.text("🔄 Deduplicating events...")
            progress_bar.progress(90)

            dedup_result = deduplicate_events_use_case(
                repository=repo,
                settings=settings,
                lookback_days=7,
            )

            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Pipeline completed!")

            # Show results
            st.success("Pipeline completed successfully!")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Messages", saved_count)
            with col2:
                st.metric("Candidates", candidate_result.candidates_created)
            with col3:
                st.metric("Events", extraction_result.events_extracted)
            with col4:
                st.metric("Final Events", dedup_result.total_events)

            # Cost information
            if extraction_result.total_cost_usd > 0:
                st.info(f"💰 Total LLM cost: ${extraction_result.total_cost_usd:.6f}")

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            return

    # Show detailed results
    show_pipeline_results()


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
        # Get repository and fetch ALL messages for UI display
        repo = get_repository()
        settings = get_settings()

        # Show which database we're using
        if settings.database_type == "postgres":
            st.caption(
                f"🐘 Source: PostgreSQL ({settings.postgres_host}:{settings.postgres_port}/{settings.postgres_database})"
            )
        else:
            st.caption(f"📁 Source: SQLite ({settings.db_path})")

        # Get all messages for UI display (not just unprocessed)
        # For now, we'll fetch pending + some processed (PostgreSQL-specific query)
        from src.adapters.postgres_repository import PostgresRepository
        from src.adapters.sqlite_repository import SQLiteRepository

        if isinstance(repo, PostgresRepository):
            # Direct SQL for PostgreSQL to get all messages
            with repo._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=__import__(
                        "psycopg2.extras", fromlist=["RealDictCursor"]
                    ).RealDictCursor
                ) as cur:
                    cur.execute("""
                        SELECT m.* FROM raw_slack_messages m
                        ORDER BY m.ts_dt DESC
                        LIMIT 100
                    """)
                    rows = cur.fetchall()
                    messages = [repo._row_to_message(dict(row)) for row in rows]
        elif isinstance(repo, SQLiteRepository):
            # Direct SQL for SQLite
            import sqlite3

            conn = sqlite3.connect(settings.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM raw_slack_messages
                ORDER BY ts_dt DESC
                LIMIT 100
            """)
            rows = cur.fetchall()
            messages = [repo._row_to_message(dict(row)) for row in rows]
            conn.close()
        else:
            messages = repo.get_new_messages_for_candidates()

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
                    "ts": pd.to_datetime(msg.ts, unit="s"),
                    "user_real_name": msg.user_real_name or "",
                    "user_email": msg.user_email or "",
                    "total_reactions": msg.total_reactions or 0,
                    "reply_count": msg.reply_count or 0,
                    "attachments_count": msg.attachments_count or 0,
                    "files_count": msg.files_count or 0,
                    "permalink": msg.permalink or "",
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
        # Get repository and fetch ALL candidates for UI display
        repo = get_repository()
        settings = get_settings()

        from src.adapters.postgres_repository import PostgresRepository
        from src.adapters.sqlite_repository import SQLiteRepository

        if isinstance(repo, PostgresRepository):
            # Direct SQL for PostgreSQL to get all candidates
            with repo._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=__import__(
                        "psycopg2.extras", fromlist=["RealDictCursor"]
                    ).RealDictCursor
                ) as cur:
                    cur.execute("""
                        SELECT * FROM event_candidates
                        ORDER BY score DESC
                        LIMIT 100
                    """)
                    rows = cur.fetchall()
                    candidates = [repo._row_to_candidate(dict(row)) for row in rows]
        elif isinstance(repo, SQLiteRepository):
            # Direct SQL for SQLite
            import sqlite3

            conn = sqlite3.connect(settings.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM event_candidates
                ORDER BY score DESC
                LIMIT 100
            """)
            rows = cur.fetchall()
            candidates = [repo._row_to_candidate(dict(row)) for row in rows]
            conn.close()
        else:
            candidates = repo.get_candidates_for_extraction(batch_size=100)

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
                    "status": cand.status,
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
        # Get repository and fetch events
        repo = get_repository()
        # Get recent events (last 90 days)
        from datetime import datetime, timedelta

        start_date = datetime.now(UTC) - timedelta(days=90)
        end_date = datetime.now(UTC) + timedelta(days=365)
        events = repo.get_events_in_window(start_date, end_date)

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
        from datetime import datetime, timedelta

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
