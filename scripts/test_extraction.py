#!/usr/bin/env python
"""Quick test: extract events from candidates."""

from src.adapters.llm_client import LLMClient
from src.adapters.repository_factory import create_repository
from src.config.settings import Settings, get_settings
from src.domain.models import MessageSource
from src.domain.protocols import RepositoryProtocol
from src.services.importance_scorer import ImportanceScorer
from src.use_cases.extract_events import build_object_registry, extract_events_use_case

if __name__ == "__main__":
    settings: Settings = get_settings()
    llm_client = LLMClient(
        settings.openai_api_key.get_secret_value(),
        settings.llm_model,
        settings.llm_temperature,
    )
    repository: RepositoryProtocol = create_repository(settings)
    object_registry = build_object_registry(settings)
    importance_scorer = ImportanceScorer()

    print("Extracting events from candidates (batch_size=5)...")
    result = extract_events_use_case(
        llm_client,
        repository,
        settings,
        source_id=MessageSource.SLACK,
        batch_size=5,
        check_budget=False,
        object_registry=object_registry,
        importance_scorer=importance_scorer,
    )
    print(f"\n✅ Events extracted: {result.events_extracted}")
    print(f"✅ LLM calls: {result.llm_calls}")
    print(f"✅ Cost: ${result.total_cost_usd:.4f}")
    print(f"⚠️ Errors: {len(result.errors)}")

    # Check database
    import sqlite3

    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM events")
    event_count = cursor.fetchone()[0]
    print(f"\n📊 Events in DB: {event_count}")
    conn.close()
