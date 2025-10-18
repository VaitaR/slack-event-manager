"""Object registry service for canonicalizing object names.

Maps raw object names from text to canonical object IDs using synonyms.
"""

from pathlib import Path
from typing import Final

import yaml
from rapidfuzz import fuzz

# Fuzzy matching threshold for near-matches
FUZZY_MATCH_THRESHOLD: Final[float] = 0.9


class ObjectRegistry:
    """Registry for mapping object names to canonical IDs."""

    def __init__(self, registry_path: str | Path) -> None:
        """Initialize object registry from YAML file.

        Args:
            registry_path: Path to registry YAML file

        Raises:
            FileNotFoundError: If registry file doesn't exist
            ValueError: If registry format is invalid
        """
        self.registry_path = Path(registry_path)
        self.objects: dict[str, list[str]] = {}
        self._reverse_index: dict[str, str] = {}

        self._load_registry()
        self._build_reverse_index()

    def _load_registry(self) -> None:
        """Load registry from YAML file."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {self.registry_path}")

        with open(self.registry_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "objects" not in data:
            raise ValueError("Registry must have 'objects' key")

        self.objects = data["objects"]

    def _build_reverse_index(self) -> None:
        """Build reverse index: synonym -> object_id."""
        self._reverse_index = {}
        for object_id, synonyms in self.objects.items():
            for synonym in synonyms:
                # Normalize: lowercase, strip whitespace
                normalized = synonym.lower().strip()
                self._reverse_index[normalized] = object_id

    def canonicalize_object(self, raw_name: str) -> str | None:
        """Map raw object name to canonical object_id.

        Uses exact match first, then fuzzy matching for near-matches.

        Args:
            raw_name: Raw object name from text

        Returns:
            Canonical object_id if found, None otherwise

        Example:
            >>> from src.config.settings import get_settings
            >>> settings = get_settings()
            >>> registry = ObjectRegistry(settings.object_registry_path)
            >>> registry.canonicalize_object("Stocks & ETFs")
            'wallet.stocks'
            >>> registry.canonicalize_object("CH cluster")
            'data.clickhouse'
            >>> registry.canonicalize_object("Unknown System")
            None
        """
        if not raw_name:
            return None

        normalized = raw_name.lower().strip()

        # Exact match
        if normalized in self._reverse_index:
            return self._reverse_index[normalized]

        # Fuzzy matching (for typos/variations)
        best_match_score = 0.0
        best_match_id = None

        for synonym, object_id in self._reverse_index.items():
            score = fuzz.ratio(normalized, synonym) / 100.0
            if score >= FUZZY_MATCH_THRESHOLD and score > best_match_score:
                best_match_score = score
                best_match_id = object_id

        return best_match_id

    def get_synonyms(self, object_id: str) -> list[str]:
        """Get all synonyms for an object_id.

        Args:
            object_id: Canonical object ID

        Returns:
            List of synonyms

        Example:
            >>> registry.get_synonyms("wallet.stocks")
            ['Stocks & ETFs', 'Stock trading', 'Equity wallet']
        """
        return self.objects.get(object_id, [])

    def get_all_object_ids(self) -> list[str]:
        """Get all registered object IDs.

        Returns:
            List of canonical object IDs
        """
        return list(self.objects.keys())
