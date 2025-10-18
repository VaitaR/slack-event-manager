"""LLM client adapter for event extraction.

Implements LLMClientProtocol with OpenAI integration.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pytz
import yaml
from openai import APIError, OpenAI
from openai import RateLimitError as OpenAIRateLimitError

from src.domain.exceptions import LLMAPIError, ValidationError
from src.domain.models import LLMCallMetadata, LLMResponse

# Token cost per 1M tokens (as of 2025-10)
TOKEN_COSTS: Final[dict[str, dict[str, float]]] = {
    "gpt-5-nano": {
        "input": 0.075,
        "output": 0.300,
    },  # per 1M tokens - 75% cheaper than gpt-4o-mini
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},  # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

# Preview lengths for logging (characters)
PREVIEW_LENGTH_PROMPT: Final[int] = 800
"""Maximum characters to show in prompt preview for logging."""

PREVIEW_LENGTH_RESPONSE: Final[int] = 1000
"""Maximum characters to show in response preview for logging."""


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptFileData:
    """Loaded prompt payload with metadata."""

    content: str
    version: str | None
    checksum: str
    size_bytes: int
    path: Path


@dataclass
class _PromptCacheEntry:
    """Cache entry storing metadata for a prompt file."""

    mtime: float
    data: PromptFileData


_PROMPT_CACHE: dict[Path, _PromptCacheEntry] = {}

DEFAULT_PROMPT_PATH: Final[Path] = Path("config/prompts/slack.yaml")


def load_prompt_from_file(file_path: str) -> PromptFileData:
    """Load prompt template from a file with caching and metadata.

    Args:
        file_path: Path to the prompt file

    Returns:
        Prompt payload metadata

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If YAML prompt file has invalid structure
    """

    raw_path = Path(file_path).expanduser()
    path = raw_path if raw_path.is_absolute() else (Path.cwd() / raw_path).resolve()

    if not path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt_path = (repo_root / raw_path).resolve()
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

    stat_result = path.stat()
    cache_entry = _PROMPT_CACHE.get(path)
    if cache_entry and cache_entry.mtime == stat_result.st_mtime:
        return cache_entry.data

    if path.suffix.lower() in {".yaml", ".yml"}:
        raw_text = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Prompt YAML must be a mapping: {path}")

        version = parsed.get("version")
        if not isinstance(version, str):
            raise ValueError(f"Prompt YAML missing 'version' string: {path}")

        system_prompt = parsed.get("system")
        if not isinstance(system_prompt, str):
            raise ValueError(f"Prompt YAML missing 'system' string: {path}")
    else:
        system_prompt = path.read_text(encoding="utf-8")
        version = None

    checksum = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    size_bytes = len(system_prompt.encode("utf-8"))
    prompt_data = PromptFileData(
        content=system_prompt,
        version=version,
        checksum=checksum,
        size_bytes=size_bytes,
        path=path,
    )

    _PROMPT_CACHE[path] = _PromptCacheEntry(
        mtime=stat_result.st_mtime, data=prompt_data
    )
    return prompt_data


class LLMClient:
    """OpenAI LLM client for event extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-nano",
        temperature: float | None = None,
        timeout: int = 30,
        verbose: bool = False,
        prompt_template: str | None = None,
        prompt_file: str | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature (1.0 for gpt-5-nano, 0.7 for gpt-4o-mini)
            timeout: Request timeout in seconds
            verbose: If True, log full prompts and responses
            prompt_template: Custom prompt template (optional)
            prompt_file: Path to prompt file (takes precedence over prompt_template)
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.verbose = verbose

        # Set optimal temperature based on model if not provided
        if temperature is None:
            self.temperature = 1.0 if model == "gpt-5-nano" else 0.7
        else:
            # gpt-5-nano only supports temperature=1.0
            if model == "gpt-5-nano" and temperature != 1.0:
                print(
                    f"⚠️ Warning: gpt-5-nano only supports temperature=1.0, using {temperature} as requested"
                )
            self.temperature = temperature

        self._prompt_metadata: PromptFileData | None = None
        self._system_prompt_path: Path | None = None
        self.prompt_version: str | None = None

        if prompt_file:
            prompt_data = load_prompt_from_file(prompt_file)
            self.system_prompt = prompt_data.content
            self._prompt_metadata = prompt_data
            self._system_prompt_path = prompt_data.path
            self.prompt_version = prompt_data.version
            self._system_prompt_size_bytes = prompt_data.size_bytes
        elif prompt_template:
            self.system_prompt = prompt_template
            self._system_prompt_size_bytes = len(self.system_prompt.encode("utf-8"))
        else:
            prompt_data = load_prompt_from_file(str(DEFAULT_PROMPT_PATH))
            self.system_prompt = prompt_data.content
            self._prompt_metadata = prompt_data
            self._system_prompt_path = prompt_data.path
            self.prompt_version = prompt_data.version
            self._system_prompt_size_bytes = prompt_data.size_bytes

        self._system_prompt_hash = hashlib.sha256(
            self.system_prompt.encode("utf-8")
        ).hexdigest()

        logger.info(
            "LLM system prompt ready",
            extra={
                "prompt_hash": self._system_prompt_hash,
                "prompt_version": self.prompt_version,
                "prompt_path": str(self._system_prompt_path)
                if self._system_prompt_path
                else "<inline>",
                "prompt_size_bytes": self._system_prompt_size_bytes,
            },
        )

        self._last_call_metadata: LLMCallMetadata | None = None

    def extract_events(
        self,
        text: str,
        links: list[str],
        message_ts_dt: datetime,
        channel_name: str = "",
    ) -> LLMResponse:
        """Extract events from message text using LLM.

        Args:
            text: Normalized message text
            links: Top 3 most relevant links
            message_ts_dt: Message timestamp for date resolution fallback
            channel_name: Channel name for context

        Returns:
            Structured LLM response

        Raises:
            LLMAPIError: On API communication errors
            ValidationError: On response validation failure
        """
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(text, links, message_ts_dt, channel_name)

        # Log request details
        import sys

        print("   📤 LLM Request:")
        print(f"      Model: {self.model}")
        print(f"      Temperature: {self.temperature}")
        print(f"      Prompt length: {len(prompt)} chars")
        print(f"      System prompt length: {len(self.system_prompt)} chars")

        if self.verbose:
            print("\n   === SYSTEM PROMPT ===")
            print(f"   {self.system_prompt[:500]}...")
            print("\n   === USER PROMPT ===")
            print(f"   {prompt[:PREVIEW_LENGTH_PROMPT]}...")
            if len(prompt) > PREVIEW_LENGTH_PROMPT:
                print(f"   ... ({len(prompt) - PREVIEW_LENGTH_PROMPT} more chars)")

        sys.stdout.flush()

        try:
            # Call OpenAI with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response
            content = response.choices[0].message.content
            if not content:
                raise ValidationError("Empty response from LLM")

            # Parse JSON
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON from LLM: {e}")

            # Validate with Pydantic
            try:
                llm_response = LLMResponse.model_validate(response_data)
            except Exception as e:
                raise ValidationError(f"Response validation failed: {e}")

            # Calculate cost
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            cost_usd = self._calculate_cost(tokens_in, tokens_out)

            # Log response details
            print("   📥 LLM Response:")
            print(f"      Latency: {latency_ms}ms ({latency_ms / 1000:.2f}s)")
            print(f"      Tokens IN: {tokens_in}")
            print(f"      Tokens OUT: {tokens_out}")
            print(f"      Total tokens: {tokens_in + tokens_out}")
            print(f"      Cost: ${cost_usd:.6f}")
            print(f"      Is event: {llm_response.is_event}")
            print(f"      Events extracted: {len(llm_response.events)}")
            if llm_response.events:
                for i, evt in enumerate(llm_response.events, 1):
                    print(
                        f"         {i}. {evt.action} {evt.object_name_raw} ({evt.category})"
                    )

            if self.verbose and content:
                print("\n   === RAW JSON RESPONSE ===")
                print(f"   {content[:PREVIEW_LENGTH_RESPONSE]}...")
                if len(content) > PREVIEW_LENGTH_RESPONSE:
                    print(
                        f"   ... ({len(content) - PREVIEW_LENGTH_RESPONSE} more chars)"
                    )

            sys.stdout.flush()

            # Store metadata
            self._last_call_metadata = LLMCallMetadata(
                message_id="",  # Will be set by caller
                prompt_hash=self._system_prompt_hash,
                model=self.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                cached=False,
                ts=datetime.utcnow().replace(tzinfo=pytz.UTC),
            )

            return llm_response

        except OpenAIRateLimitError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            print(f"   ❌ Rate limit after {latency_ms}ms: {e}")
            sys.stdout.flush()
            raise LLMAPIError(f"Rate limit exceeded: {e}")
        except APIError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            print(f"   ❌ API error after {latency_ms}ms: {e}")
            sys.stdout.flush()
            raise LLMAPIError(f"OpenAI API error: {e}")
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            if isinstance(e, ValidationError | LLMAPIError):
                print(f"   ❌ Error after {latency_ms}ms: {e}")
                sys.stdout.flush()
                raise
            print(f"   ❌ Unexpected error after {latency_ms}ms: {e}")
            sys.stdout.flush()
            raise LLMAPIError(f"Unexpected error: {e}")

    def extract_events_with_retry(
        self,
        text: str,
        links: list[str],
        message_ts_dt: datetime,
        channel_name: str = "",
        max_retries: int = 3,
    ) -> LLMResponse:
        """Extract events with retry on failures (timeout, rate limit, validation).

        Args:
            text: Message text
            links: Links
            message_ts_dt: Message timestamp
            channel_name: Channel name
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            LLM response

        Raises:
            LLMAPIError: On API errors after all retries
            ValidationError: On validation failure after all retries
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return self.extract_events(text, links, message_ts_dt, channel_name)
            except (ValidationError, LLMAPIError) as e:
                last_error = e
                error_msg = str(e)

                # Determine if we should retry
                is_timeout = (
                    "timed out" in error_msg.lower() or "timeout" in error_msg.lower()
                )
                is_rate_limit = "rate limit" in error_msg.lower()
                is_validation = isinstance(e, ValidationError)

                should_retry = is_timeout or is_rate_limit or is_validation

                if attempt < max_retries and should_retry:
                    # Calculate backoff delay
                    if is_rate_limit:
                        delay = 10 * (attempt + 1)  # 10s, 20s, 30s for rate limits
                    elif is_timeout:
                        delay = 5 * (attempt + 1)  # 5s, 10s, 15s for timeouts
                    else:
                        delay = 2 * (attempt + 1)  # 2s, 4s, 6s for validation

                    print(f"   ⚠️ Retry {attempt + 1}/{max_retries}: {error_msg}")
                    print(f"   ⏳ Waiting {delay}s before retry...")
                    import sys

                    sys.stdout.flush()

                    time.sleep(delay)
                    continue

                # No more retries or non-retriable error
                raise

        # Should not reach here, but for type safety
        if isinstance(last_error, ValidationError):
            raise ValidationError(
                f"Failed after {max_retries + 1} attempts: {last_error}"
            )
        else:
            raise LLMAPIError(f"Failed after {max_retries + 1} attempts: {last_error}")

    def get_call_metadata(self) -> LLMCallMetadata:
        """Get metadata for last LLM call.

        Returns:
            Call metadata

        Raises:
            RuntimeError: If no call has been made
        """
        if self._last_call_metadata is None:
            raise RuntimeError("No LLM call has been made yet")

        return self._last_call_metadata

    def _build_prompt(
        self, text: str, links: list[str], message_ts_dt: datetime, channel_name: str
    ) -> str:
        """Build user prompt for LLM.

        Args:
            text: Message text
            links: Links
            message_ts_dt: Message timestamp
            channel_name: Channel name

        Returns:
            Formatted prompt
        """
        ts_str = message_ts_dt.strftime("%Y-%m-%d %H:%M UTC")

        prompt_parts = [
            f"Channel: #{channel_name}" if channel_name else "",
            f"Message timestamp: {ts_str}",
            f"\nMessage text:\n{text}",
        ]

        if links:
            prompt_parts.append(
                "\nRelevant links:\n" + "\n".join(f"- {link}" for link in links[:3])
            )

        return "\n".join(part for part in prompt_parts if part)

    def _calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost for API call.

        Args:
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Cost in USD
        """
        if self.model not in TOKEN_COSTS:
            # Unknown model, use gpt-5-nano pricing as fallback
            costs = TOKEN_COSTS["gpt-5-nano"]
        else:
            costs = TOKEN_COSTS[self.model]

        cost_in = (tokens_in / 1_000_000) * costs["input"]
        cost_out = (tokens_out / 1_000_000) * costs["output"]

        return cost_in + cost_out
