"""LLM client adapter for event extraction.

Implements LLMClientProtocol with OpenAI integration.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pytz
import yaml
from openai import APIError, OpenAI
from openai import RateLimitError as OpenAIRateLimitError

from src.config.logging_config import get_logger
from src.domain.exceptions import LLMAPIError, ValidationError
from src.domain.models import LLMCallMetadata, LLMResponse

logger = get_logger(__name__)

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

VERBOSE_ENV_FLAG: Final[str] = "LLM_ALLOW_VERBOSE_LOGS"


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
        self._verbose_requested = verbose
        self.verbose = verbose and self._is_verbose_allowed()

        # Set optimal temperature based on model if not provided
        if temperature is None:
            self.temperature = 1.0 if model == "gpt-5-nano" else 0.7
        else:
            # gpt-5-nano only supports temperature=1.0
            if model == "gpt-5-nano" and temperature != 1.0:
                logger.warning(
                    "gpt5_nano_temperature_override",
                    model=model,
                    requested_temperature=temperature,
                    note="gpt-5-nano only supports temperature=1.0",
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

        if self._verbose_requested and not self.verbose:
            logger.warning(
                "llm_verbose_disabled",
                env_flag=VERBOSE_ENV_FLAG,
                reason="env_flag_missing",
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
        logger.info(
            "llm_request_start",
            model=self.model,
            temperature=self.temperature,
            prompt_length=len(prompt),
            system_prompt_length=len(self.system_prompt),
            channel=channel_name,
        )

        if self.verbose:
            logger.debug(
                "llm_request_verbose",
                **self._build_prompt_log_payload(prompt),
            )

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
            redacted_events = [
                {
                    "action": evt.action,
                    "category": evt.category,
                    "anchor_present": bool(evt.anchor),
                    "link_count": len(evt.links),
                }
                for evt in (llm_response.events or [])
            ]

            logger.info(
                "llm_response_success",
                latency_ms=latency_ms,
                latency_s=round(latency_ms / 1000, 2),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                total_tokens=tokens_in + tokens_out,
                cost_usd=round(cost_usd, 6),
                is_event=llm_response.is_event,
                events_count=len(llm_response.events) if llm_response.events else 0,
                events=redacted_events[:5],
                events_redacted=True,
            )

            if self.verbose and content:
                logger.debug(
                    "llm_response_verbose",
                    **self._build_response_log_payload(content),
                )

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
                ts=datetime.now(tz=pytz.UTC),
            )

            return llm_response

        except OpenAIRateLimitError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "llm_rate_limit_error",
                latency_ms=latency_ms,
                error=str(e),
                model=self.model,
            )
            raise LLMAPIError(f"Rate limit exceeded: {e}")
        except APIError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "llm_api_error",
                latency_ms=latency_ms,
                error=str(e),
                model=self.model,
            )
            raise LLMAPIError(f"OpenAI API error: {e}")
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            if isinstance(e, ValidationError | LLMAPIError):
                logger.error(
                    "llm_validation_error",
                    latency_ms=latency_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            logger.error(
                "llm_unexpected_error",
                latency_ms=latency_ms,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise LLMAPIError(f"Unexpected error: {e}")

    @staticmethod
    def _is_verbose_allowed() -> bool:
        flag = os.getenv(VERBOSE_ENV_FLAG, "").strip().lower()
        return flag in {"1", "true", "yes"}

    @property
    def system_prompt_hash(self) -> str:
        """Expose system prompt hash for cache fingerprinting."""

        return self._system_prompt_hash

    def _build_prompt_log_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "system_prompt_checksum": self._system_prompt_hash,
            "system_prompt_char_length": len(self.system_prompt),
            "prompt_checksum": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_char_length": len(prompt),
            "prompt_redacted": True,
        }

    def _build_response_log_payload(self, content: str) -> dict[str, Any]:
        return {
            "response_checksum": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "response_char_length": len(content),
            "response_redacted": True,
        }

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

                    logger.warning(
                        "llm_retry_attempt",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type="rate_limit"
                        if is_rate_limit
                        else "timeout"
                        if is_timeout
                        else "validation",
                        error_msg=error_msg,
                        backoff_delay_s=delay,
                    )

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
