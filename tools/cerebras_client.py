from __future__ import annotations

import os
import re
import threading
import time
from typing import List, Optional, Tuple

from cerebras.cloud.sdk import Cerebras, RateLimitError


class TokenPerMinuteLimitError(RuntimeError):
    """
    Raised when the Cerebras token-per-minute limit is exceeded.

    The caller should record the current sample as an error and continue
    to the next molecule instead of retrying the same molecule indefinitely.
    """

    pass


class CerebrasRotatingClient:
    """
    Cerebras client with multiple API key rotation.

    Features:
    - Auto-detect CEREBRAS_API, CEREBRAS_API1, CEREBRAS_API2, ...
    - Round-robin key selection
    - General RateLimitError / queue_exceeded / high traffic:
      mark the key as cooling down and rotate to another key
    - Tokens-per-minute limit:
      raise TokenPerMinuteLimitError so the caller can skip this sample
    - disabled_organization:
      permanently disable that key and rotate to the next key
    - Other non-retryable errors:
      raise immediately
    - If all keys are disabled:
      raise an error and let the caller move to the next molecule
    """

    def __init__(
        self,
        api_env_names: Optional[List[str]] = None,
        model: str = "qwen-3-235b-a22b-instruct-2507",
        max_completion_tokens: int = 2000,
        temperature: float = 0.0,
        top_p: float = 0.95,
        key_cooldown_seconds: int = 90,
    ):
        key_records = self._load_api_keys(api_env_names)

        if not key_records:
            raise RuntimeError(
                "No Cerebras API keys found. Please set CEREBRAS_API "
                "or CEREBRAS_API1, CEREBRAS_API2, ..."
            )

        self._keys = [
            {
                "value": key_value,
                "source": source,
                "available_at": 0.0,
                "disabled": False,
                "disabled_reason": "",
            }
            for key_value, source in key_records
        ]

        self._idx = 0
        self._lock = threading.Lock()

        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.key_cooldown_seconds = key_cooldown_seconds

        print(
            f"[CerebrasRotatingClient] "
            f"Loaded {len(self._keys)} API keys."
        )

    def _auto_discover_env_names(self) -> List[str]:
        """
        Detect API key environment variable names.

        Supported examples:
        - CEREBRAS_API
        - CEREBRAS_API1
        - CEREBRAS_API2
        - ...
        """
        names = [
            name
            for name in os.environ
            if name.upper().startswith("CEREBRAS_API")
        ]

        if names:
            return sorted(
                names,
                key=self._env_name_sort_key,
            )

        return [
            f"CEREBRAS_API{i}"
            for i in range(1, 101)
        ]

    @staticmethod
    def _env_name_sort_key(name: str):
        """
        Sort CEREBRAS_API first, followed by numbered variables.
        """
        upper = name.upper()

        if upper == "CEREBRAS_API":
            return 0, 0

        match = re.match(
            r"^CEREBRAS_API(\d+)$",
            upper,
        )

        if match:
            return 1, int(match.group(1))

        return 2, upper

    @staticmethod
    def _split_env_value(value: str) -> List[str]:
        """
        Allow one environment variable to contain multiple keys separated by:
        - comma
        - semicolon
        - newline
        """
        value = value.strip()

        if not value:
            return []

        if any(
            separator in value
            for separator in [",", ";", "\n"]
        ):
            return [
                part.strip()
                for part in re.split(
                    r"[,\n;]+",
                    value,
                )
                if part.strip()
            ]

        return [value]

    def _load_api_keys(
        self,
        api_env_names: Optional[List[str]],
    ) -> List[Tuple[str, str]]:
        """
        Load and deduplicate API keys from environment variables.

        Returns:
            List of:
                (api_key_value, source_label)
        """
        names = (
            api_env_names
            or self._auto_discover_env_names()
        )

        keys = []
        seen_values = set()

        for env_name in names:
            raw = os.environ.get(env_name)

            if not raw:
                continue

            for idx, key in enumerate(
                self._split_env_value(raw)
            ):
                if not key:
                    continue

                if key in seen_values:
                    continue

                if idx == 0:
                    label = env_name
                else:
                    label = f"{env_name}[{idx}]"

                keys.append(
                    (
                        key,
                        label,
                    )
                )

                seen_values.add(key)

        return keys

    def _active_key_count(self) -> int:
        """
        Return the number of keys that have not been permanently disabled.
        """
        with self._lock:
            return sum(
                1
                for info in self._keys
                if not info["disabled"]
            )

    def _active_key_sources(self) -> List[str]:
        """
        Return source names of active keys.
        """
        with self._lock:
            return [
                str(info["source"])
                for info in self._keys
                if not info["disabled"]
            ]

    def _next_key(self) -> Tuple[int, str, str]:
        """
        Return the next available API key using round-robin selection.

        If all active keys are cooling down, wait until the earliest key
        becomes available.

        If every key is permanently disabled, raise immediately.
        """
        while True:
            with self._lock:
                now = time.time()
                num_keys = len(self._keys)

                active_keys = [
                    info
                    for info in self._keys
                    if not info["disabled"]
                ]

                if not active_keys:
                    disabled_sources = [
                        (
                            f"{info['source']} "
                            f"({info['disabled_reason']})"
                        )
                        for info in self._keys
                    ]

                    raise RuntimeError(
                        "All Cerebras API keys are disabled or unusable. "
                        f"Disabled keys: {disabled_sources}"
                    )

                for _ in range(num_keys):
                    idx = self._idx
                    self._idx = (
                        self._idx + 1
                    ) % num_keys

                    info = self._keys[idx]

                    if info["disabled"]:
                        continue

                    if info["available_at"] <= now:
                        return (
                            idx,
                            info["value"],
                            info["source"],
                        )

                next_ready = min(
                    info["available_at"]
                    for info in self._keys
                    if not info["disabled"]
                )

                wait_seconds = max(
                    0.0,
                    next_ready - now,
                )

            print(
                "[CerebrasRotatingClient] "
                "All active keys are cooling down. "
                f"Sleeping {wait_seconds:.0f}s..."
            )

            time.sleep(
                wait_seconds + 1.0
            )

    def _mark_key_on_ratelimit(
        self,
        key_idx: int,
        source: str,
    ):
        """
        Temporarily mark one key as unavailable.
        """
        available_at = (
            time.time()
            + self.key_cooldown_seconds
        )

        with self._lock:
            self._keys[key_idx][
                "available_at"
            ] = available_at

        print(
            "[CerebrasRotatingClient] "
            f"Key {source} cooldown "
            f"{self.key_cooldown_seconds}s "
            "due to rate limit/high traffic."
        )

    def _mark_key_disabled(
        self,
        key_idx: int,
        source: str,
        reason: str,
    ):
        """
        Permanently disable one API key for the current program run.
        """
        with self._lock:
            self._keys[key_idx]["disabled"] = True

            self._keys[key_idx][
                "disabled_reason"
            ] = reason

            active_left = sum(
                1
                for info in self._keys
                if not info["disabled"]
            )

        print(
            "[CerebrasRotatingClient] "
            f"Key {source} permanently disabled "
            f"for this run. "
            f"reason={reason}. "
            f"active_keys_left={active_left}"
        )

    @staticmethod
    def _normalize_error_message(
        error: Exception,
    ) -> str:
        """
        Normalize an exception message for rule matching.
        """
        return " ".join(
            str(error)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )

    def _is_token_per_minute_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Detect Cerebras tokens-per-minute errors.

        This check must run before the general 429 / rate-limit check.

        These errors should not cause indefinite key rotation because the
        request itself may be too large for the current minute-level quota.
        The current sample should be skipped.
        """
        msg = self._normalize_error_message(
            error
        )

        compact_msg = re.sub(
            r"[^a-z0-9]+",
            "",
            str(error).lower(),
        )

        explicit_phrases = [
            "tokens per minute",
            "token per minute",
            "token rate limit",
            "token throughput limit",
            "minute token limit",
            "per minute token",
            "tpm limit",
            "tpm exceeded",
            "tpm quota",
        ]

        if any(
            phrase in msg
            for phrase in explicit_phrases
        ):
            return True

        if any(
            token in compact_msg
            for token in [
                "tokensperminute",
                "tokenperminute",
                "tpmexceeded",
                "tpmlimit",
                "tpmquota",
            ]
        ):
            return True

        has_token_word = (
            "token" in msg
            or "tokens" in msg
        )

        has_minute_word = (
            "minute" in msg
            or "per min" in msg
        )

        has_limit_word = any(
            word in msg
            for word in [
                "limit",
                "exceed",
                "quota",
                "maximum",
                "too many",
            ]
        )

        return (
            has_token_word
            and has_minute_word
            and has_limit_word
        )

    def _is_high_traffic_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Detect retryable request-rate or server-traffic errors.

        Token-per-minute errors are explicitly excluded because those must be
        handled by TokenPerMinuteLimitError.
        """
        if self._is_token_per_minute_error(
            error
        ):
            return False

        msg = str(error).lower()

        return (
            isinstance(error, RateLimitError)
            or "queue_exceeded" in msg
            or "queue exceeded" in msg
            or "too_many_requests" in msg
            or "too many requests" in msg
            or "high traffic" in msg
            or "server is busy" in msg
            or "temporarily overloaded" in msg
            or "rate limit" in msg
            or "429" in msg
        )

    def _is_disabled_organization_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Detect API keys whose organization has been disabled.
        """
        msg = str(error).lower()

        return (
            "disabled_organization" in msg
            or (
                "organization this api key belongs"
                in msg
                and "disabled" in msg
            )
            or (
                "error code: 401" in msg
                and "organization" in msg
                and "disabled" in msg
            )
            or (
                "status code: 401" in msg
                and "organization" in msg
                and "disabled" in msg
            )
        )

    def complete_content_only(
        self,
        prompt: str,
        system_prompt: str = "",
        response_format=None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Call the Cerebras model and return response message content only.

        Error behavior:
        - Token-per-minute:
          raise TokenPerMinuteLimitError immediately
        - disabled_organization:
          disable key and try another key
        - general 429/high traffic:
          cooldown key and try another key
        - any other error:
          raise immediately
        """
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        attempt = 0
        disabled_org_sources_this_call = set()

        while True:
            attempt += 1

            (
                key_idx,
                api_key,
                source,
            ) = self._next_key()

            print(
                "[CerebrasRotatingClient] "
                f"Calling model with key={source} "
                f"global_attempt={attempt}"
            )

            if metadata:
                sample_index = metadata.get(
                    "sample_index",
                    "",
                )

                sample_id = metadata.get(
                    "id",
                    "",
                )

                print(
                    "[CerebrasRotatingClient] "
                    f"sample_index={sample_index} "
                    f"id={sample_id}"
                )

            try:
                client = Cerebras(
                    api_key=api_key
                )

                response = (
                    client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=(
                            self.max_completion_tokens
                        ),
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stream=False,
                        response_format=response_format,
                    )
                )

                content = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                if content is None:
                    raise RuntimeError(
                        "Cerebras returned empty content: "
                        f"{response}"
                    )

                return content

            except Exception as e:
                # This check must come before the general rate-limit check.
                if self._is_token_per_minute_error(
                    e
                ):
                    print(
                        "[LLM_TOKEN_PER_MINUTE_LIMIT] "
                        f"key={source} "
                        f"attempt={attempt} "
                        f"error={e}"
                    )

                    print(
                        "[LLM_TOKEN_PER_MINUTE_LIMIT] "
                        "The current molecule will not be retried. "
                        "The caller should record an error and "
                        "continue to the next molecule."
                    )

                    raise TokenPerMinuteLimitError(
                        "Cerebras token-per-minute limit "
                        f"exceeded for key={source}: {e}"
                    ) from e

                if self._is_disabled_organization_error(
                    e
                ):
                    disabled_org_sources_this_call.add(
                        source
                    )

                    self._mark_key_disabled(
                        key_idx=key_idx,
                        source=source,
                        reason="disabled_organization",
                    )

                    print(
                        "[LLM_DISABLED_ORG_RETRY] "
                        f"key={source} "
                        f"attempt={attempt} "
                        f"error={e}"
                    )

                    print(
                        "[LLM_DISABLED_ORG_RETRY] "
                        "Will try the next API key "
                        "for the same molecule."
                    )

                    if self._active_key_count() == 0:
                        raise RuntimeError(
                            "All Cerebras API keys failed "
                            "with disabled_organization. "
                            "failed_keys="
                            f"{sorted(disabled_org_sources_this_call)}"
                        ) from e

                    continue

                if self._is_high_traffic_error(
                    e
                ):
                    self._mark_key_on_ratelimit(
                        key_idx,
                        source,
                    )

                    print(
                        "[LLM_RATE_LIMIT_RETRY] "
                        f"key={source} "
                        f"attempt={attempt} "
                        f"error={e}"
                    )

                    print(
                        "[LLM_RATE_LIMIT_RETRY] "
                        "Will try another API key "
                        "for the same molecule."
                    )

                    continue

                print(
                    "[LLM_NON_RETRYABLE_ERROR] "
                    f"key={source} "
                    f"attempt={attempt} "
                    f"error={e}"
                )

                raise
