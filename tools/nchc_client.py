from __future__ import annotations

import os
import time
from typing import Optional

import requests


class NCHCClient:
    """
    Client for NCHC's OpenAI-compatible Chat Completions API.

    Key-selection design
    --------------------
    This client intentionally uses exactly ONE NCHC API key for the
    lifetime of the Python process/work. It does NOT rotate across keys.

    Put multiple real keys in .env, for example:

        NCHC_API1='key_for_work_1'
        NCHC_API2='key_for_work_2'
        NCHC_API3='key_for_work_3'

    Then, in each independent PBS/job script, choose which key this work
    should use:

        export NCHC_API_KEY_NAME=NCHC_API2

    NCHC_API_KEY_NAME contains the *environment-variable name*, not the
    secret key itself. This keeps the actual key out of command-line
    arguments and normal logs.

    Environment variables
    ---------------------
    Required unless ``api_key`` is passed directly:
    - NCHC_API_KEY_NAME
        Name of the environment variable that stores the key.
        Example: NCHC_API_KEY_NAME=NCHC_API2
    - The selected variable itself, e.g. NCHC_API2='...'

    Optional:
    - NCHC_BASE_URL
        Default: https://portal.genai.nchc.org.tw/api/v1
    - NCHC_TIMEOUT_SECONDS
        Default: 180
    - NCHC_MAX_RETRIES
        Default: 3
    - NCHC_RETRY_BACKOFF_SECONDS
        Default: 5

    The class exposes the same ``complete_content_only`` method used by
    ``CerebrasRotatingClient`` so the rest of the retrosynthesis pipeline
    can switch providers without changing call sites.
    """

    def __init__(
        self,
        model: str,
        max_completion_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 0.95,
        api_key: str | None = None,
        api_key_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        self.model = str(model).strip()

        if not self.model:
            raise ValueError(
                "NCHC model cannot be empty."
            )

        # ----------------------------------------------------
        # Resolve exactly ONE API key for this work.
        # ----------------------------------------------------
        # ``api_key`` is kept as an explicit override for testing or
        # direct programmatic use. Normal jobs should use
        # NCHC_API_KEY_NAME instead.
        if api_key is not None:
            resolved_api_key = str(
                api_key
            ).strip()

            if not resolved_api_key:
                raise RuntimeError(
                    "The explicitly supplied NCHC api_key is empty."
                )

            resolved_key_name = "<direct-api-key>"

        else:
            resolved_key_name = str(
                api_key_name
                if api_key_name is not None
                else os.getenv(
                    "NCHC_API_KEY_NAME",
                    "",
                )
            ).strip()

            if not resolved_key_name:
                raise RuntimeError(
                    "NCHC_API_KEY_NAME is not set. "
                    "Set it in the current work/job, for example:\n"
                    "export NCHC_API_KEY_NAME=NCHC_API1"
                )

            resolved_api_key = str(
                os.getenv(
                    resolved_key_name,
                    "",
                )
            ).strip()

            if not resolved_api_key:
                raise RuntimeError(
                    f"NCHC_API_KEY_NAME points to {resolved_key_name!r}, "
                    "but that environment variable is missing or empty. "
                    "Add the real key to .env, for example:\n"
                    f"{resolved_key_name}='YOUR_NCHC_API_KEY'"
                )

        self.api_key = resolved_api_key
        self.api_key_name = resolved_key_name

        # ----------------------------------------------------
        # API configuration.
        # ----------------------------------------------------
        self.base_url = str(
            base_url
            if base_url is not None
            else os.getenv(
                "NCHC_BASE_URL",
                "https://portal.genai.nchc.org.tw/api/v1",
            )
        ).strip().rstrip("/")

        if not self.base_url:
            raise ValueError(
                "NCHC_BASE_URL cannot be empty."
            )

        self.max_completion_tokens = int(
            max_completion_tokens
        )

        self.temperature = float(
            temperature
        )

        # Retained for interface compatibility with CerebrasRotatingClient.
        # The current NCHC request payload below deliberately sends only the
        # parameters needed by this project and documented by NCHC.
        self.top_p = float(
            top_p
        )

        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv(
                "NCHC_TIMEOUT_SECONDS",
                "180",
            )
        )

        self.max_retries = int(
            max_retries
            if max_retries is not None
            else os.getenv(
                "NCHC_MAX_RETRIES",
                "3",
            )
        )

        self.retry_backoff_seconds = float(
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else os.getenv(
                "NCHC_RETRY_BACKOFF_SECONDS",
                "5",
            )
        )

        if self.max_completion_tokens < 1:
            raise ValueError(
                "max_completion_tokens must be at least 1."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "NCHC_TIMEOUT_SECONDS must be positive."
            )

        if self.max_retries < 1:
            raise ValueError(
                "NCHC_MAX_RETRIES must be at least 1."
            )

        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "NCHC_RETRY_BACKOFF_SECONDS cannot be negative."
            )

        print(
            "[NCHCClient] "
            f"model={self.model} "
            f"key_name={self.api_key_name} "
            "single-key mode"
        )

    @property
    def chat_completions_url(self) -> str:
        return (
            f"{self.base_url}/chat/completions"
        )

    @staticmethod
    def _response_error_text(
        response: requests.Response,
    ) -> str:
        """
        Extract a useful error message without exposing request headers or
        the API key.
        """
        try:
            data = response.json()

        except Exception:
            return response.text[:4000]

        if isinstance(
            data,
            dict,
        ):
            error = data.get(
                "error"
            )

            if isinstance(
                error,
                dict,
            ):
                message = error.get(
                    "message"
                )

                if message:
                    return str(
                        message
                    )[:4000]

            if error:
                return str(
                    error
                )[:4000]

            message = data.get(
                "message"
            )

            if message:
                return str(
                    message
                )[:4000]

        return str(
            data
        )[:4000]

    @staticmethod
    def _is_retryable_status(
        status_code: int,
    ) -> bool:
        """
        Retry transient request/server errors using the SAME selected key.

        There is intentionally no key rotation here.
        """
        return status_code in {
            408,
            409,
            425,
            429,
            500,
            502,
            503,
            504,
        }

    def _retry_sleep_seconds(
        self,
        attempt: int,
        response: requests.Response | None = None,
    ) -> float:
        """
        Prefer Retry-After when the server supplies a numeric value;
        otherwise use linear backoff.
        """
        if response is not None:
            retry_after = response.headers.get(
                "Retry-After"
            )

            if retry_after:
                try:
                    value = float(
                        retry_after
                    )

                    if value >= 0:
                        return value

                except ValueError:
                    pass

        return (
            self.retry_backoff_seconds
            * attempt
        )

    @staticmethod
    def _normalize_usage(
        data: dict,
    ) -> dict:
        """
        Normalize the OpenAI-compatible ``usage`` object returned by NCHC.

        NCHC documents prompt/completion token usage for Chat Completions.
        ``total_tokens`` is used directly when provided. If it is absent but
        both prompt and completion counts are available, it is computed as
        their sum. Missing usage is returned as an empty dictionary rather
        than estimated locally.
        """
        if not isinstance(data, dict):
            return {}

        raw_usage = data.get("usage")
        if not isinstance(raw_usage, dict):
            return {}

        normalized = {}

        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            value = raw_usage.get(key)
            if value is None:
                continue

            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                continue

        if (
            "total_tokens" not in normalized
            and "prompt_tokens" in normalized
            and "completion_tokens" in normalized
        ):
            normalized["total_tokens"] = (
                normalized["prompt_tokens"]
                + normalized["completion_tokens"]
            )

        # Preserve the provider response for debugging/auditing.
        normalized["raw_usage"] = raw_usage

        return normalized

    def complete_content_with_usage(
        self,
        prompt: str,
        system_prompt: str = "",
        response_format=None,
        metadata: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """
        Call NCHC Chat Completions and return message content only.

        Important behavior:
        - Every retry uses the SAME API key selected by
          NCHC_API_KEY_NAME.
        - This method never rotates to another NCHC_API* variable.
        - Parallel calls inside one MoA work therefore share the same key,
          which is intentional for experiment-level key control.
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

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": (
                self.max_completion_tokens
            ),
            "temperature": (
                self.temperature
            ),
        }

        # Preserve compatibility with callers that may eventually request
        # a structured response format. Omit it when None.
        if response_format is not None:
            payload[
                "response_format"
            ] = response_format

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        sample_index = ""
        sample_id = ""
        moa_stage = ""

        if metadata:
            sample_index = metadata.get(
                "sample_index",
                "",
            )

            sample_id = metadata.get(
                "id",
                "",
            )

            moa_stage = metadata.get(
                "moa_stage",
                "",
            )

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            print(
                "[NCHCClient] "
                f"model={self.model} "
                f"key_name={self.api_key_name} "
                f"attempt={attempt}/{self.max_retries} "
                f"sample_index={sample_index} "
                f"id={sample_id} "
                f"stage={moa_stage}"
            )

            try:
                response = requests.post(
                    self.chat_completions_url,
                    headers=headers,
                    json=payload,
                    timeout=(
                        self.timeout_seconds
                    ),
                )

            except requests.RequestException as error:
                last_error = error

                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "NCHC request failed after "
                        f"{attempt} attempt(s) using "
                        f"key_name={self.api_key_name}: "
                        f"{error}"
                    ) from error

                sleep_seconds = (
                    self._retry_sleep_seconds(
                        attempt=attempt,
                    )
                )

                print(
                    "[NCHCClient] "
                    "Network error; retrying with the SAME key "
                    f"{self.api_key_name} in "
                    f"{sleep_seconds:.1f}s: {error}"
                )

                time.sleep(
                    sleep_seconds
                )

                continue

            if not response.ok:
                error_text = (
                    self._response_error_text(
                        response
                    )
                )

                error = RuntimeError(
                    "NCHC HTTP "
                    f"{response.status_code} "
                    f"using key_name={self.api_key_name}: "
                    f"{error_text}"
                )

                last_error = error

                if (
                    self._is_retryable_status(
                        response.status_code
                    )
                    and attempt < self.max_retries
                ):
                    sleep_seconds = (
                        self._retry_sleep_seconds(
                            attempt=attempt,
                            response=response,
                        )
                    )

                    print(
                        "[NCHCClient] "
                        "Retryable HTTP error; retrying with "
                        f"the SAME key {self.api_key_name} in "
                        f"{sleep_seconds:.1f}s: {error}"
                    )

                    time.sleep(
                        sleep_seconds
                    )

                    continue

                raise error

            try:
                data = response.json()

            except ValueError as error:
                raise RuntimeError(
                    "NCHC returned a non-JSON response: "
                    f"{response.text[:4000]}"
                ) from error

            try:
                content = (
                    data[
                        "choices"
                    ][0][
                        "message"
                    ][
                        "content"
                    ]
                )

            except (
                KeyError,
                IndexError,
                TypeError,
            ) as error:
                raise RuntimeError(
                    "Unexpected NCHC Chat Completions "
                    "response structure: "
                    f"{str(data)[:4000]}"
                ) from error

            if content is None:
                raise RuntimeError(
                    "NCHC returned empty message content."
                )

            usage = self._normalize_usage(
                data
            )

            if usage:
                print(
                    "[NCHC_USAGE] "
                    f"prompt_tokens={usage.get('prompt_tokens', '')} "
                    f"completion_tokens={usage.get('completion_tokens', '')} "
                    f"total_tokens={usage.get('total_tokens', '')}"
                )
            else:
                print(
                    "[NCHC_USAGE] "
                    "No usage object was returned by this response."
                )

            return (
                str(content),
                usage,
            )

        raise RuntimeError(
            "NCHC request exhausted all retries using "
            f"key_name={self.api_key_name}. "
            f"Last error: {last_error}"
        )

    def complete_content_only(
        self,
        prompt: str,
        system_prompt: str = "",
        response_format=None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Backward-compatible content-only interface used by existing MoA code.
        """
        content, _usage = self.complete_content_with_usage(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            metadata=metadata,
        )
        return content
