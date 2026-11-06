from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from prompt import build_user_prompt, get_system_prompt
from tools.cerebras_client import CerebrasRotatingClient
from tools.nchc_client import NCHCClient


ENV_FILE = os.getenv("ASKCOS_ENV_FILE", ".env")
load_dotenv(dotenv_path=ENV_FILE)

from tools.token_counter import count_chat_characters, count_chat_tokens

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-31b")
LLM_MAX_COMPLETION_TOKENS = int(
    os.getenv("LLM_MAX_COMPLETION_TOKENS", "8192")
)

SUPPORTED_LLM_PROVIDERS = {"cerebras", "nchc"}


def validate_output_top_k(output_top_k: int) -> int:
    output_top_k = int(output_top_k)
    if output_top_k < 1:
        raise ValueError("output_top_k must be at least 1.")
    return output_top_k


def validate_llm_provider(llm_provider: str) -> str:
    provider = str(llm_provider).strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported llm_provider={provider!r}. "
            f"Supported providers: {sorted(SUPPORTED_LLM_PROVIDERS)}"
        )
    return provider


@lru_cache(maxsize=None)
def get_llm_client(llm_provider: str, llm_model: str):
    """Return a cached provider-specific client.

    Provider and model are separate cache keys. No provider is inferred from
    the model name. This also means the same model string can be used once
    through Cerebras and once through NCHC in the same process.
    """
    provider = validate_llm_provider(llm_provider)
    model = str(llm_model).strip()
    if not model:
        raise ValueError("llm_model cannot be empty.")

    print(f"[LLM_CLIENT] provider={provider} model={model}")

    if provider == "cerebras":
        return CerebrasRotatingClient(
            model=model,
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
            temperature=0.0,
            top_p=0.95,
            key_cooldown_seconds=60,
        )

    if provider == "nchc":
        return NCHCClient(
            model=model,
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
            temperature=0.0,
            top_p=0.95,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def estimate_total_prompt_tokens(
    target: str,
    candidates: list,
    output_top_k: int = 10,
    include_reasoning: bool = True,
) -> int:
    output_top_k = validate_output_top_k(output_top_k)
    user_prompt = build_user_prompt(
        target=target,
        candidates=candidates,
        output_top_k=output_top_k,
        include_reasoning=include_reasoning,
    )
    system_prompt = get_system_prompt(
        candidates=candidates,
        output_top_k=output_top_k,
    )
    return count_chat_tokens(system_prompt, user_prompt)


def estimate_total_prompt_characters(
    target: str,
    candidates: list,
    output_top_k: int = 10,
    include_reasoning: bool = True,
) -> int:
    output_top_k = validate_output_top_k(output_top_k)
    user_prompt = build_user_prompt(
        target=target,
        candidates=candidates,
        output_top_k=output_top_k,
        include_reasoning=include_reasoning,
    )
    system_prompt = get_system_prompt(
        candidates=candidates,
        output_top_k=output_top_k,
    )
    return count_chat_characters(system_prompt, user_prompt)


def select_top_candidates(
    target: str,
    candidates: list,
    output_top_k: int = 10,
    metadata=None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    include_reasoning: bool = True,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """Ask one explicitly selected provider/model pair to rank candidates.

    When ``return_usage`` is True, providers that expose API usage return
    ``(content, usage_dict)``. Providers without usage support return an
    empty usage dictionary; no local token estimate is substituted.
    """
    output_top_k = validate_output_top_k(output_top_k)

    if llm_provider is None:
        raise ValueError(
            "llm_provider must be explicitly specified as 'cerebras' or 'nchc'."
        )
    selected_provider = validate_llm_provider(llm_provider)

    selected_model = (
        str(llm_model).strip() if llm_model is not None else DEFAULT_LLM_MODEL
    )
    if not selected_model:
        raise ValueError("llm_model cannot be empty.")

    client = get_llm_client(selected_provider, selected_model)

    user_prompt = build_user_prompt(
        target=target,
        candidates=candidates,
        output_top_k=output_top_k,
        include_reasoning=include_reasoning,
    )
    system_prompt = get_system_prompt(
        candidates=candidates,
        output_top_k=output_top_k,
    )

    print(f"\n[DEBUG] LLM provider: {selected_provider}")
    print(f"[DEBUG] LLM model: {selected_model}")
    print(f"[DEBUG] Candidate rows passed to LLM: {len(candidates)}")
    print(f"[DEBUG] Requested output top-k: {output_top_k}")
    print(f"[DEBUG] Include reasoning: {include_reasoning}")

    if any(
        str(candidate.get("model", "")).strip().lower() == "retroknn"
        for candidate in candidates
    ):
        print("\n[DEBUG] Prompt mode:\nRetroKNN prompt")
    else:
        print("\n[DEBUG] Prompt mode:\nBase ASKCOS-only prompt")

    print("\n[DEBUG] Prompt preview:")
    print(user_prompt[:500])

    request_metadata = dict(metadata or {})
    request_metadata["llm_provider"] = selected_provider
    request_metadata["llm_model"] = selected_model
    request_metadata["output_top_k"] = output_top_k
    request_metadata["include_reasoning"] = bool(include_reasoning)

    if (
        return_usage
        and hasattr(
            client,
            "complete_content_with_usage",
        )
    ):
        return client.complete_content_with_usage(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=None,
            metadata=request_metadata,
        )

    content = client.complete_content_only(
        prompt=user_prompt,
        system_prompt=system_prompt,
        response_format=None,
        metadata=request_metadata,
    )

    if return_usage:
        return (
            content,
            {},
        )

    return content


def select_top10_candidates(
    target: str,
    candidates: list,
    metadata=None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    include_reasoning: bool = True,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """Backward-compatible Top-10 wrapper."""
    return select_top_candidates(
        target=target,
        candidates=candidates,
        output_top_k=10,
        metadata=metadata,
        llm_provider=llm_provider,
        llm_model=llm_model,
        include_reasoning=include_reasoning,
        return_usage=return_usage,
    )
