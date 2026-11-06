import os
from functools import lru_cache

TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507")

@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_chat_tokens(system_prompt: str, user_prompt: str) -> int:
    tokenizer = get_tokenizer()
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    rendered_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids = tokenizer(
        rendered_text,
        add_special_tokens=False,
    )["input_ids"]

    return len(token_ids)


def count_chat_characters(system_prompt: str, user_prompt: str) -> int:
    tokenizer = get_tokenizer()
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    rendered_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return len(rendered_text)
