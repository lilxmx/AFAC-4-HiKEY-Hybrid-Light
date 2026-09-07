"""
LLM client factories.

All methods can use the same client constructors so we don't repeat
api_key / base_url plumbing across configs.
"""
import os
from typing import Optional

from openai import OpenAI


# ----- Dashscope (Qwen) -----
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def make_dashscope_client(
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL,
) -> OpenAI:
    """OpenAI-compatible client for Aliyun Dashscope (Qwen-plus / Qwen-max)."""
    return OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url=base_url,
    )


# ----- Azure OpenAI (GPT-4.1 etc.) -----
def make_azure_client(api_key: str, base_url: str) -> OpenAI:
    """
    Azure OpenAI via OpenAI-compatible endpoint.

    NOTE: We deliberately use the plain OpenAI client (not AzureOpenAI) because
    Azure exposes an OpenAI-compatible /v1/ endpoint that doesn't require the
    Azure-specific signing flow.
    """
    return OpenAI(api_key=api_key, base_url=base_url)


# ----- OpenRouter (e.g. for embeddings) -----
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_openrouter_client(
    api_key: str,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> OpenAI:
    """OpenAI-compatible client for OpenRouter."""
    return OpenAI(api_key=api_key, base_url=base_url)
