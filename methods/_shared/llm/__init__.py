"""Shared LLM utilities: client factories and answer normalizer."""
from .clients import (
    make_dashscope_client,
    make_azure_client,
    make_openrouter_client,
)
from .normalizer import normalize_answer, extract_letters

__all__ = [
    "make_dashscope_client",
    "make_azure_client",
    "make_openrouter_client",
    "normalize_answer",
    "extract_letters",
]
