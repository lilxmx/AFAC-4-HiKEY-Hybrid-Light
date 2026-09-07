"""Retriever factory + registry.

The factory keeps QA pipelines decoupled from concrete retriever
classes. New retrievers can be plugged in by calling
``register_retriever('my_name', MyClass)`` from anywhere — typically
from the module that defines them, or from a config bootstrap step.

Built-in registrations are done lazily inside ``build_retriever`` so
that importing this module does not trigger heavy dependencies.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .api import Retriever

_REGISTRY: Dict[str, Callable[..., Retriever]] = {}


def register_retriever(name: str, factory: Callable[..., Retriever]) -> None:
    """Register a new retriever implementation under ``name``.

    ``factory`` must be a callable that accepts keyword arguments and
    returns an object satisfying the :class:`Retriever` protocol.
    Re-registering an existing name overwrites the previous entry —
    this is intentional so teammates can shadow defaults during local
    experimentation.
    """
    _REGISTRY[name] = factory


def available_retrievers() -> Dict[str, Callable[..., Retriever]]:
    """Return a snapshot of registered retrievers (mostly for debugging)."""
    return dict(_REGISTRY)


def _ensure_builtins_registered() -> None:
    """Lazy registration of bundled retrievers."""
    if "hikey" not in _REGISTRY:
        from .hikey_retriever import HiKEYRetriever
        _REGISTRY["hikey"] = HiKEYRetriever


def build_retriever(name: str, **kwargs: Any) -> Retriever:
    """Instantiate a registered retriever by name.

    Example
    -------
    >>> from methods._shared.retrieval import build_retriever
    >>> r = build_retriever('hikey', index_dir='/path/to/_HiKEY_cache')
    >>> r.warmup(domains=['financial_reports'])
    >>> pack = r.retrieve('宁德时代2024 现金分红', mode='global', topk=10)
    """
    _ensure_builtins_registered()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown retriever: {name!r}. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "register_retriever",
    "available_retrievers",
    "build_retriever",
]
