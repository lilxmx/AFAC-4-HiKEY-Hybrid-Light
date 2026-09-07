"""Shared retrieval API for QA pipelines.

Public surface
--------------
- ``Retriever``           : duck-typed protocol QA agents depend on
- ``RetrieverBase``       : optional ABC with overridable hooks
- ``EvidencePack``        : dict shape returned by ``retrieve`` / ``merge``
- ``EvidenceItem``        : single evidence record inside a pack
- ``RetrievalFilters``    : per-call filtering knobs
- ``RetrievalMode``       : "global" | "option" | "field" | "counter" | "vlm" | "custom"
- ``HiKEYRetriever``      : default implementation backed by m04 cache
- ``build_retriever``     : factory ``build_retriever('hikey', **kwargs)``
- ``register_retriever``  : plug new implementations in by name

Importing this package is cheap; heavy dependencies (PyMuPDF, m04 parser)
are only loaded when an index is actually accessed.
"""
from .api import (
    EvidenceItem,
    EvidencePack,
    RetrievalFilters,
    RetrievalMode,
    Retriever,
    RetrieverBase,
    SourceTag,
)
from .factory import (
    available_retrievers,
    build_retriever,
    register_retriever,
)
from .hikey_retriever import HiKEYRetriever

__all__ = [
    "EvidenceItem",
    "EvidencePack",
    "RetrievalFilters",
    "RetrievalMode",
    "Retriever",
    "RetrieverBase",
    "SourceTag",
    "HiKEYRetriever",
    "available_retrievers",
    "build_retriever",
    "register_retriever",
]
