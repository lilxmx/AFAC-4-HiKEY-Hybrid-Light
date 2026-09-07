"""Shared document parsers package."""
from .doc_resolver import resolve_doc_path, get_all_doc_paths, get_file_type
from .txt_parser import parse_txt
from .html_parser import parse_html
from .pdf_parser import parse_pdf, parse_pdf_with_tables
from .full_text import (
    parse_document_blocks,
    parse_document_fulltext,
    clear_cache,
)

__all__ = [
    "resolve_doc_path",
    "get_all_doc_paths",
    "get_file_type",
    "parse_txt",
    "parse_html",
    "parse_pdf",
    "parse_pdf_with_tables",
    "parse_document_blocks",
    "parse_document_fulltext",
    "clear_cache",
]
