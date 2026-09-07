"""
Unified document parsing entry point with shared on-disk cache.

Two main APIs:

- parse_document_blocks(doc_id, domain) -> List[Dict]
    Returns structured blocks (one per page/section). Used by chunk-based
    methods (m01, m02). Cached per (doc_id, domain) as individual page files.

- parse_document_fulltext(doc_id, domain) -> str
    Returns the full document concatenated as a single string. Used by
    full-text methods (m03). Cached per (doc_id, domain) as fulltext.txt.

Cache layout (human-readable):
    _cache/{domain}/{doc_id}/
        fulltext.txt          - concatenated full text
        page_001.txt          - block for page 1
        page_002.txt          - block for page 2
        ...
        metadata.json         - block metadata (section_title per page)
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..config_base import SHARED_CACHE_ROOT
from .doc_resolver import resolve_doc_path
from .html_parser import parse_html
from .pdf_parser import parse_pdf, parse_pdf_with_tables
from .txt_parser import parse_txt


def _sanitize_dirname(name: str) -> str:
    """
    Sanitize a doc_id for use as a directory name.
    Remove or replace characters that are problematic for filesystems.
    Keep Chinese characters and most content intact.
    """
    # Replace filesystem-unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_{2,}', '_', name)
    # Trim to reasonable length (keep first 200 chars)
    if len(name) > 200:
        name = name[:200]
    return name.strip('_. ')


def _get_doc_cache_dir(doc_id: str, domain: str) -> Path:
    """Get the cache directory for a specific document."""
    safe_name = _sanitize_dirname(doc_id)
    return SHARED_CACHE_ROOT / domain / safe_name


def parse_document_blocks(
    doc_id: str,
    domain: str,
    use_tables: Optional[bool] = None,
    use_cache: bool = True,
) -> List[Dict]:
    """
    Parse a document into structured blocks.

    Args:
        doc_id: Document identifier.
        domain: Domain name.
        use_tables: If True, use table-extraction PDF parser (forced).
                    If None, default to True for financial_reports.
        use_cache: Read/write cache.

    Returns:
        List of {text, page_num, section_title}. Empty list on failure.
    """
    cache_dir = _get_doc_cache_dir(doc_id, domain)
    metadata_path = cache_dir / "metadata.json"

    # Try reading from cache
    if use_cache and metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            blocks = []
            for entry in metadata["blocks"]:
                page_file = cache_dir / entry["filename"]
                if page_file.exists():
                    text = page_file.read_text(encoding="utf-8")
                    blocks.append({
                        "text": text,
                        "page_num": entry["page_num"],
                        "section_title": entry.get("section_title", ""),
                    })
            if blocks:
                return blocks
        except Exception:
            pass  # fall through and re-parse

    # Parse the document
    file_path, file_type = resolve_doc_path(doc_id, domain)
    if file_path is None or not file_path.exists():
        print(f"[WARNING] Cannot resolve doc_id={doc_id} in domain={domain}")
        return []

    if use_tables is None:
        use_tables = (domain == "financial_reports")

    blocks: List[Dict] = []
    try:
        if file_type == "txt":
            blocks = parse_txt(file_path)
        elif file_type == "html":
            blocks = parse_html(file_path)
        elif file_type == "pdf":
            if use_tables:
                blocks = parse_pdf_with_tables(file_path)
            else:
                blocks = parse_pdf(file_path)
        else:
            print(f"[WARNING] Unknown file type for {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path}: {e}")
        blocks = []

    # Write to cache
    if use_cache and blocks:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            metadata_entries = []
            for i, block in enumerate(blocks):
                page_num = block.get("page_num", i + 1)
                # Use sequential index (i+1) for filename to avoid overwrites
                # when multiple blocks share the same page_num (e.g. txt/html)
                seq = i + 1
                filename = f"page_{seq:03d}.txt"
                page_path = cache_dir / filename
                page_path.write_text(block["text"], encoding="utf-8")
                metadata_entries.append({
                    "filename": filename,
                    "page_num": page_num,
                    "section_title": block.get("section_title", ""),
                })
            metadata = {
                "doc_id": doc_id,
                "domain": domain,
                "num_blocks": len(blocks),
                "blocks": metadata_entries,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to write blocks cache for {doc_id}: {e}")

    return blocks


def parse_document_fulltext(
    doc_id: str,
    domain: str,
    use_cache: bool = True,
) -> str:
    """
    Parse a document and return its full text content (single string).

    Tries the dedicated full-text cache first; if absent, derives full text
    from the blocks parser (which itself is cached) and stores the result.
    """
    cache_dir = _get_doc_cache_dir(doc_id, domain)
    fulltext_path = cache_dir / "fulltext.txt"

    if use_cache and fulltext_path.exists():
        try:
            return fulltext_path.read_text(encoding="utf-8")
        except Exception:
            pass

    blocks = parse_document_blocks(doc_id, domain, use_cache=use_cache)
    if not blocks:
        return ""

    # Concatenate blocks. Page-level text already includes headings.
    text = "\n\n".join(b["text"] for b in blocks if b.get("text"))

    if use_cache and text:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            fulltext_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"[WARNING] Failed to write fulltext cache for {doc_id}: {e}")

    return text


def clear_cache(domain: Optional[str] = None) -> None:
    """Delete cached parses. If domain specified, only clear that domain."""
    import shutil

    if domain:
        domain_dir = SHARED_CACHE_ROOT / domain
        if domain_dir.exists():
            shutil.rmtree(domain_dir)
    else:
        # Clear all domain directories under cache root
        if SHARED_CACHE_ROOT.exists():
            for child in SHARED_CACHE_ROOT.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
