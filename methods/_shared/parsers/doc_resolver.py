"""
Document ID -> file path resolver (shared across all methods).

Maps a (doc_id, domain) pair to an actual file on disk under RAW_BASE.
Centralizes the file-naming conventions for each domain.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from ..config_base import RAW_BASE


def resolve_doc_path(doc_id: str, domain: str) -> Tuple[Optional[Path], str]:
    """
    Resolve doc_id to (file_path, file_type).

    Args:
        doc_id: Document identifier from question JSON.
        domain: One of insurance/regulatory/financial_contracts/financial_reports/research.

    Returns:
        (Path to file, file type 'txt'|'html'|'pdf'). (None, 'unknown') if not found.
    """
    if domain == "insurance":
        path = RAW_BASE / "insurance" / f"{doc_id}.pdf"
        return (path, "pdf") if path.exists() else (None, "unknown")

    if domain == "financial_reports":
        # .PDF (uppercase) takes priority, then .pdf
        path = RAW_BASE / "financial_reports" / f"{doc_id}.PDF"
        if not path.exists():
            path = RAW_BASE / "financial_reports" / f"{doc_id}.pdf"
        return (path, "pdf") if path.exists() else (None, "unknown")

    if domain == "financial_contracts":
        path = RAW_BASE / "financial_contracts" / f"{doc_id}.pdf"
        return (path, "pdf") if path.exists() else (None, "unknown")

    if domain == "research":
        path = RAW_BASE / "research" / f"{doc_id}.pdf"
        return (path, "pdf") if path.exists() else (None, "unknown")

    if domain == "regulatory":
        # strict_v3_XXX series -> txt (highest quality cleaned text)
        if doc_id.startswith("strict_v3_"):
            txt_dir = RAW_BASE / "regulatory" / "txt"
            if txt_dir.exists():
                # Try fuzzy match (filename may have extra suffix)
                for fname in os.listdir(txt_dir):
                    if fname.startswith(doc_id) and fname.endswith(".txt"):
                        return (txt_dir / fname, "txt")
                exact_path = txt_dir / f"{doc_id}.txt"
                if exact_path.exists():
                    return (exact_path, "txt")
            return (None, "unknown")

        # csrc_XXXX_attN -> pdf attachment
        if "_att" in doc_id:
            path = RAW_BASE / "regulatory" / "attachments" / f"{doc_id}.pdf"
            return (path, "pdf") if path.exists() else (None, "unknown")

        # csrc_XXXX -> html main document
        if doc_id.startswith("csrc_"):
            path = RAW_BASE / "regulatory" / "html" / f"{doc_id}.html"
            return (path, "html") if path.exists() else (None, "unknown")

    return (None, "unknown")


def get_file_type(path: Path) -> str:
    """Determine file type from extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".txt":
        return "txt"
    if ext in (".html", ".htm"):
        return "html"
    return "unknown"


def get_all_doc_paths(domain: str) -> List[Tuple[str, Path, str]]:
    """
    Enumerate all documents under a domain directory.

    Returns:
        List of (doc_id, file_path, file_type) tuples.
    """
    results: List[Tuple[str, Path, str]] = []

    if domain == "insurance":
        d = RAW_BASE / "insurance"
        if d.exists():
            for f in sorted(d.glob("*.pdf")):
                results.append((f.stem, f, "pdf"))

    elif domain == "financial_reports":
        d = RAW_BASE / "financial_reports"
        if d.exists():
            for f in list(sorted(d.glob("*.PDF"))) + list(sorted(d.glob("*.pdf"))):
                results.append((f.stem, f, "pdf"))

    elif domain == "financial_contracts":
        d = RAW_BASE / "financial_contracts"
        if d.exists():
            for f in sorted(d.glob("*.pdf")):
                results.append((f.stem, f, "pdf"))

    elif domain == "research":
        d = RAW_BASE / "research"
        if d.exists():
            for f in sorted(d.glob("*.pdf")):
                results.append((f.stem, f, "pdf"))

    elif domain == "regulatory":
        txt_dir = RAW_BASE / "regulatory" / "txt"
        if txt_dir.exists():
            for f in sorted(txt_dir.glob("*.txt")):
                results.append((f.stem, f, "txt"))
        html_dir = RAW_BASE / "regulatory" / "html"
        if html_dir.exists():
            for f in sorted(html_dir.glob("*.html")):
                results.append((f.stem, f, "html"))
        att_dir = RAW_BASE / "regulatory" / "attachments"
        if att_dir.exists():
            for f in sorted(att_dir.glob("*.pdf")):
                results.append((f.stem, f, "pdf"))

    return results
