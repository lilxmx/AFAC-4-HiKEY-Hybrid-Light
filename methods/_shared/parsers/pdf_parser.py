"""
PDF document parser (insurance, financial reports/contracts, research, regulatory attachments).

- parse_pdf            : page-level text blocks via pdfplumber (fallback PyMuPDF).
- parse_pdf_with_tables: page-level text + extracted tables (used for financial_reports).
"""
import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_pdf(file_path: Path, max_pages: Optional[int] = None) -> List[Dict]:
    """Parse a PDF into one block per page with section-title heuristic."""
    import pdfplumber

    blocks: List[Dict] = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            pages = pdf.pages
            if max_pages:
                pages = pages[:max_pages]
            for page in pages:
                text = page.extract_text()
                if text and text.strip():
                    lines = text.strip().split("\n")
                    section_title = ""
                    if lines and _is_section_header(lines[0].strip()):
                        section_title = lines[0].strip()
                    blocks.append({
                        "text": text.strip(),
                        "page_num": page.page_number,
                        "section_title": section_title,
                    })
    except Exception as e:
        print(f"[WARNING] pdfplumber failed for {file_path}: {e}, falling back to PyMuPDF")
        try:
            blocks = _parse_with_pymupdf(file_path, max_pages)
        except Exception as e2:
            print(f"[ERROR] Both PDF parsers failed for {file_path}: {e2}")
            blocks = []

    return blocks


def parse_pdf_with_tables(file_path: Path, max_pages: Optional[int] = None) -> List[Dict]:
    """
    Parse a PDF with table extraction. Recommended for financial_reports
    where numeric tables matter.
    """
    import pdfplumber

    blocks: List[Dict] = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            pages = pdf.pages
            if max_pages:
                pages = pages[:max_pages]
            for page in pages:
                text = page.extract_text() or ""
                tables = page.extract_tables()

                table_text = ""
                if tables:
                    for table in tables:
                        if table:
                            table_text += _table_to_text(table) + "\n"

                combined_text = text.strip()
                if table_text.strip():
                    combined_text += "\n\n[TABLE]\n" + table_text.strip()

                if combined_text.strip():
                    lines = combined_text.strip().split("\n")
                    section_title = ""
                    if lines and _is_section_header(lines[0].strip()):
                        section_title = lines[0].strip()
                    blocks.append({
                        "text": combined_text.strip(),
                        "page_num": page.page_number,
                        "section_title": section_title,
                    })
    except Exception as e:
        print(f"[WARNING] Table extraction failed for {file_path}: {e}, falling back to text-only")
        blocks = parse_pdf(file_path, max_pages)

    return blocks


def _parse_with_pymupdf(file_path: Path, max_pages: Optional[int] = None) -> List[Dict]:
    import fitz  # PyMuPDF

    blocks: List[Dict] = []
    doc = fitz.open(str(file_path))
    try:
        pages = range(len(doc))
        if max_pages:
            pages = range(min(max_pages, len(doc)))

        for page_num in pages:
            page = doc[page_num]
            text = page.get_text()
            if text and text.strip():
                lines = text.strip().split("\n")
                section_title = ""
                if lines and _is_section_header(lines[0].strip()):
                    section_title = lines[0].strip()
                blocks.append({
                    "text": text.strip(),
                    "page_num": page_num + 1,
                    "section_title": section_title,
                })
    finally:
        doc.close()
    return blocks


def _table_to_text(table: list) -> str:
    rows = []
    for row in table:
        if row:
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_section_header(line: str) -> bool:
    if re.match(r"^第[一二三四五六七八九十百千万\d]+[章条节编部分]", line):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、．.]", line):
        return True
    if re.match(r"^（[一二三四五六七八九十]+）", line):
        return True
    if re.match(r"^\d+[、．.\s]", line) and len(line) < 80:
        return True
    return False
