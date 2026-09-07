"""
TXT document parser (regulatory strict_v3 series).
Parses into structured blocks (one block per logical section).
"""
import re
from pathlib import Path
from typing import Dict, List


def parse_txt(file_path: Path) -> List[Dict]:
    """
    Parse a TXT file into structured blocks split by section headers.

    Returns:
        List of dicts: {text, page_num, section_title}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    blocks: List[Dict] = []
    current_section = ""
    current_text: List[str] = []

    for line in content.split("\n"):
        stripped = line.strip()

        if _is_section_header(stripped):
            if current_text:
                blocks.append({
                    "text": "\n".join(current_text),
                    "page_num": 1,
                    "section_title": current_section,
                })
            current_section = stripped
            current_text = [stripped]
        else:
            current_text.append(line)

    if current_text:
        blocks.append({
            "text": "\n".join(current_text),
            "page_num": 1,
            "section_title": current_section,
        })

    return blocks


def _is_section_header(line: str) -> bool:
    """Heuristic detection of section headers in Chinese regulatory text."""
    if re.match(r"^第[一二三四五六七八九十百千万\d]+[章条节编]", line):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、．.]", line):
        return True
    if re.match(r"^（[一二三四五六七八九十]+）", line):
        return True
    if re.match(r"^\d+[、．.\s]", line) and len(line) < 50:
        return True
    return False
