"""
HTML document parser (regulatory csrc_* series).
"""
import re
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup


def parse_html(file_path: Path) -> List[Dict]:
    """
    Parse an HTML file into structured blocks split by section headers.

    Returns:
        List of dicts: {text, page_num, section_title}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    content_div = (
        soup.find("div", class_="detail-news")
        or soup.find("div", class_="content")
        or soup.find("div", class_="article")
        or soup.find("div", id="content")
        or soup.find("article")
        or soup.body
    )
    if content_div is None:
        content_div = soup

    text = content_div.get_text(separator="\n", strip=True)
    if not text.strip():
        return []

    blocks: List[Dict] = []
    current_section = ""
    current_text: List[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

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
            current_text.append(stripped)

    if current_text:
        blocks.append({
            "text": "\n".join(current_text),
            "page_num": 1,
            "section_title": current_section,
        })

    return blocks


def _is_section_header(line: str) -> bool:
    if re.match(r"^第[一二三四五六七八九十百千万\d]+[章条节编]", line):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、．.]", line):
        return True
    if re.match(r"^（[一二三四五六七八九十]+）", line):
        return True
    if re.match(r"^\d+[、．.\s]", line) and len(line) < 50:
        return True
    return False
