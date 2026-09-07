#!/usr/bin/env python3
"""
build_regulatory_text_cache.py - Build HiKEY-compatible indices for HTML and TXT regulatory documents.

The original build_hikey_cache.py only processes PDFs. This script handles:
1. TXT files (strict_v3_* series) - structured legal text with chapters/articles
2. HTML files (csrc_XXXX without _att suffix) - CSRC web pages

Output structure (same as PDF indices):
    _shared/_HiKEY_cache/regulatory/{doc_id}/
        ├── doc_card.json
        ├── sections.jsonl
        └── units.jsonl

Usage:
    # Build all HTML + TXT regulatory documents
    python build_regulatory_text_cache.py

    # Build only documents referenced in questions
    python build_regulatory_text_cache.py --questions-only

    # Force rebuild
    python build_regulatory_text_cache.py --force
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import RAW_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HIKEY_CACHE_ROOT = _PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"

# --- Heading patterns for regulatory text ---
# Chapter: 第一章, 第二章, ...
RE_CHAPTER = re.compile(r"^(第[一二三四五六七八九十百零〇]+章)\s*(.*)")
# Article: 第一条, 第二条, ...
RE_ARTICLE = re.compile(r"^(第[一二三四五六七八九十百零〇]+条)\s*(.*)")
# Section within chapter: 第一节, 第二节, ...
RE_SECTION = re.compile(r"^(第[一二三四五六七八九十百零〇]+节)\s*(.*)")
# Numbered items: （一）, （二）, ...
RE_NUMBERED_ITEM = re.compile(r"^[（(][一二三四五六七八九十百零〇]+[)）]\s*(.*)")
# Appendix
RE_APPENDIX = re.compile(r"^(附[则录表件])\s*(.*)")


def parse_txt_document(txt_path: Path, doc_id: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Parse a regulatory TXT file into sections and units.

    Returns:
        (sections, units, doc_card)
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    lines = raw_text.split("\n")

    # Extract title from first non-empty lines
    title_lines = []
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if title_lines:
                content_start = i + 1
                break
            continue
        # Check if this looks like a chapter/article start
        if RE_CHAPTER.match(stripped) or RE_ARTICLE.match(stripped):
            content_start = i
            break
        title_lines.append(stripped)
        content_start = i + 1

    doc_title = " ".join(title_lines) if title_lines else doc_id

    sections: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []

    # Root section
    sections.append({
        "section_id": "root",
        "doc_id": doc_id,
        "title": "ROOT",
        "level": 0,
        "section_path": "ROOT",
        "start_page": 1,
        "end_page": 1,
        "parent_path": "",
        "unit_ids": [],
        "text_preview": "",
        "metadata": {}
    })

    # Title section
    title_sec_id = f"{doc_id}_sec_title"
    sections.append({
        "section_id": title_sec_id,
        "doc_id": doc_id,
        "title": doc_title,
        "level": 1,
        "section_path": doc_title,
        "start_page": 1,
        "end_page": 1,
        "parent_path": "ROOT",
        "unit_ids": [],
        "text_preview": doc_title,
        "metadata": {"parser": "regulatory-text-v1.0"}
    })

    # Parse content into structured sections
    current_chapter = None
    current_section_name = None
    current_article = None
    section_counter = 0
    unit_counter = 0
    paragraph_buffer: List[str] = []

    def flush_paragraph():
        nonlocal unit_counter, paragraph_buffer
        if not paragraph_buffer:
            return
        text = "\n".join(paragraph_buffer)
        paragraph_buffer = []
        if not text.strip():
            return

        unit_counter += 1
        unit_id = f"{doc_id}_unit_{unit_counter:06d}"

        # Determine section context
        sec_id = title_sec_id
        sec_path = doc_title
        if current_article:
            sec_id = current_article["section_id"]
            sec_path = current_article["section_path"]
        elif current_chapter:
            sec_id = current_chapter["section_id"]
            sec_path = current_chapter["section_path"]

        unit = {
            "unit_id": unit_id,
            "doc_id": doc_id,
            "section_id": sec_id,
            "section_path": sec_path,
            "page": 1,
            "unit_type": "text_block",
            "text": text,
            "bbox": None,
            "table_name": None,
            "row_header": None,
            "col_headers": None,
            "values": None,
            "unit": None,
            "metadata": {}
        }
        units.append(unit)

        # Add unit_id to its section
        for sec in sections:
            if sec["section_id"] == sec_id:
                sec["unit_ids"].append(unit_id)
                break

    for i in range(content_start, len(lines)):
        line = lines[i].strip()
        if not line:
            flush_paragraph()
            continue

        # Check for chapter heading
        m = RE_CHAPTER.match(line)
        if m:
            flush_paragraph()
            section_counter += 1
            chapter_num = m.group(1)
            chapter_title = m.group(2).strip() if m.group(2) else ""
            full_title = f"{chapter_num} {chapter_title}".strip()

            sec_id = f"{doc_id}_sec_{section_counter:04d}"
            sec_path = f"{doc_title} > {full_title}"
            current_chapter = {
                "section_id": sec_id,
                "section_path": sec_path,
            }
            current_article = None

            sections.append({
                "section_id": sec_id,
                "doc_id": doc_id,
                "title": full_title,
                "level": 2,
                "section_path": sec_path,
                "start_page": 1,
                "end_page": 1,
                "parent_path": doc_title,
                "unit_ids": [],
                "text_preview": full_title,
                "metadata": {"parser": "regulatory-text-v1.0"}
            })

            # Also add heading as a unit
            unit_counter += 1
            unit_id = f"{doc_id}_unit_{unit_counter:06d}"
            units.append({
                "unit_id": unit_id,
                "doc_id": doc_id,
                "section_id": sec_id,
                "section_path": sec_path,
                "page": 1,
                "unit_type": "heading",
                "text": full_title,
                "bbox": None,
                "table_name": None,
                "row_header": None,
                "col_headers": None,
                "values": None,
                "unit": None,
                "metadata": {}
            })
            sections[-1]["unit_ids"].append(unit_id)
            continue

        # Check for section heading (第X节)
        m = RE_SECTION.match(line)
        if m:
            flush_paragraph()
            section_counter += 1
            sec_num = m.group(1)
            sec_title = m.group(2).strip() if m.group(2) else ""
            full_title = f"{sec_num} {sec_title}".strip()

            parent_path = current_chapter["section_path"] if current_chapter else doc_title
            sec_id = f"{doc_id}_sec_{section_counter:04d}"
            sec_path = f"{parent_path} > {full_title}"
            current_section_name = full_title

            sections.append({
                "section_id": sec_id,
                "doc_id": doc_id,
                "title": full_title,
                "level": 3,
                "section_path": sec_path,
                "start_page": 1,
                "end_page": 1,
                "parent_path": parent_path,
                "unit_ids": [],
                "text_preview": full_title,
                "metadata": {"parser": "regulatory-text-v1.0"}
            })

            # Update current_chapter to point to this sub-section for article assignment
            current_article = None

            unit_counter += 1
            unit_id = f"{doc_id}_unit_{unit_counter:06d}"
            units.append({
                "unit_id": unit_id,
                "doc_id": doc_id,
                "section_id": sec_id,
                "section_path": sec_path,
                "page": 1,
                "unit_type": "heading",
                "text": full_title,
                "bbox": None,
                "table_name": None,
                "row_header": None,
                "col_headers": None,
                "values": None,
                "unit": None,
                "metadata": {}
            })
            sections[-1]["unit_ids"].append(unit_id)
            continue

        # Check for article heading
        m = RE_ARTICLE.match(line)
        if m:
            flush_paragraph()
            section_counter += 1
            article_num = m.group(1)
            article_rest = m.group(2).strip() if m.group(2) else ""

            parent_path = doc_title
            if current_chapter:
                parent_path = current_chapter["section_path"]

            sec_id = f"{doc_id}_sec_{section_counter:04d}"
            sec_path = f"{parent_path} > {article_num}"
            current_article = {
                "section_id": sec_id,
                "section_path": sec_path,
            }

            sections.append({
                "section_id": sec_id,
                "doc_id": doc_id,
                "title": article_num,
                "level": 4,
                "section_path": sec_path,
                "start_page": 1,
                "end_page": 1,
                "parent_path": parent_path,
                "unit_ids": [],
                "text_preview": f"{article_num} {article_rest[:80]}",
                "metadata": {"parser": "regulatory-text-v1.0"}
            })

            # The article text starts with the rest of the line
            if article_rest:
                paragraph_buffer.append(f"{article_num} {article_rest}")
            continue

        # Check for appendix
        m = RE_APPENDIX.match(line)
        if m:
            flush_paragraph()
            section_counter += 1
            appendix_title = line

            sec_id = f"{doc_id}_sec_{section_counter:04d}"
            sec_path = f"{doc_title} > {appendix_title}"
            current_chapter = {
                "section_id": sec_id,
                "section_path": sec_path,
            }
            current_article = None

            sections.append({
                "section_id": sec_id,
                "doc_id": doc_id,
                "title": appendix_title,
                "level": 2,
                "section_path": sec_path,
                "start_page": 1,
                "end_page": 1,
                "parent_path": doc_title,
                "unit_ids": [],
                "text_preview": appendix_title,
                "metadata": {"parser": "regulatory-text-v1.0"}
            })
            continue

        # Regular content line
        paragraph_buffer.append(line)

    # Flush remaining
    flush_paragraph()

    # Doc card
    doc_card = {
        "doc_id": doc_id,
        "title": doc_title,
        "file_type": "txt",
        "total_pages": 1,
        "total_sections": len(sections),
        "total_units": len(units),
        "metadata": {
            "parser": "regulatory-text-v1.0",
            "source_path": str(txt_path),
        }
    }

    return sections, units, doc_card


def parse_html_document(html_path: Path, doc_id: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Parse a CSRC HTML regulatory page into sections and units.

    Returns:
        (sections, units, doc_card)
    """
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 is required for HTML parsing. Install with: pip install beautifulsoup4")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract metadata from meta tags
    meta_title = ""
    meta_date = ""
    meta_source = ""

    title_tag = soup.find("meta", attrs={"name": "ArticleTitle"})
    if title_tag:
        meta_title = title_tag.get("content", "")
    if not meta_title:
        title_el = soup.find("title")
        if title_el:
            meta_title = title_el.get_text(strip=True)

    date_tag = soup.find("meta", attrs={"name": "PubDate"})
    if date_tag:
        meta_date = date_tag.get("content", "")

    # Extract main content
    content_div = None
    for selector in [
        ("div", {"class": "detail-content"}),
        ("div", {"class": "TRS_Editor"}),
        ("div", {"class": "content"}),
        ("div", {"id": "ContentRegion"}),
        ("article", {}),
    ]:
        content_div = soup.find(selector[0], selector[1]) if selector[1] else soup.find(selector[0])
        if content_div:
            break

    if not content_div:
        # Fallback: use body
        content_div = soup.find("body") or soup

    # Get text content, preserving paragraph structure
    paragraphs: List[str] = []
    for element in content_div.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]):
        text = element.get_text(strip=True)
        if text and len(text) > 1:
            # Avoid duplicates from nested elements
            if not paragraphs or text != paragraphs[-1]:
                paragraphs.append(text)

    if not paragraphs:
        # Fallback: split by newlines
        text = content_div.get_text(separator="\n", strip=True)
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Remove navigation/header/footer noise
    noise_keywords = ["首页", "网站地图", "联系我们", "版权所有", "ICP备", "京公网安备",
                      "移动端", "微博", "微信", "无障碍", "English", "请输入关键字"]
    clean_paragraphs = []
    for p in paragraphs:
        if len(p) < 3:
            continue
        if any(kw in p for kw in noise_keywords) and len(p) < 50:
            continue
        clean_paragraphs.append(p)

    doc_title = meta_title or doc_id

    sections: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []

    # Root section
    sections.append({
        "section_id": "root",
        "doc_id": doc_id,
        "title": "ROOT",
        "level": 0,
        "section_path": "ROOT",
        "start_page": 1,
        "end_page": 1,
        "parent_path": "",
        "unit_ids": [],
        "text_preview": "",
        "metadata": {}
    })

    # Main document section
    main_sec_id = f"{doc_id}_sec_main"
    sections.append({
        "section_id": main_sec_id,
        "doc_id": doc_id,
        "title": doc_title,
        "level": 1,
        "section_path": doc_title,
        "start_page": 1,
        "end_page": 1,
        "parent_path": "ROOT",
        "unit_ids": [],
        "text_preview": doc_title,
        "metadata": {
            "parser": "regulatory-html-v1.0",
            "pub_date": meta_date,
        }
    })

    # Parse paragraphs into structured content
    current_section = {"section_id": main_sec_id, "section_path": doc_title}
    section_counter = 0
    unit_counter = 0

    # Group consecutive short lines into blocks, keep long paragraphs as-is
    i = 0
    while i < len(clean_paragraphs):
        para = clean_paragraphs[i]

        # Check if this is a chapter/article heading
        m_chapter = RE_CHAPTER.match(para)
        m_article = RE_ARTICLE.match(para)
        m_section = RE_SECTION.match(para)
        m_appendix = RE_APPENDIX.match(para)

        if m_chapter:
            section_counter += 1
            chapter_title = para
            sec_id = f"{doc_id}_sec_{section_counter:04d}"
            sec_path = f"{doc_title} > {chapter_title}"
            current_section = {"section_id": sec_id, "section_path": sec_path}

            sections.append({
                "section_id": sec_id,
                "doc_id": doc_id,
                "title": chapter_title,
                "level": 2,
                "section_path": sec_path,
                "start_page": 1,
                "end_page": 1,
                "parent_path": doc_title,
                "unit_ids": [],
                "text_preview": chapter_title,
                "metadata": {"parser": "regulatory-html-v1.0"}
            })

            unit_counter += 1
            unit_id = f"{doc_id}_unit_{unit_counter:06d}"
            units.append({
                "unit_id": unit_id,
                "doc_id": doc_id,
                "section_id": sec_id,
                "section_path": sec_path,
                "page": 1,
                "unit_type": "heading",
                "text": chapter_title,
                "bbox": None,
                "table_name": None,
                "row_header": None,
                "col_headers": None,
                "values": None,
                "unit": None,
                "metadata": {}
            })
            sections[-1]["unit_ids"].append(unit_id)
            i += 1
            continue

        if m_appendix:
            section_counter += 1
            appendix_title = para
            sec_id = f"{doc_id}_sec_{section_counter:04d}"
            sec_path = f"{doc_title} > {appendix_title}"
            current_section = {"section_id": sec_id, "section_path": sec_path}

            sections.append({
                "section_id": sec_id,
                "doc_id": doc_id,
                "title": appendix_title,
                "level": 2,
                "section_path": sec_path,
                "start_page": 1,
                "end_page": 1,
                "parent_path": doc_title,
                "unit_ids": [],
                "text_preview": appendix_title,
                "metadata": {"parser": "regulatory-html-v1.0"}
            })
            i += 1
            continue

        # Regular content - create text_block unit
        # Group consecutive content lines that belong together
        block_lines = [para]
        j = i + 1
        while j < len(clean_paragraphs):
            next_para = clean_paragraphs[j]
            # Stop if next line is a structural heading
            if (RE_CHAPTER.match(next_para) or RE_ARTICLE.match(next_para) or
                RE_SECTION.match(next_para) or RE_APPENDIX.match(next_para)):
                break
            # If current block is already long enough, stop
            if sum(len(l) for l in block_lines) > 500:
                break
            block_lines.append(next_para)
            j += 1

        text = "\n".join(block_lines)
        if text.strip():
            unit_counter += 1
            unit_id = f"{doc_id}_unit_{unit_counter:06d}"
            units.append({
                "unit_id": unit_id,
                "doc_id": doc_id,
                "section_id": current_section["section_id"],
                "section_path": current_section["section_path"],
                "page": 1,
                "unit_type": "text_block",
                "text": text,
                "bbox": None,
                "table_name": None,
                "row_header": None,
                "col_headers": None,
                "values": None,
                "unit": None,
                "metadata": {}
            })
            # Add to section
            for sec in sections:
                if sec["section_id"] == current_section["section_id"]:
                    sec["unit_ids"].append(unit_id)
                    break

        i = j

    # Doc card
    doc_card = {
        "doc_id": doc_id,
        "title": doc_title,
        "file_type": "html",
        "total_pages": 1,
        "total_sections": len(sections),
        "total_units": len(units),
        "metadata": {
            "parser": "regulatory-html-v1.0",
            "source_path": str(html_path),
            "pub_date": meta_date,
        }
    }

    return sections, units, doc_card


def build_single_doc(doc_id: str, file_path: Path, file_type: str, output_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Build HiKEY index for a single HTML/TXT document."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already built
    if not force and (output_dir / "sections.jsonl").exists() and (output_dir / "units.jsonl").exists():
        return {"doc_id": doc_id, "status": "skipped", "msg": "already exists"}

    try:
        t0 = time.time()

        if file_type == "txt":
            sections, units, doc_card = parse_txt_document(file_path, doc_id)
        elif file_type == "html":
            sections, units, doc_card = parse_html_document(file_path, doc_id)
        else:
            return {"doc_id": doc_id, "status": "error", "msg": f"unsupported file type: {file_type}"}

        elapsed = time.time() - t0

        # Write outputs
        with open(output_dir / "sections.jsonl", "w", encoding="utf-8") as f:
            for sec in sections:
                f.write(json.dumps(sec, ensure_ascii=False) + "\n")

        with open(output_dir / "units.jsonl", "w", encoding="utf-8") as f:
            for unit in units:
                f.write(json.dumps(unit, ensure_ascii=False) + "\n")

        with open(output_dir / "doc_card.json", "w", encoding="utf-8") as f:
            json.dump(doc_card, f, ensure_ascii=False, indent=2)

        # Empty field_cards.jsonl for compatibility
        with open(output_dir / "field_cards.jsonl", "w", encoding="utf-8") as f:
            pass

        return {
            "doc_id": doc_id,
            "status": "success",
            "elapsed": round(elapsed, 3),
            "sections": len(sections),
            "units": len(units),
            "file_type": file_type,
        }
    except Exception as e:
        import traceback
        return {"doc_id": doc_id, "status": "error", "msg": f"{e}\n{traceback.format_exc()}"}


def get_question_doc_ids() -> set:
    """Get doc_ids referenced in regulatory questions."""
    questions_path = _PROJECT_ROOT / "public_dataset_upload" / "questions" / "group_a" / "regulatory_questions.json"
    if not questions_path.exists():
        return set()
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    doc_ids = set()
    for q in questions:
        doc_ids.update(q.get("doc_ids", []))
    return doc_ids


def main():
    parser = argparse.ArgumentParser(description="Build HiKEY cache for HTML/TXT regulatory documents")
    parser.add_argument("--questions-only", action="store_true",
                        help="Only build indices for documents referenced in questions")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument("--doc-ids", nargs="+", default=None, help="Specific doc_ids to build")
    args = parser.parse_args()

    output_root = HIKEY_CACHE_ROOT / "regulatory"
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect tasks
    tasks: List[Tuple[str, Path, str]] = []

    # TXT files
    txt_dir = RAW_BASE / "regulatory" / "txt"
    if txt_dir.exists():
        for f in sorted(txt_dir.glob("*.txt")):
            doc_id = f.stem
            tasks.append((doc_id, f, "txt"))

    # HTML files
    html_dir = RAW_BASE / "regulatory" / "html"
    if html_dir.exists():
        for f in sorted(html_dir.glob("*.html")):
            doc_id = f.stem
            # Skip HTML files that already have PDF attachments indexed
            # (csrc_XXXX without _att suffix are the main HTML pages)
            tasks.append((doc_id, f, "html"))

    # Filter by question doc_ids if requested
    if args.questions_only:
        question_doc_ids = get_question_doc_ids()
        tasks = [(did, path, ftype) for did, path, ftype in tasks if did in question_doc_ids]
        logger.info(f"Filtered to {len(tasks)} documents referenced in questions")

    # Filter by specific doc_ids if provided
    if args.doc_ids:
        target_ids = set(args.doc_ids)
        tasks = [(did, path, ftype) for did, path, ftype in tasks if did in target_ids]

    logger.info(f"Total documents to process: {len(tasks)} (TXT: {sum(1 for _,_,t in tasks if t=='txt')}, HTML: {sum(1 for _,_,t in tasks if t=='html')})")

    if not tasks:
        logger.info("No documents to process. Exiting.")
        return

    # Process
    results = []
    t_start = time.time()

    for i, (doc_id, file_path, file_type) in enumerate(tasks, 1):
        output_dir = output_root / doc_id
        result = build_single_doc(doc_id, file_path, file_type, output_dir, args.force)
        results.append(result)

        status = result["status"]
        if status == "success":
            logger.info(
                f"[{i}/{len(tasks)}] ✓ {doc_id} ({file_type}, {result['elapsed']}s, "
                f"sec={result['sections']}, units={result['units']})"
            )
        elif status == "skipped":
            logger.info(f"[{i}/{len(tasks)}] ⊘ {doc_id} (skipped)")
        else:
            logger.error(f"[{i}/{len(tasks)}] ✗ {doc_id}: {result.get('msg', 'unknown error')}")

    elapsed_total = time.time() - t_start

    # Update manifest
    manifest_path = output_root / "manifest.json"
    existing_manifest = {"domain": "regulatory", "documents": []}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)

    existing_doc_ids = {d["doc_id"] for d in existing_manifest.get("documents", [])}

    for result in results:
        if result["status"] in ("success", "skipped"):
            doc_id = result["doc_id"]
            if doc_id not in existing_doc_ids:
                existing_manifest["documents"].append({
                    "doc_id": doc_id,
                    "path": str(output_root / doc_id),
                })
                existing_doc_ids.add(doc_id)

    existing_manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    existing_manifest["total_docs"] = len(existing_manifest["documents"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing_manifest, f, ensure_ascii=False, indent=2)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    logger.info(f"\n{'='*60}")
    logger.info(f"BUILD COMPLETE in {elapsed_total:.1f}s")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Errors:  {errors}")
    logger.info(f"  Output:  {output_root}")
    logger.info(f"  Manifest updated: {manifest_path}")
    logger.info(f"{'='*60}")

    if errors:
        logger.info("\nFailed documents:")
        for r in results:
            if r["status"] == "error":
                logger.info(f"  {r['doc_id']}: {r.get('msg', '')[:200]}")


if __name__ == "__main__":
    main()
