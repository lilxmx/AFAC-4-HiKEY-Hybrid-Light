#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiKEY-Finance: a financial-report oriented implementation of the core HiKEY idea.

This file is intentionally written as a competition-ready scaffold rather than a paper-perfect
reproduction. It keeps the HiKEY design pattern:

    PDF -> DocCard -> SectionCards -> EvidenceUnits -> FieldCards -> Retrieval -> EvidencePack

Major upgrades over the first Lite version:
1) Stronger annual-report heading detection and section-tree construction.
2) More finance-specific evidence units: text_block, table, table_row, financial_field, page_image_stub.
3) FieldCard extraction from table rows and heuristic financial-field lines.
4) Query planning with metric aliases, company/year extraction, and unit/section boosts.
5) Evidence packing: anchor unit + ancestry + sibling units + semantic/lexical associates.
6) VLM-ready hooks: page rendering and JSONL manifest generation for Qwen-VL/Qwen-plus fallback.

@TODO(server): connect the VLM hooks to your Qwen-plus / Qwen-VL API. The local environment here
cannot call your production model endpoint, so the code prepares page images and prompts but does
not send network requests.

@TODO(server): replace the lightweight lexical retriever with bge-m3 / BCE / jina / m3e embeddings
and a cross-encoder reranker if your competition budget allows it.

@TODO(server): for high-value PDFs, plug in a stronger table engine such as MinerU, PaddleOCR PP-Structure,
Docling, Camelot/Tabula, or a VLM table-to-JSON prompt. pdfplumber is used as a lightweight fallback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

VERSION = "hikey-finance-v2.0"
CJK_SECTION_NUM = "一二三四五六七八九十百零〇壹贰叁肆伍陆柒捌玖拾"

# Finance metric ontology. Keep this explicit: it works as field normalization, retrieval boosting,
# and option verification vocabulary.
METRIC_ALIASES: Dict[str, List[str]] = {
    "营业收入": ["营业收入", "营业总收入", "营业额", "收入"],
    "利润总额": ["利润总额"],
    "归属于上市公司股东的净利润": ["归属于上市公司股东的净利润", "归母净利润", "母公司拥有人应占溢利", "归属于母公司股东的净利润"],
    "扣非归母净利润": ["归属于上市公司股东的扣除非经常性损益的净利润", "扣非归母净利润", "扣除非经常性损益后的净利润"],
    "经营活动产生的现金流量净额": ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流量净额", "经营活动现金流"],
    "归属于上市公司股东的净资产": ["归属于上市公司股东的净资产", "归母净资产"],
    "总资产": ["总资产", "资产总计"],
    "总负债": ["负债合计", "总负债"],
    "所有者权益": ["所有者权益", "股东权益"],
    "期末总股本": ["期末总股本", "总股本"],
    "基本每股收益": ["基本每股收益"],
    "稀释每股收益": ["稀释每股收益"],
    "加权平均净资产收益率": ["加权平均净资产收益率", "ROE", "净资产收益率"],
    "研发投入": ["研发投入", "研发费用", "研发经费"],
    "现金红利": ["现金红利", "现金分红", "每10股派", "每 10 股派", "派发现金", "股息", "派息"],
    "资本公积金转增股本": ["资本公积金转增股本", "转增股本", "公积金转增"],
    "送红股": ["送红股", "送股"],
    "新签合同额": ["新签合同额", "新签合同"],
    "合约销售额": ["合约销售额", "销售额"],
    "合约销售面积": ["合约销售面积", "销售面积"],
    "土地储备": ["土地储备"],
    "资产负债率": ["资产负债率"],
    "流动资产": ["流动资产"],
    "非流动资产": ["非流动资产"],
    "流动负债": ["流动负债"],
    "非流动负债": ["非流动负债"],
    "发行规模": ["发行规模", "发行金额", "注册金额"],
    "利率": ["利率", "票面利率"],
    "到期日": ["到期日"],
    "起息日": ["起息日"],
}

COMMON_FINANCIAL_METRICS = sorted({alias for aliases in METRIC_ALIASES.values() for alias in aliases}, key=len, reverse=True)

SECTION_PRIORS = [
    "主要会计数据", "主要财务指标", "管理层讨论与分析", "财务报告", "利润分配", "重要提示",
    "董事会报告", "债券相关", "分季度", "现金流量表", "资产负债表", "利润表", "股东情况",
]

COMPANY_ALIASES = {
    "中国建筑股份有限公司": ["中国建筑股份有限公司", "中国建筑", "中建", "CSCEC"],
    "美的集团股份有限公司": ["美的集团股份有限公司", "美的集团", "美的"],
    "比亚迪股份有限公司": ["比亚迪股份有限公司", "比亚迪", "BYD"],
    "宁德时代新能源科技股份有限公司": ["宁德时代新能源科技股份有限公司", "宁德时代", "CATL"],
    "中国移动有限公司": ["中国移动有限公司", "中国移动"],
    "招商银行股份有限公司": ["招商银行股份有限公司", "招商银行", "招行"],
}

VALUE_RE = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?%?|[-+]?\d+(?:\.\d+)?\s*(?:亿元|万元|千元|元|港元|美元|万平方米|平方米|万m²|m²|GWh|MWh|TWh|%)")
YEAR_RE = re.compile(r"20\d{2}")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ").replace("\ufeff", "")
    s = s.replace(" ", " ").replace("　", " ").replace("ﾠ", " ")
    s = s.replace("，", ",")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def compact(s: str) -> str:
    return re.sub(r"\s+", "", normalize_text(s))


def stable_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def canonical_metric(text: str) -> Optional[str]:
    t = compact(text)
    for canonical, aliases in METRIC_ALIASES.items():
        for a in sorted(aliases, key=len, reverse=True):
            if compact(a) and compact(a) in t:
                return canonical
    return None


def metric_aliases_for_query(query: str) -> List[str]:
    result = []
    q = compact(query)
    for canonical, aliases in METRIC_ALIASES.items():
        if canonical in q or any(compact(a) in q for a in aliases):
            result.extend([canonical] + aliases)
    return list(dict.fromkeys(result))


def normalize_number_token(x: str) -> str:
    return normalize_text(x).replace(",", "").replace("，", "")


def is_noise_line(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    tc = compact(t)
    if re.fullmatch(r"\d{1,4}", tc):
        return True
    # keep these as possible title/context lines near front matter
    if tc in {"目录", "目次", "中国建筑股份有限公司", "2025年年度报告", "年度报告"}:
        return False
    # page footers / headers that repeat and carry little evidence
    if len(tc) <= 2 and not re.search(r"[一二三四五六七八九十章节]", tc):
        return True
    return False


def detect_heading_level(text: str, font_size: float = 0.0) -> Optional[int]:
    """Return heading level 1..5 or None.

    The rules are tuned for Chinese listed-company annual reports. We intentionally do not rely
    exclusively on font size because extracted Chinese PDF font-size metadata is often noisy.
    """
    t = normalize_text(text)
    tc = compact(t)
    if not t or len(tc) > 90:
        return None

    # Main sections or chapters: 第一节 / 第一章.
    if re.match(rf"^第[{CJK_SECTION_NUM}\d]+[章节][\s　\u3000]*.+", tc):
        return 1

    # Major report front-matter headings.
    if tc in {"董事长致辞", "行长致辞", "致股东", "致股东的信", "年报速览", "重要提示", "释义", "目录", "财务报告", "审计报告"}:
        return 1

    # Chinese numbered headings: 一、公司信息
    if re.match(rf"^[{CJK_SECTION_NUM}]+[、.．].+", tc):
        return 2

    # Parenthesized Chinese headings: （一）主要会计数据
    if re.match(rf"^[（(][{CJK_SECTION_NUM}]+[）)].+", tc):
        return 3

    # Arabic multi-level headings: 3.1 总体经营情况分析 / 1.2.3 xxx
    if re.match(r"^\d+(\.\d+)+[\s、.．].+", tc):
        return 3

    # Simple Arabic list headings: 1. xxx / 1、xxx, only when short and looks like title.
    if re.match(r"^\d+[、.．]\s*[^\d].+", t) and len(tc) <= 55:
        return 4

    # Short bold/large lines that look like report headings.
    if font_size >= 15 and len(tc) <= 35 and not VALUE_RE.search(tc):
        return 2

    return None


def update_stack(stack: List[Tuple[int, str]], level: int, title: str) -> List[Tuple[int, str]]:
    title = normalize_text(title)
    new_stack = [(l, h) for l, h in stack if l < level]
    if new_stack and new_stack[-1][1] == title:
        return new_stack
    if stack and any(l == level and h == title for l, h in stack[-2:]):
        return stack
    new_stack.append((level, title))
    return new_stack


def section_path_from_stack(stack: List[Tuple[int, str]]) -> str:
    return " > ".join(h for _, h in stack if h)


def simple_tokens(text: str) -> List[str]:
    """Mixed CJK n-gram + Latin/number tokenizer for dependency-free retrieval."""
    text = normalize_text(text).lower()
    terms: List[str] = []
    terms.extend(re.findall(r"[a-zA-Z]+\d*|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}", text))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    terms.extend(cjk[i:i + 2] for i in range(max(0, len(cjk) - 1)))
    terms.extend(cjk[i:i + 3] for i in range(max(0, len(cjk) - 2)))
    return terms


@dataclass
class DocCard:
    doc_id: str
    source_pdf: str
    title: str
    company: str
    year: Optional[int]
    page_count: int
    parser_version: str = VERSION
    top_sections: List[str] = field(default_factory=list)
    toc_entries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SectionCard:
    section_id: str
    doc_id: str
    title: str
    level: int
    section_path: str
    start_page: int
    end_page: int
    parent_path: str
    unit_ids: List[str] = field(default_factory=list)
    text_preview: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceUnit:
    unit_id: str
    doc_id: str
    section_id: str
    section_path: str
    page: int
    unit_type: str
    text: str
    bbox: Optional[List[float]] = None
    table_name: Optional[str] = None
    row_header: Optional[str] = None
    col_headers: Optional[List[str]] = None
    values: Optional[List[str]] = None
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldCard:
    field_id: str
    doc_id: str
    company: str
    report_year: Optional[int]
    metric: str
    raw_metric: str
    value_map: Dict[str, str]
    unit: Optional[str]
    source_unit_id: str
    source_page: int
    section_path: str
    table_name: Optional[str] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    query: str
    companies: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    answer_format: Optional[str] = None


class HiKEYFinanceParser:
    def __init__(self, pdf_path: str | Path, out_dir: str | Path):
        self.pdf_path = Path(pdf_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.doc_id = self.pdf_path.stem
        self.sections: Dict[str, SectionCard] = {}
        self.units: List[EvidenceUnit] = []
        self.field_cards: List[FieldCard] = []
        self.page_to_section: Dict[int, str] = {}
        self.current_section_id: str = "root"
        self.section_counter = 0
        self.unit_counter = 0
        self.doc_card: Optional[DocCard] = None
        self.page_text_cache: Dict[int, str] = {}

    def next_section_id(self) -> str:
        self.section_counter += 1
        return f"{self.doc_id}_sec_{self.section_counter:04d}"

    def next_unit_id(self, suffix: str = "u") -> str:
        self.unit_counter += 1
        return f"{self.doc_id}_{suffix}_{self.unit_counter:06d}"

    def extract_doc_card(self, doc: fitz.Document, top_sections: List[str]) -> DocCard:
        first_pages = "\n".join(doc[i].get_text("text") for i in range(min(4, doc.page_count)))
        text = normalize_text(first_pages)
        title = ""
        company = ""
        year = None

        m_title = re.search(r"([^\n]{2,80}?(?:股份有限公司|有限公司))\s*(20\d{2})\s*年(?:度)?报告", text)
        if m_title:
            company = normalize_text(m_title.group(1))
            year = int(m_title.group(2))
            title = f"{company}{year}年度报告"
        else:
            m_company = re.search(r"([\u4e00-\u9fffA-Za-z（）()]{2,50}(?:股份有限公司|有限公司))", text)
            if m_company:
                company = normalize_text(m_company.group(1))
            for c, aliases in COMPANY_ALIASES.items():
                if any(a in text[:1200] for a in aliases):
                    company = c
                    break
            m_year = re.search(r"(20\d{2})\s*年(?:度)?报告", text) or re.search(r"年度报告\s*(20\d{2})", text)
            if m_year:
                year = int(m_year.group(1))
            title = f"{company}{year}年度报告" if company and year else self.doc_id

        toc_entries = self.extract_toc_entries(doc)
        return DocCard(
            doc_id=self.doc_id,
            source_pdf=str(self.pdf_path),
            title=title,
            company=company,
            year=year,
            page_count=doc.page_count,
            top_sections=top_sections,
            toc_entries=toc_entries,
        )

    def extract_toc_entries(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        entries = []
        for pno in range(1, min(doc.page_count, 10) + 1):
            lines = [normalize_text(x) for x in doc[pno - 1].get_text("text").splitlines() if normalize_text(x)]
            for i, line in enumerate(lines):
                level = detect_heading_level(line)
                if level is not None:
                    page = None
                    if i + 1 < len(lines) and re.fullmatch(r"\d{1,4}", compact(lines[i + 1])):
                        page = int(compact(lines[i + 1]))
                    else:
                        mm = re.search(r"(\d{1,4})$", compact(line))
                        if mm:
                            page = int(mm.group(1))
                    entries.append({"title": line, "level": level, "page": page, "source_page": pno})
        seen = set()
        result = []
        for e in entries:
            key = (e["title"], e.get("page"))
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    def get_page_lines(self, page: fitz.Page) -> List[Dict[str, Any]]:
        data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        lines = []
        for b in data.get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                spans = line.get("spans", [])
                text = normalize_text("".join(s.get("text", "") for s in spans))
                if not text:
                    continue
                font_size = max((s.get("size", 0.0) for s in spans), default=0.0)
                font_flags = max((s.get("flags", 0) for s in spans), default=0)
                bbox = list(line.get("bbox", b.get("bbox", [0, 0, 0, 0])))
                lines.append({"text": text, "bbox": bbox, "font_size": font_size, "font_flags": font_flags})
        lines.sort(key=lambda x: (round(x["bbox"][1], 1), x["bbox"][0]))
        return lines

    def create_section_if_needed(self, title: str, level: int, stack: List[Tuple[int, str]], page_no: int) -> Tuple[List[Tuple[int, str]], str]:
        new_stack = update_stack(stack, level, title)
        section_path = section_path_from_stack(new_stack)
        parent_path = section_path_from_stack(new_stack[:-1])

        current = self.sections.get(self.current_section_id)
        if current and current.title == normalize_text(title) and current.section_path == section_path:
            current.end_page = page_no
            return new_stack, current.section_id

        # Reuse exact section path if encountered because of repeated headers.
        for sec in self.sections.values():
            if sec.section_path == section_path:
                sec.end_page = max(sec.end_page, page_no)
                self.current_section_id = sec.section_id
                return new_stack, sec.section_id

        section_id = self.next_section_id()
        self.sections[section_id] = SectionCard(
            section_id=section_id,
            doc_id=self.doc_id,
            title=normalize_text(title),
            level=level,
            section_path=section_path,
            start_page=page_no,
            end_page=page_no,
            parent_path=parent_path,
            metadata={"parser": VERSION},
        )
        self.current_section_id = section_id
        return new_stack, section_id

    def add_text_unit(self, text: str, page_no: int, bbox: Optional[List[float]], section_id: str, unit_type: str = "text_block"):
        text = normalize_text(text)
        if len(compact(text)) < 8:
            return
        sec = self.sections.get(section_id) or self.sections["root"]
        unit = EvidenceUnit(
            unit_id=self.next_unit_id("text" if unit_type == "text_block" else unit_type),
            doc_id=self.doc_id,
            section_id=sec.section_id,
            section_path=sec.section_path,
            page=page_no,
            unit_type=unit_type,
            text=text,
            bbox=bbox,
        )
        self.units.append(unit)
        sec.unit_ids.append(unit.unit_id)
        if len(sec.text_preview) < 900:
            sec.text_preview = normalize_text((sec.text_preview + "\n" + text)[:1200])

    def is_probable_toc_page(self, page_no: int, lines: List[Dict[str, Any]]) -> bool:
        if page_no > 12:
            return False
        texts = [compact(x["text"]) for x in lines if compact(x["text"])]
        heading_like = sum(1 for t in texts if detect_heading_level(t) is not None)
        number_like = sum(1 for t in texts if re.fullmatch(r"\d{1,4}", t))
        has_toc = any(t in {"目录", "目次"} for t in texts)
        return has_toc and heading_like >= 4 and number_like >= 4

    def parse_text_and_sections(self, doc: fitz.Document) -> None:
        self.sections["root"] = SectionCard(
            section_id="root",
            doc_id=self.doc_id,
            title="ROOT",
            level=0,
            section_path="ROOT",
            start_page=1,
            end_page=doc.page_count,
            parent_path="",
        )
        stack: List[Tuple[int, str]] = []
        current_section_id = "root"

        for pidx in range(doc.page_count):
            page_no = pidx + 1
            lines = self.get_page_lines(doc[pidx])
            self.page_text_cache[page_no] = "\n".join(x["text"] for x in lines)
            toc_page = self.is_probable_toc_page(page_no, lines)
            buffer: List[str] = []
            buffer_bbox: Optional[List[float]] = None

            def flush():
                nonlocal buffer, buffer_bbox, current_section_id
                if buffer:
                    self.add_text_unit("\n".join(buffer), page_no, buffer_bbox, current_section_id)
                    buffer = []
                    buffer_bbox = None

            for item in lines:
                text = item["text"]
                if is_noise_line(text):
                    continue
                level = None if toc_page else detect_heading_level(text, item.get("font_size", 0.0))
                if level is not None:
                    flush()
                    stack, current_section_id = self.create_section_if_needed(text, level, stack, page_no)
                    self.sections[current_section_id].end_page = page_no
                    self.page_to_section[page_no] = current_section_id
                    self.add_text_unit(text, page_no, item.get("bbox"), current_section_id, unit_type="heading")
                else:
                    if buffer_bbox is None:
                        buffer_bbox = list(item.get("bbox", [0, 0, 0, 0]))
                    else:
                        b = item.get("bbox", [0, 0, 0, 0])
                        buffer_bbox = [min(buffer_bbox[0], b[0]), min(buffer_bbox[1], b[1]), max(buffer_bbox[2], b[2]), max(buffer_bbox[3], b[3])]
                    buffer.append(text)
                    if sum(len(x) for x in buffer) > 1100:
                        flush()
            flush()
            if page_no not in self.page_to_section:
                self.page_to_section[page_no] = current_section_id
            if current_section_id in self.sections:
                self.sections[current_section_id].end_page = max(self.sections[current_section_id].end_page, page_no)

    def infer_unit_from_context(self, section_id: str, table_text: str, page_no: Optional[int] = None) -> Optional[str]:
        candidates = [table_text]
        sec = self.sections.get(section_id)
        if sec:
            candidates.append(sec.text_preview)
        if page_no and page_no in self.page_text_cache:
            candidates.append(self.page_text_cache[page_no])
        text = "\n".join(candidates)
        # Common report table unit forms.
        patterns = [
            r"单位[:：]\s*([^\n\s,，；;]{1,24})",
            r"金额单位[:：]\s*([^\n\s,，；;]{1,24})",
            r"除特别注明外[,，]?单位[:：]\s*([^\n\s,，；;]{1,24})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        for u in ["千元", "百万元", "万元", "亿元", "元", "港元", "美元", "%", "万平方米", "GWh"]:
            if f"单位：{u}" in text or f"单位:{u}" in text:
                return u
        return None

    def find_nearest_table_name(self, section_path: str, preview: str, page_no: int) -> str:
        # Better table-name candidate from section path + nearby preview.
        for pat in [r"表\s*\d+[:：]?([^\n]{2,40})", r"([\u4e00-\u9fff]{2,30}(?:表|指标|情况|明细))"]:
            m = re.search(pat, preview)
            if m:
                return normalize_text(m.group(1))
        parts = [p.strip() for p in section_path.split(">") if p.strip()]
        return parts[-1] if parts else f"page_{page_no}_table"

    def add_financial_line_units(self, doc: fitz.Document) -> None:
        """Create finance-specific field units from text lines.

        This is robust when table extraction fails. It captures a metric line plus nearby numbers and
        header labels. It is not a substitute for table parsing, but a strong fallback for competition QA.
        """
        metric_re = re.compile("|".join(re.escape(x) for x in COMMON_FINANCIAL_METRICS))
        for pidx in range(doc.page_count):
            page_no = pidx + 1
            section_id = self.page_to_section.get(page_no, "root")
            sec = self.sections.get(section_id) or self.sections["root"]
            lines = [x["text"] for x in self.get_page_lines(doc[pidx])]
            i = 0
            while i < len(lines):
                line = normalize_text(lines[i])
                m = metric_re.search(line)
                if not m:
                    i += 1
                    continue
                raw_metric = m.group(0)
                metric = canonical_metric(raw_metric) or raw_metric
                collected = [line]
                j = i + 1
                while j < min(len(lines), i + 10):
                    nxt = normalize_text(lines[j])
                    if not nxt:
                        j += 1
                        continue
                    if detect_heading_level(nxt) is not None:
                        break
                    if j > i + 1 and metric_re.search(nxt) and not VALUE_RE.search(nxt):
                        break
                    if VALUE_RE.search(nxt) or YEAR_RE.search(nxt) or re.search(r"调整后|调整前|同期增减|本期比上年|本年比上年|第一季度|第二季度|第三季度|第四季度|不适用|适用|不实施|不以", nxt) or len(compact(nxt)) <= 22:
                        collected.append(nxt)
                    j += 1
                row_text = "；".join(collected)
                if (VALUE_RE.search(row_text) or any(k in row_text for k in ["不实施", "不以", "不送红股"])) and len(compact(row_text)) >= 8:
                    unit_obj = EvidenceUnit(
                        unit_id=self.next_unit_id("field"),
                        doc_id=self.doc_id,
                        section_id=sec.section_id,
                        section_path=sec.section_path,
                        page=page_no,
                        unit_type="financial_field",
                        text=row_text,
                        table_name=self.find_nearest_table_name(sec.section_path, sec.text_preview, page_no),
                        row_header=metric,
                        unit=self.infer_unit_from_context(section_id, row_text, page_no=page_no),
                        metadata={"source": "line_heuristic", "line_index": i, "raw_metric": raw_metric, "canonical_metric": metric},
                    )
                    self.units.append(unit_obj)
                    sec.unit_ids.append(unit_obj.unit_id)
                    i = max(i + 1, j - 1)
                else:
                    i += 1

    def add_page_image_stub_units(self, doc: fitz.Document) -> None:
        """Add page-level VLM fallback stubs.

        These units are not meant for normal text QA. They mark pages that may need visual inspection:
        pages with many images or low text density. Use `render-pages` or `prepare-vlm` to create images.
        """
        for pidx in range(doc.page_count):
            page_no = pidx + 1
            page = doc[pidx]
            images = page.get_images(full=True)
            text_len = len(compact(page.get_text("text")))
            if images or text_len < 80:
                section_id = self.page_to_section.get(page_no, "root")
                sec = self.sections.get(section_id) or self.sections["root"]
                u = EvidenceUnit(
                    unit_id=self.next_unit_id("pageimg"),
                    doc_id=self.doc_id,
                    section_id=sec.section_id,
                    section_path=sec.section_path,
                    page=page_no,
                    unit_type="page_image_stub",
                    text=f"VLM fallback candidate page {page_no}; text_len={text_len}; images={len(images)}; section={sec.section_path}",
                    metadata={"image_count": len(images), "text_len": text_len, "todo": "@TODO(server): render and send this page to Qwen-plus/Qwen-VL when text/table retrieval is insufficient."},
                )
                self.units.append(u)
                sec.unit_ids.append(u.unit_id)

    def add_table_units(self) -> None:
        if pdfplumber is None:
            print("[WARN] pdfplumber is unavailable; skip table extraction. @TODO(server): install pdfplumber/camelot/MinerU for stronger table parsing.")
            return
        with pdfplumber.open(str(self.pdf_path)) as pdf:
            for pidx, page in enumerate(pdf.pages):
                page_no = pidx + 1
                section_id = self.page_to_section.get(page_no, "root")
                sec = self.sections.get(section_id) or self.sections["root"]
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for ti, table in enumerate(tables or []):
                    cleaned = []
                    for row in table:
                        row = [normalize_text(c or "") for c in row]
                        if any(c for c in row):
                            cleaned.append(row)
                    if len(cleaned) < 2:
                        continue
                    text_len = sum(len(c) for row in cleaned for c in row)
                    if text_len < 20:
                        continue
                    max_cols = max(len(r) for r in cleaned)
                    rows = [r + [""] * (max_cols - len(r)) for r in cleaned]
                    table_markdown = self.rows_to_markdown(rows[:120])
                    unit_label = self.infer_unit_from_context(section_id, table_markdown, page_no=page_no)
                    table_unit_id = self.next_unit_id("table")
                    table_name = self.find_nearest_table_name(sec.section_path, sec.text_preview, page_no)
                    table_unit = EvidenceUnit(
                        unit_id=table_unit_id,
                        doc_id=self.doc_id,
                        section_id=sec.section_id,
                        section_path=sec.section_path,
                        page=page_no,
                        unit_type="table",
                        text=table_markdown,
                        table_name=table_name,
                        unit=unit_label,
                        metadata={"table_index_on_page": ti, "n_rows": len(rows), "n_cols": max_cols, "parser": "pdfplumber"},
                    )
                    self.units.append(table_unit)
                    sec.unit_ids.append(table_unit_id)

                    header_rows = self.detect_header_rows(rows)
                    col_headers = self.make_col_headers(header_rows, max_cols)
                    start_idx = max(1, len(header_rows))
                    for ri, row in enumerate(rows[start_idx:], start=start_idx):
                        if not any(row):
                            continue
                        row_header = next((c for c in row if c and not VALUE_RE.fullmatch(c)), row[0] if row else "")
                        row_text = "；".join(f"{col_headers[i] or f'col{i + 1}'}={row[i]}" for i in range(max_cols) if row[i])
                        if len(compact(row_text)) < 8:
                            continue
                        value_map = {col_headers[i] or f"col{i + 1}": row[i] for i in range(max_cols) if row[i]}
                        row_unit = EvidenceUnit(
                            unit_id=self.next_unit_id("row"),
                            doc_id=self.doc_id,
                            section_id=sec.section_id,
                            section_path=sec.section_path,
                            page=page_no,
                            unit_type="table_row",
                            text=row_text,
                            table_name=table_name,
                            row_header=canonical_metric(row_header) or row_header,
                            col_headers=col_headers,
                            values=row,
                            unit=unit_label,
                            metadata={"table_index_on_page": ti, "row_index": ri, "parent_table_unit_id": table_unit_id, "value_map": value_map},
                        )
                        self.units.append(row_unit)
                        sec.unit_ids.append(row_unit.unit_id)

    @staticmethod
    def rows_to_markdown(rows: List[List[str]]) -> str:
        if not rows:
            return ""
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        header = rows[0]
        sep = ["---"] * n
        body = rows[1:]

        def fmt(r: List[str]) -> str:
            return "| " + " | ".join((c or "").replace("\n", " ") for c in r) + " |"

        return "\n".join([fmt(header), fmt(sep)] + [fmt(r) for r in body])

    @staticmethod
    def detect_header_rows(rows: List[List[str]]) -> List[List[str]]:
        # Header rows usually contain years/labels and fewer numeric values than data rows.
        if not rows:
            return []
        header_rows = [rows[0]]
        if len(rows) > 1:
            second = " ".join(rows[1])
            first = " ".join(rows[0])
            if YEAR_RE.search(second) or "调整" in second or (not VALUE_RE.search(first) and not VALUE_RE.search(second)):
                header_rows.append(rows[1])
        return header_rows[:2]

    @staticmethod
    def make_col_headers(header_rows: List[List[str]], n: int) -> List[str]:
        headers = []
        for i in range(n):
            parts = []
            for r in header_rows:
                if i < len(r) and r[i]:
                    parts.append(r[i])
            header = "/".join(dict.fromkeys(parts)) if parts else f"col{i + 1}"
            headers.append(header)
        return headers

    def build_field_cards(self) -> None:
        """Create FieldCard cache from financial_field/table_row units.

        This is the key competition optimization: many QA options should be verified against FieldCards
        before falling back to free-form RAG.
        """
        company = self.doc_card.company if self.doc_card else ""
        report_year = self.doc_card.year if self.doc_card else None
        cards: List[FieldCard] = []
        seen = set()
        for u in self.units:
            if u.unit_type not in {"financial_field", "table_row"}:
                continue
            metric = canonical_metric(u.row_header or u.text)
            if not metric:
                continue
            value_map: Dict[str, str] = {}
            if u.metadata.get("value_map"):
                value_map.update({normalize_text(k): normalize_text(v) for k, v in u.metadata["value_map"].items() if normalize_text(v)})
            else:
                nums = VALUE_RE.findall(u.text)
                years = YEAR_RE.findall(u.text)
                # Heuristic alignment: if years exist, pair nearby numeric values in order; otherwise v1/v2...
                if years and nums:
                    for y, v in zip(years, nums):
                        value_map[y] = normalize_text(v)
                for idx, v in enumerate(nums, start=1):
                    value_map.setdefault(f"v{idx}", normalize_text(v))
            if not value_map and not any(k in u.text for k in ["不实施", "不以", "不送红股"]):
                continue
            key = (metric, u.page, u.section_path, u.text[:80])
            if key in seen:
                continue
            seen.add(key)
            confidence = 0.75 if u.unit_type == "table_row" else 0.58
            if u.unit:
                confidence += 0.05
            if any(p in u.section_path for p in SECTION_PRIORS):
                confidence += 0.05
            cards.append(FieldCard(
                field_id=f"{self.doc_id}_field_{stable_hash(metric + u.unit_id)}",
                doc_id=self.doc_id,
                company=company,
                report_year=report_year,
                metric=metric,
                raw_metric=u.row_header or metric,
                value_map=value_map,
                unit=u.unit,
                source_unit_id=u.unit_id,
                source_page=u.page,
                section_path=u.section_path,
                table_name=u.table_name,
                confidence=round(min(confidence, 0.95), 3),
                metadata={"source_unit_type": u.unit_type, "source_text": u.text[:500]},
            ))
        self.field_cards = cards

    def write_outputs(self, doc_card: DocCard) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        doc_card_path = self.out_dir / f"{self.doc_id}_doc_card.json"
        with open(doc_card_path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc_card), f, ensure_ascii=False, indent=2)
        paths["doc_card"] = str(doc_card_path)

        sections_path = self.out_dir / f"{self.doc_id}_sections.jsonl"
        with open(sections_path, "w", encoding="utf-8") as f:
            for sec in self.sections.values():
                f.write(json.dumps(asdict(sec), ensure_ascii=False) + "\n")
        paths["sections"] = str(sections_path)

        units_path = self.out_dir / f"{self.doc_id}_units.jsonl"
        with open(units_path, "w", encoding="utf-8") as f:
            for u in self.units:
                f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
        paths["units"] = str(units_path)

        field_cards_path = self.out_dir / f"{self.doc_id}_field_cards.jsonl"
        with open(field_cards_path, "w", encoding="utf-8") as f:
            for c in self.field_cards:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        paths["field_cards"] = str(field_cards_path)

        csv_path = self.out_dir / f"{self.doc_id}_table_rows.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["unit_id", "page", "section_path", "table_name", "row_header", "unit", "unit_type", "text"])
            writer.writeheader()
            for u in self.units:
                if u.unit_type in {"table_row", "financial_field"}:
                    writer.writerow({
                        "unit_id": u.unit_id,
                        "page": u.page,
                        "section_path": u.section_path,
                        "table_name": u.table_name or "",
                        "row_header": u.row_header or "",
                        "unit": u.unit or "",
                        "unit_type": u.unit_type,
                        "text": u.text,
                    })
        paths["table_rows_csv"] = str(csv_path)

        field_csv_path = self.out_dir / f"{self.doc_id}_field_cards.csv"
        with open(field_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["field_id", "metric", "unit", "source_page", "section_path", "table_name", "confidence", "value_map", "source_text"])
            writer.writeheader()
            for c in self.field_cards:
                writer.writerow({
                    "field_id": c.field_id,
                    "metric": c.metric,
                    "unit": c.unit or "",
                    "source_page": c.source_page,
                    "section_path": c.section_path,
                    "table_name": c.table_name or "",
                    "confidence": c.confidence,
                    "value_map": json.dumps(c.value_map, ensure_ascii=False),
                    "source_text": c.metadata.get("source_text", ""),
                })
        paths["field_cards_csv"] = str(field_csv_path)

        report_path = self.out_dir / f"{self.doc_id}_parse_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.make_report(doc_card))
        paths["report"] = str(report_path)
        return paths

    def make_report(self, doc_card: DocCard) -> str:
        unit_type_counter = Counter(u.unit_type for u in self.units)
        lines = [
            f"# HiKEY-Finance Parse Report: {self.doc_id}",
            "",
            "## Parser",
            f"- version: {VERSION}",
            "- status: competition-oriented scaffold",
            "",
            "## Doc Card",
            f"- title: {doc_card.title}",
            f"- company: {doc_card.company}",
            f"- year: {doc_card.year}",
            f"- page_count: {doc_card.page_count}",
            "",
            "## Counts",
            f"- sections: {len(self.sections)}",
            f"- units: {len(self.units)}",
            f"- field_cards: {len(self.field_cards)}",
        ]
        for k, v in unit_type_counter.most_common():
            lines.append(f"  - {k}: {v}")
        lines += ["", "## Top Sections"]
        for sec in list(self.sections.values())[:40]:
            lines.append(f"- p.{sec.start_page}-{sec.end_page} | L{sec.level} | {sec.section_path} | units={len(sec.unit_ids)}")
        lines += ["", "## Sample FieldCards"]
        for c in self.field_cards[:30]:
            lines.append(f"- p.{c.source_page} | {c.metric} | unit={c.unit} | values={json.dumps(c.value_map, ensure_ascii=False)[:160]} | {c.section_path}")
        lines += [
            "",
            "## @TODO(server)",
            "- Connect `prepare-vlm` manifests to Qwen-plus/Qwen-VL for page/table fallback.",
            "- Replace lexical retrieval with embedding + reranker if token/time budget allows.",
            "- Use a stronger table parser or VLM table-to-JSON for cross-page and merged-cell tables.",
            "- Add an option-level verifier for single/multiple-choice QA: claim -> evidence pack -> supported/contradicted/insufficient.",
        ]
        return "\n".join(lines) + "\n"

    def parse(self, with_tables: bool = True, add_vlm_stubs: bool = True) -> Dict[str, str]:
        doc = fitz.open(str(self.pdf_path))
        self.parse_text_and_sections(doc)
        top_sections = []
        seen = set()
        for sec in self.sections.values():
            if sec.level == 1 and sec.title not in seen:
                seen.add(sec.title)
                top_sections.append(sec.title)
        self.doc_card = self.extract_doc_card(doc, top_sections)
        self.add_financial_line_units(doc)
        if with_tables:
            self.add_table_units()
        if add_vlm_stubs:
            self.add_page_image_stub_units(doc)
        self.build_field_cards()
        return self.write_outputs(self.doc_card)


class QueryPlanner:
    @staticmethod
    def plan(query: str) -> QueryPlan:
        q = normalize_text(query)
        companies = []
        for canonical, aliases in COMPANY_ALIASES.items():
            if any(a and a in q for a in aliases):
                companies.append(canonical)
        years = sorted({int(x) for x in YEAR_RE.findall(q)})
        metrics = []
        for canonical, aliases in METRIC_ALIASES.items():
            if canonical in q or any(a and a in q for a in aliases):
                metrics.append(canonical)
        operations = []
        if any(k in q for k in ["同比", "比上年", "增长", "下降", "增加", "减少"]):
            operations.append("compare_yoy")
        if any(k in q for k in ["是否", "正确", "错误", "符合", "不符合"]):
            operations.append("verify_claim")
        if any(k in q for k in ["多选", "哪些", "有？", "正确的有"]):
            answer_format = "multi"
        elif any(k in q for k in ["单选", "哪一项", "哪项"]):
            answer_format = "single"
        else:
            answer_format = None
        return QueryPlan(query=q, companies=companies, years=years, metrics=metrics, operations=operations, answer_format=answer_format)


class SimpleRetriever:
    def __init__(self, sections_path: str | Path, units_path: str | Path, field_cards_path: Optional[str | Path] = None):
        self.sections = [json.loads(line) for line in open(sections_path, encoding="utf-8")]
        self.units = [json.loads(line) for line in open(units_path, encoding="utf-8")]
        self.field_cards = []
        if field_cards_path and Path(field_cards_path).exists():
            self.field_cards = [json.loads(line) for line in open(field_cards_path, encoding="utf-8")]
        self.unit_by_id = {u["unit_id"]: u for u in self.units}
        self.section_by_id = {s["section_id"]: s for s in self.sections}
        self.units_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for u in self.units:
            self.units_by_section[u.get("section_id", "")].append(u)
        self.docs: List[Tuple[str, Dict[str, Any], List[str], Counter]] = []
        for obj in self.sections:
            text = " ".join(str(obj.get(k, "")) for k in ["section_path", "title", "text_preview"])
            toks = simple_tokens(text)
            self.docs.append(("section", obj, toks, Counter(toks)))
        for obj in self.units:
            text = " ".join(str(obj.get(k, "")) for k in ["section_path", "unit_type", "text", "row_header", "table_name", "unit"])
            toks = simple_tokens(text)
            self.docs.append(("unit", obj, toks, Counter(toks)))
        for obj in self.field_cards:
            text = " ".join([obj.get("section_path", ""), obj.get("metric", ""), obj.get("raw_metric", ""), json.dumps(obj.get("value_map", {}), ensure_ascii=False), obj.get("unit") or ""])
            toks = simple_tokens(text)
            self.docs.append(("field_card", obj, toks, Counter(toks)))
        self.df = Counter()
        for _, _, toks, _ in self.docs:
            self.df.update(set(toks))
        self.N = len(self.docs)
        self.avgdl = statistics.mean(len(toks) for _, _, toks, _ in self.docs) if self.docs else 1

    def score_bm25(self, q_toks: List[str], toks: List[str], tf: Counter) -> float:
        score = 0.0
        dl = max(1, len(toks))
        k1, b = 1.5, 0.75
        for t in q_toks:
            if t not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
        return score

    def domain_boost(self, obj_type: str, obj: Dict[str, Any], plan: QueryPlan) -> float:
        hay = json.dumps(obj, ensure_ascii=False)
        boost = 0.0
        for metric in plan.metrics:
            aliases = METRIC_ALIASES.get(metric, [metric])
            if obj.get("metric") == metric or obj.get("row_header") == metric:
                boost += 8.0
            if any(a in hay for a in aliases):
                boost += 3.0
        for y in plan.years:
            if str(y) in hay:
                boost += 1.2
        for c in plan.companies:
            if c in hay or any(a in hay for a in COMPANY_ALIASES.get(c, [])):
                boost += 1.0
        if any(p in obj.get("section_path", "") for p in SECTION_PRIORS):
            boost += 0.8
        if obj_type == "field_card":
            boost += 2.5
        elif obj.get("unit_type") in {"financial_field", "table_row"}:
            boost += 1.5
        elif obj.get("unit_type") == "page_image_stub":
            boost -= 1.0
        return boost

    def search(self, query: str, topk: int = 10, unit_type: Optional[str] = None, object_type: Optional[str] = None) -> List[Dict[str, Any]]:
        plan = QueryPlanner.plan(query)
        expanded_query = query + " " + " ".join(metric_aliases_for_query(query))
        q_toks = simple_tokens(expanded_query)
        results = []
        for obj_type, obj, toks, tf in self.docs:
            if object_type and obj_type != object_type:
                continue
            if unit_type and obj.get("unit_type") != unit_type:
                continue
            sc = self.score_bm25(q_toks, toks, tf) + self.domain_boost(obj_type, obj, plan)
            hay = json.dumps(obj, ensure_ascii=False)
            for word in set(re.findall(r"[\u4e00-\u9fff]{2,}|20\d{2}|\d+(?:\.\d+)?%?", query)):
                if word in hay:
                    sc += 0.6
            if sc > 0:
                item = dict(obj)
                item["object_type"] = obj_type
                item["score"] = round(sc, 4)
                item["query_plan"] = asdict(plan)
                results.append(item)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]

    def pack_evidence(self, query: str, topk: int = 5, max_siblings: int = 4) -> Dict[str, Any]:
        """HiKEY-style evidence pack: anchor + ancestry + siblings + lexical associates."""
        anchors = self.search(query, topk=topk)
        packed = []
        used_ids = set()
        for a in anchors:
            if a.get("object_type") == "field_card":
                source_unit = self.unit_by_id.get(a.get("source_unit_id"), {})
                section_id = source_unit.get("section_id", "")
            else:
                source_unit = a if a.get("unit_id") else {}
                section_id = a.get("section_id", "")
            sec = self.section_by_id.get(section_id, {})
            siblings = []
            for u in self.units_by_section.get(section_id, []):
                if u.get("unit_id") == source_unit.get("unit_id"):
                    continue
                if u.get("unit_type") in {"heading", "financial_field", "table_row", "table"} or any(k in u.get("text", "") for k in ["单位", "调整", "同比", "不实施", "现金分红"]):
                    siblings.append(u)
                if len(siblings) >= max_siblings:
                    break
            item = {
                "anchor": a,
                "ancestry": {
                    "section_id": section_id,
                    "section_path": sec.get("section_path") or a.get("section_path"),
                    "start_page": sec.get("start_page"),
                    "end_page": sec.get("end_page"),
                    "title": sec.get("title"),
                },
                "siblings": siblings,
            }
            anchor_id = a.get("unit_id") or a.get("field_id") or stable_hash(json.dumps(a, ensure_ascii=False))
            if anchor_id not in used_ids:
                used_ids.add(anchor_id)
                packed.append(item)
        return {"query": query, "query_plan": asdict(QueryPlanner.plan(query)), "evidence_pack": packed}


def render_pages(pdf_path: str | Path, out_dir: str | Path, pages: Sequence[int], zoom: float = 2.0) -> List[Dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    manifest = []
    mat = fitz.Matrix(zoom, zoom)
    for page_no in pages:
        if page_no < 1 or page_no > doc.page_count:
            continue
        pix = doc[page_no - 1].get_pixmap(matrix=mat, alpha=False)
        img_path = out / f"{Path(pdf_path).stem}_p{page_no:04d}.png"
        pix.save(str(img_path))
        manifest.append({"page": page_no, "image_path": str(img_path), "width": pix.width, "height": pix.height})
    return manifest


def prepare_vlm_manifest(pdf_path: str | Path, units_path: str | Path, out_dir: str | Path, top_pages: Optional[List[int]] = None) -> str:
    """Render candidate pages and create a JSONL manifest for VLM fallback.

    @TODO(server): call Qwen-plus/Qwen-VL with these image paths and prompts, then merge the returned
    table JSON or visual captions back into EvidenceUnits / FieldCards.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    units = [json.loads(line) for line in open(units_path, encoding="utf-8")]
    if top_pages is None:
        # Prioritize page_image_stub pages, then pages with financial fields.
        pages = []
        for u in units:
            if u.get("unit_type") == "page_image_stub":
                pages.append(int(u["page"]))
        for u in units:
            if u.get("unit_type") in {"financial_field", "table_row"}:
                pages.append(int(u["page"]))
        top_pages = list(dict.fromkeys(pages))[:60]
    manifest_imgs = render_pages(pdf_path, out / "vlm_pages", top_pages)
    manifest_path = out / f"{Path(pdf_path).stem}_vlm_manifest.jsonl"
    prompt = (
        "你是金融年报PDF解析助手。请从图片中提取表格、图表和关键财务字段。"
        "输出JSON，字段包括：page, table_name, unit, rows，其中rows包含row_header、col_headers、values。"
        "重点关注营业收入、归母净利润、经营活动现金流量净额、资产负债率、现金分红、研发投入、债券信息。"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest_imgs:
            rec = {**item, "prompt": prompt, "todo": "@TODO(server): send image_path + prompt to Qwen-plus/Qwen-VL and save result as *_vlm_tables.jsonl"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(manifest_path)


def main():
    ap = argparse.ArgumentParser(description=f"{VERSION}: HiKEY-style parser/retriever for financial PDFs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse")
    p_parse.add_argument("pdf")
    p_parse.add_argument("--out_dir", default="/mnt/data/hikey_finance_output")
    p_parse.add_argument("--no_tables", action="store_true")
    p_parse.add_argument("--no_vlm_stubs", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("--sections", required=True)
    p_search.add_argument("--units", required=True)
    p_search.add_argument("--field_cards", default=None)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--topk", type=int, default=10)
    p_search.add_argument("--unit_type", default=None)
    p_search.add_argument("--object_type", default=None, choices=[None, "section", "unit", "field_card"])
    p_search.add_argument("--pack", action="store_true", help="return HiKEY-style evidence pack instead of flat hits")

    p_render = sub.add_parser("render-pages")
    p_render.add_argument("pdf")
    p_render.add_argument("--out_dir", required=True)
    p_render.add_argument("--pages", required=True, help="comma-separated page numbers, e.g. 13,14,132")
    p_render.add_argument("--zoom", type=float, default=2.0)

    p_vlm = sub.add_parser("prepare-vlm")
    p_vlm.add_argument("pdf")
    p_vlm.add_argument("--units", required=True)
    p_vlm.add_argument("--out_dir", required=True)
    p_vlm.add_argument("--pages", default=None, help="optional comma-separated page numbers")

    args = ap.parse_args()
    if args.cmd == "parse":
        parser = HiKEYFinanceParser(args.pdf, args.out_dir)
        paths = parser.parse(with_tables=not args.no_tables, add_vlm_stubs=not args.no_vlm_stubs)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
    elif args.cmd == "search":
        retriever = SimpleRetriever(args.sections, args.units, args.field_cards)
        if args.pack:
            res = retriever.pack_evidence(args.query, topk=args.topk)
        else:
            res = retriever.search(args.query, topk=args.topk, unit_type=args.unit_type, object_type=args.object_type)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "render-pages":
        pages = [int(x) for x in args.pages.split(",") if x.strip()]
        res = render_pages(args.pdf, args.out_dir, pages, zoom=args.zoom)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "prepare-vlm":
        pages = [int(x) for x in args.pages.split(",") if x.strip()] if args.pages else None
        res = prepare_vlm_manifest(args.pdf, args.units, args.out_dir, top_pages=pages)
        print(json.dumps({"vlm_manifest": res}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
