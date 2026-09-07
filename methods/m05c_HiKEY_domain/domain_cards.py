"""Lightweight domain-specific cards built from existing HiKEY cache.

This module deliberately uses deterministic rules first. The goal is to add
retrieval handles that match each financial domain's recurring question types
without increasing prompt size.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]+|20\d{2}|\d+(?:\.\d+)?%?")


def tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def _text_of(obj: Dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(k, "") or "")
        for k in ("section_path", "metric", "raw_metric", "text", "row_header", "table_name", "unit")
    )


def _contains_any(text: str, words: Sequence[str]) -> bool:
    return any(w in text for w in words)


@dataclass
class DomainCard:
    card_id: str
    domain: str
    card_type: str
    doc_id: str
    text: str
    source: Dict[str, Any]
    structured: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_prompt_block(self, evidence_id: str) -> str:
        fields = [
            f"[{evidence_id}] DomainCard/{self.card_type}",
            f"文档: {self.doc_id}",
        ]
        page = self.source.get("source_page") or self.source.get("page")
        if page:
            fields.append(f"页码: {page}")
        section = self.source.get("section_path") or self.structured.get("section_path")
        if section:
            fields.append(f"章节: {section}")
        if self.structured:
            fields.append(f"结构化字段: {json.dumps(self.structured, ensure_ascii=False)[:900]}")
        fields.append(f"内容: {self.text[:900]}")
        return "\n".join(fields)


class DomainCardIndex:
    def __init__(self) -> None:
        self.cards: List[DomainCard] = []
        self._card_tokens: List[set[str]] = []

    def add_cards(self, cards: Iterable[DomainCard]) -> None:
        for card in cards:
            self.cards.append(card)
            self._card_tokens.append(set(tokens(card.text + " " + json.dumps(card.structured, ensure_ascii=False))))

    def search(
        self,
        query: str,
        domain: str,
        topk: int,
        target_doc_ids: Sequence[str] | None = None,
    ) -> List[DomainCard]:
        q_tokens = set(tokens(query))
        q_text = query or ""
        target = set(target_doc_ids or [])
        scored: List[DomainCard] = []
        for card, toks in zip(self.cards, self._card_tokens):
            if card.domain != domain:
                continue
            if target and card.doc_id not in target:
                continue
            overlap = len(q_tokens & toks)
            if overlap <= 0:
                continue
            text = card.text + " " + json.dumps(card.structured, ensure_ascii=False)
            score = float(overlap)
            for word in q_tokens:
                if len(word) >= 3 and word in text.lower():
                    score += 0.35
            for year in re.findall(r"20\d{2}", q_text):
                if year in text:
                    score += 1.0
            for number in re.findall(r"\d+(?:\.\d+)?%?", q_text):
                if number in text:
                    score += 0.8
            if card.card_type in _preferred_card_types(domain, q_text):
                score += 2.0
            item = DomainCard(
                card_id=card.card_id,
                domain=card.domain,
                card_type=card.card_type,
                doc_id=card.doc_id,
                text=card.text,
                source=card.source,
                structured=card.structured,
                score=round(score, 4),
            )
            scored.append(item)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:topk]


def build_domain_cards(domain: str, doc_id: str, retriever: Any) -> List[DomainCard]:
    cards: List[DomainCard] = []
    field_cards = getattr(retriever, "field_cards", []) or []
    units = getattr(retriever, "units", []) or []

    for i, fc in enumerate(field_cards):
        cards.extend(_cards_from_field(domain, doc_id, i, fc))

    for i, unit in enumerate(units):
        text = _text_of(unit)
        if not text.strip():
            continue
        cards.extend(_cards_from_unit(domain, doc_id, i, unit, text))

    # Avoid prompt/retrieval bloat from exact duplicate snippets.
    dedup: Dict[str, DomainCard] = {}
    for card in cards:
        key = f"{card.domain}|{card.doc_id}|{card.card_type}|{card.text[:240]}"
        dedup.setdefault(key, card)
    return list(dedup.values())


def _cards_from_field(domain: str, doc_id: str, idx: int, fc: Dict[str, Any]) -> List[DomainCard]:
    text = _text_of(fc) + " " + json.dumps(fc.get("value_map", {}), ensure_ascii=False)
    metric = str(fc.get("metric") or fc.get("raw_metric") or "")
    structured = {
        "metric": metric,
        "unit": fc.get("unit"),
        "value_map": fc.get("value_map", {}),
        "section_path": fc.get("section_path"),
        "table_name": fc.get("table_name"),
    }
    card_type = {
        "financial_reports": "MetricYearCard",
        "financial_contracts": "BondTermCard",
        "research": "ForecastMetricCard",
    }.get(domain, "FieldFactCard")
    if domain == "financial_reports" and _contains_any(text, ["分红", "派息", "现金红利", "每10股", "每股"]):
        card_type = "DividendCard"
    if domain == "research" and _contains_any(text, ["市场规模", "CAGR", "复合增速", "增速", "预测", "预计"]):
        card_type = "MarketSizeCard"
    return [
        DomainCard(
            card_id=f"{domain}:{doc_id}:field:{idx}",
            domain=domain,
            card_type=card_type,
            doc_id=doc_id,
            text=text,
            source=fc,
            structured=structured,
        )
    ]


def _cards_from_unit(domain: str, doc_id: str, idx: int, unit: Dict[str, Any], text: str) -> List[DomainCard]:
    cards: List[DomainCard] = []
    for card_type, words in _domain_patterns(domain).items():
        if _contains_any(text, words):
            structured = _extract_structured_fields(domain, card_type, text, unit)
            cards.append(
                DomainCard(
                    card_id=f"{domain}:{doc_id}:unit:{idx}:{card_type}",
                    domain=domain,
                    card_type=card_type,
                    doc_id=doc_id,
                    text=text,
                    source=unit,
                    structured=structured,
                )
            )
    return cards


def _domain_patterns(domain: str) -> Dict[str, Sequence[str]]:
    return {
        "financial_reports": {
            "MetricYearCard": ["营业收入", "净利润", "现金流量", "研发投入", "资产负债率", "同比"],
            "DividendCard": ["分红", "派息", "现金红利", "每10股", "每股"],
            "CompareCard": ["增长", "下降", "增加", "减少", "同比", "较上年"],
        },
        "financial_contracts": {
            "BondTermCard": ["发行人", "发行规模", "发行金额", "期限", "票面利率", "债券评级", "主体评级"],
            "PartyRoleCard": ["主承销商", "受托管理人", "评级机构", "簿记管理人", "律师事务所"],
            "ClauseCard": ["信息披露", "违约", "赎回", "回售", "担保", "偿债保障"],
        },
        "insurance": {
            "BenefitRuleCard": ["保险金", "保险责任", "给付", "赔付", "身故", "重大疾病", "医疗"],
            "ExclusionCard": ["责任免除", "不承担", "不予赔付", "除外", "免责"],
            "SurrenderRuleCard": ["退保", "现金价值", "账户价值", "退保费用", "犹豫期"],
            "ClaimCalcCard": ["免赔额", "赔付比例", "报销", "已交保费", "基本保险金额"],
        },
        "regulatory": {
            "ArticleCard": ["第", "条", "办法", "规定", "条例", "施行"],
            "ObligationCard": ["应当", "不得", "可以", "金融机构", "监管", "提交", "保存"],
            "DeadlineCard": ["日内", "工作日", "年内", "施行", "期限", "保存"],
            "ExceptionCard": ["除外", "但是", "但", "另有规定", "不适用"],
        },
        "research": {
            "ForecastMetricCard": ["预计", "预测", "有望", "目标价", "市场规模", "增速"],
            "MarketSizeCard": ["市场规模", "CAGR", "复合增速", "渗透率", "亿元", "亿美元"],
            "CompanyCompareCard": ["同比", "营收", "净利润", "公司", "行业", "对比"],
        },
    }.get(domain, {})


def _extract_structured_fields(domain: str, card_type: str, text: str, unit: Dict[str, Any]) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "section_path": unit.get("section_path"),
        "unit_type": unit.get("unit_type"),
    }
    years = re.findall(r"20\d{2}", text)
    nums = re.findall(r"\d+(?:\.\d+)?%?|\d+(?:,\d{3})+(?:\.\d+)?", text)
    if years:
        data["years"] = sorted(set(years))
    if nums:
        data["numbers"] = nums[:12]
    if domain == "regulatory":
        for modal in ("应当", "不得", "可以"):
            if modal in text:
                data["modal"] = modal
                break
        m = re.search(r"(\d+\s*个?工作日|\d+\s*日|\d+\s*年)内?", text)
        if m:
            data["deadline"] = m.group(1).replace(" ", "")
    if domain == "insurance":
        m = re.search(r"(免赔额|赔付比例|退保费用|现金价值|账户价值)[为是：:]?([^，。；;]{1,30})", text)
        if m:
            data["rule_key"] = m.group(1)
            data["rule_value"] = m.group(2)
    return data


def _preferred_card_types(domain: str, query: str) -> set[str]:
    prefs: set[str] = set()
    if domain == "regulatory":
        if _contains_any(query, ["日内", "工作日", "期限", "施行"]):
            prefs.add("DeadlineCard")
        if _contains_any(query, ["应当", "不得", "可以", "义务"]):
            prefs.add("ObligationCard")
    elif domain == "insurance":
        if _contains_any(query, ["赔", "免赔", "报销", "给付"]):
            prefs.update({"BenefitRuleCard", "ClaimCalcCard"})
        if _contains_any(query, ["退保", "现金价值", "账户价值"]):
            prefs.add("SurrenderRuleCard")
    elif domain == "financial_reports":
        if _contains_any(query, ["分红", "派息", "每10股"]):
            prefs.add("DividendCard")
        if _contains_any(query, ["同比", "增长", "下降"]):
            prefs.add("CompareCard")
    elif domain == "financial_contracts":
        if _contains_any(query, ["承销商", "受托管理人", "评级机构"]):
            prefs.add("PartyRoleCard")
        if _contains_any(query, ["违约", "赎回", "回售", "担保"]):
            prefs.add("ClauseCard")
    elif domain == "research":
        if _contains_any(query, ["市场规模", "预测", "预计", "增速"]):
            prefs.update({"ForecastMetricCard", "MarketSizeCard"})
    return prefs

