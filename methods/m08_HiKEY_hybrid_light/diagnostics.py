"""Local retrieval diagnostics for m08.

This keeps the m07-style weak recall accounting, and also records which top-k
evidence items appear to support the gold option(s).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


VALID = set("ABCD")

DOMAIN_TERMS = {
    "insurance": [
        "保险责任",
        "责任免除",
        "等待期",
        "犹豫期",
        "给付",
        "退保",
        "现金价值",
        "受益",
        "合同解除",
        "身故",
        "疾病",
        "赔付",
        "免责",
    ],
    "regulatory": [
        "总则",
        "适用范围",
        "应当",
        "可以",
        "不得",
        "例外",
        "处罚",
        "报告",
        "核实",
        "差异",
        "存量",
        "施行",
        "期限",
        "监管",
        "管理",
        "义务",
    ],
    "financial_contracts": [
        "发行",
        "募集资金",
        "评级",
        "回售",
        "赎回",
        "违约",
        "担保",
        "期限",
        "票面",
        "到期",
        "本金",
        "利息",
        "条款",
    ],
    "financial_reports": [
        "营业收入",
        "归母净利润",
        "扣非",
        "净利润",
        "现金流",
        "研发投入",
        "研发费用",
        "分红",
        "每股",
        "同比",
        "环比",
        "资产负债率",
    ],
    "research": [
        "行业",
        "公司",
        "同比",
        "环比",
        "CAGR",
        "预测",
        "假设",
        "结论",
        "图",
        "表",
        "趋势",
        "增长",
        "下降",
        "市场份额",
    ],
}

STOP_TERMS = {
    "以下",
    "下列",
    "关于",
    "描述",
    "正确",
    "错误",
    "文件",
    "第一",
    "第二",
    "其中",
    "基于",
    "根据",
    "提供",
    "的是",
    "是否",
    "均为",
    "均是",
    "哪些",
}


def normalize_answer(answer: str, answer_format: str = "multi") -> str:
    letters = [c for c in (answer or "").upper() if c in VALID]
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    return "".join(sorted(set(letters)))


def load_reference_answers(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("qid"):
                result[item["qid"]] = item
    return result


def compact(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_proxy_terms(text: str, domain: str) -> List[str]:
    raw = str(text or "")
    terms: List[str] = []

    terms.extend(re.findall(r"20\d{2}", raw))
    terms.extend(re.findall(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*(?:年|个月|亿元|万元|倍|期|天)", raw))
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9.-]{1,}", raw))

    hay = compact(raw)
    for term in DOMAIN_TERMS.get(domain, []):
        if compact(term) in hay:
            terms.append(term)

    cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", raw)
    for span in cjk_spans:
        span = compact(span)
        if not span:
            continue
        if len(span) <= 8 and span not in STOP_TERMS:
            terms.append(span)
        else:
            for n in (4, 3):
                for i in range(0, max(0, len(span) - n + 1)):
                    gram = span[i : i + n]
                    if gram not in STOP_TERMS:
                        terms.append(gram)

    return _unique(terms)[:30]


def option_proxy_coverage(option_text: str, evidence_text: str, domain: str, threshold: float) -> Dict[str, Any]:
    terms = extract_proxy_terms(option_text, domain)
    evidence = compact(evidence_text)
    matched = [t for t in terms if compact(t) and compact(t) in evidence]
    missing = [t for t in terms if t not in matched]
    coverage = (len(matched) / len(terms)) if terms else 0.0

    numeric_terms = [t for t in terms if re.search(r"\d", t)]
    numeric_matched = [t for t in numeric_terms if t in matched]
    numeric_ok = (len(numeric_matched) == len(numeric_terms)) if numeric_terms else True
    hit = coverage >= threshold and numeric_ok

    return {
        "hit": bool(hit),
        "coverage": round(coverage, 4),
        "term_count": len(terms),
        "matched_terms": matched,
        "missing_terms": missing[:20],
        "numeric_terms": numeric_terms,
        "numeric_matched": numeric_matched,
    }


def _describe_item(item: Dict[str, Any]) -> str:
    anchor = item.get("anchor", {}) or {}
    ancestry = item.get("ancestry", {}) or {}
    siblings = item.get("siblings", []) or []
    parts: List[str] = []

    for key in ("section_path", "title", "text", "row_header", "metric", "raw_metric", "unit", "table_name"):
        value = anchor.get(key)
        if value:
            parts.append(str(value))

    for key in ("section_path", "title"):
        value = ancestry.get(key)
        if value:
            parts.append(str(value))

    for sibling in siblings[:3]:
        for key in ("text", "row_header", "metric", "table_name"):
            value = sibling.get(key)
            if value:
                parts.append(str(value))

    if anchor.get("value_map"):
        parts.append(json.dumps(anchor.get("value_map"), ensure_ascii=False))

    return " ".join(parts)


def build_diagnostics(
    q: Dict[str, Any],
    pred_answer: str,
    retrieval_meta: Dict[str, Any],
    reference_answers: Dict[str, Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    qid = q["qid"]
    answer_format = q.get("answer_format", "mcq")
    domain = q.get("domain", "")
    ref = reference_answers.get(qid, {}) if reference_answers else {}
    gold_answer = normalize_answer(ref.get("answer", ""), answer_format) if ref else ""
    pred_norm = normalize_answer(pred_answer, answer_format)
    answer_exact_match = bool(gold_answer and pred_norm == gold_answer)

    options = q.get("options", {}) or {}
    merged_items = retrieval_meta.get("merged_evidence_pack", []) or []
    topk_items = list(merged_items)
    evidence_blob = " ".join(_describe_item(item) for item in topk_items)

    option_proxy: Dict[str, Any] = {}
    for key in sorted(options.keys()):
        option_proxy[key] = option_proxy_coverage(str(options[key]), evidence_blob, domain, threshold)
        option_proxy[key]["is_gold"] = key in set(gold_answer)

    gold_options = [k for k in sorted(options.keys()) if k in set(gold_answer)]
    supported_gold = [k for k in gold_options if option_proxy.get(k, {}).get("hit")]
    missing_gold = [k for k in gold_options if k not in supported_gold]

    topk_hit_items: List[Dict[str, Any]] = []
    supported_by_item: set[str] = set()
    for idx, item in enumerate(topk_items, 1):
        item_text = _describe_item(item)
        gold_match: Dict[str, Any] = {}
        item_supported: List[str] = []
        for letter in gold_options:
            opt_text = str(options.get(letter, ""))
            cov = option_proxy_coverage(opt_text, item_text, domain, threshold)
            gold_match[letter] = cov
            if cov.get("hit"):
                item_supported.append(letter)
                supported_by_item.add(letter)
        topk_hit_items.append(
            {
                "rank": idx,
                "source_doc_id": item.get("source_doc_id") or item.get("anchor", {}).get("doc_id", ""),
                "query_labels": item.get("_query_labels", []),
                "matched_gold_options": item_supported,
                "gold_option_proxy": gold_match,
                "hit": bool(item_supported),
                "item_summary": item_text[:600],
            }
        )

    hit_items = [item for item in topk_hit_items if item["hit"]]

    return {
        "reference_available": bool(gold_answer),
        "gold_answer": gold_answer,
        "pred_answer": pred_norm,
        "answer_exact_match": answer_exact_match,
        "gold_any_option_proxy_hit": bool(supported_gold),
        "gold_all_options_proxy_hit": bool(gold_options and len(supported_gold) == len(gold_options)),
        "supported_gold_options": supported_gold,
        "missing_gold_options": missing_gold,
        "topk_hit_count": len(hit_items),
        "topk_hit_indices": [item["rank"] for item in hit_items],
        "topk_hit_items": hit_items,
        "topk_items": topk_items,
        "option_proxy": option_proxy,
        "supported_gold_options_by_item": sorted(supported_by_item),
    }


def summarize_diagnostics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    exact = 0
    any_hit = 0
    all_hit = 0
    hit_items_total = 0.0
    by_domain: Dict[str, Dict[str, float]] = {}

    for qid, result in results.items():
        diag = result.get("retrieval_diagnostics") or {}
        if not diag.get("reference_available"):
            continue
        total += 1
        domain = result.get("domain", "unknown")
        s = by_domain.setdefault(
            domain,
            {"total": 0, "answer_exact": 0, "gold_any_proxy": 0, "gold_all_proxy": 0, "hit_items": 0.0},
        )
        s["total"] += 1
        hit_items = float(diag.get("topk_hit_count", 0) or 0)
        hit_items_total += hit_items
        s["hit_items"] += hit_items
        if diag.get("answer_exact_match"):
            exact += 1
            s["answer_exact"] += 1
        if diag.get("gold_any_option_proxy_hit"):
            any_hit += 1
            s["gold_any_proxy"] += 1
        if diag.get("gold_all_options_proxy_hit"):
            all_hit += 1
            s["gold_all_proxy"] += 1

    def ratio(n: float, d: float) -> float:
        return round(n / d, 4) if d else 0.0

    domain_summary = {}
    for domain, s in by_domain.items():
        domain_summary[domain] = {
            **s,
            "answer_exact_rate": ratio(s["answer_exact"], s["total"]),
            "gold_any_proxy_rate": ratio(s["gold_any_proxy"], s["total"]),
            "gold_all_proxy_rate": ratio(s["gold_all_proxy"], s["total"]),
            "avg_hit_items_per_question": ratio(s["hit_items"], s["total"]),
        }

    return {
        "total_with_reference": total,
        "answer_exact": exact,
        "gold_any_proxy": any_hit,
        "gold_all_proxy": all_hit,
        "answer_exact_rate": ratio(exact, total),
        "gold_any_proxy_rate": ratio(any_hit, total),
        "gold_all_proxy_rate": ratio(all_hit, total),
        "avg_hit_items_per_question": ratio(hit_items_total, total),
        "by_domain": domain_summary,
    }
