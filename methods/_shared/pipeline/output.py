"""
Shared output writers:
- answer.csv          (submission format)
- evidence.json       (per-question evidence / reasoning)
- run_summary.json    (token stats, per-domain breakdown, score estimates)
"""
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from ..config_base import (
    QID_PREFIX_TO_DOMAIN,
    compute_token_score,
    estimate_final_multiplier,
    infer_domain_from_qid,
)


def save_answer_csv(
    path: Path,
    results: Dict[str, Dict],
    token_stats: Dict[str, int],
    delimiter: str = ",",
) -> None:
    """
    Write the submission-format answer.csv.

    Columns: qid, answer, prompt_tokens, completion_tokens, total_tokens
    First data row is the 'summary' aggregate row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow([
            "summary", "",
            token_stats.get("total_prompt", 0),
            token_stats.get("total_completion", 0),
            token_stats.get("total_tokens", 0),
        ])
        for qid in sorted(results.keys()):
            r = results[qid]
            writer.writerow([
                qid,
                r.get("answer", ""),
                r.get("prompt_tokens", 0),
                r.get("completion_tokens", 0),
                r.get("total_tokens", 0),
            ])


def save_evidence_json(path: Path, results: Dict[str, Dict]) -> None:
    """Save per-question evidence + raw reasoning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for qid, r in results.items():
        payload[qid] = {
            "answer": r.get("answer", ""),
            "evidence": r.get("evidence", []),
            "raw_answer": r.get("raw_answer", "") or r.get("reasoning", ""),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compute_domain_stats(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Aggregate token usage per domain (inferred from qid prefix)."""
    stats: Dict[str, Dict] = {}
    for qid, r in results.items():
        domain = r.get("domain") or infer_domain_from_qid(qid)
        s = stats.setdefault(domain, {"count": 0, "total_tokens": 0})
        s["count"] += 1
        s["total_tokens"] += int(r.get("total_tokens", 0) or 0)

    for s in stats.values():
        s["avg_tokens"] = (s["total_tokens"] // s["count"]) if s["count"] else 0
    return stats


def save_run_summary(
    path: Path,
    method_name: str,
    model_name: str,
    results: Dict[str, Dict],
    token_stats: Dict[str, int],
    extra: Optional[Dict] = None,
) -> Dict:
    """Save run_summary.json and return the dict that was saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = int(token_stats.get("total_tokens", 0))
    summary = {
        "method": method_name,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(results),
        "token_stats": dict(token_stats),
        "token_score": compute_token_score(total_tokens),
        "estimated_multiplier": estimate_final_multiplier(total_tokens),
        "avg_tokens_per_question": (total_tokens / len(results)) if results else 0,
        "domain_stats": compute_domain_stats(results),
    }
    if extra:
        summary.update(extra)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
