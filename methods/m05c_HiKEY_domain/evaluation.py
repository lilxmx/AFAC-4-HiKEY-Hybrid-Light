"""Evaluation helpers for m05c outputs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple


def normalize_answer(ans: Any) -> str:
    s = "" if ans is None else str(ans).strip().upper().replace(" ", "")
    letters = [c for c in s if c in "ABCD"]
    return "".join(sorted(set(letters)))


def load_ground_truth(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        qid: (value.get("answer", "") if isinstance(value, dict) else str(value))
        for qid, value in data.items()
    }


def write_comparison_files(output_dir: Path, results: Dict[str, Dict[str, Any]], gt_path: Path) -> Dict[str, Any]:
    gt = load_ground_truth(gt_path)
    details: Dict[str, Any] = {}
    domain_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "topk_answer_hit": 0})

    total = correct = topk_hits = 0
    for qid in sorted(results.keys()):
        r = results[qid]
        if qid not in gt:
            continue
        pred_norm = normalize_answer(r.get("answer", ""))
        gt_norm = normalize_answer(gt[qid])
        is_correct = pred_norm == gt_norm
        topk_hit = bool(r.get("topk_answer_hit", is_correct))
        domain = r.get("domain", "")
        total += 1
        correct += int(is_correct)
        topk_hits += int(topk_hit)
        domain_stats[domain]["total"] += 1
        domain_stats[domain]["correct"] += int(is_correct)
        domain_stats[domain]["topk_answer_hit"] += int(topk_hit)
        details[qid] = {
            "domain": domain,
            "prediction": r.get("answer", ""),
            "ground_truth": gt[qid],
            "prediction_norm": pred_norm,
            "ground_truth_norm": gt_norm,
            "answer_correct": is_correct,
            "topk_answer_hit": topk_hit,
            "final_evidence_count": r.get("retrieval_meta", {}).get("final_evidence_count", 0),
            "domain_card_count": r.get("retrieval_meta", {}).get("domain_card_count", 0),
            "topk_evidence_ids": r.get("retrieval_meta", {}).get("topk_evidence_ids", []),
        }

    domain_summary = {}
    for domain, s in sorted(domain_stats.items()):
        denom = s["total"] or 1
        domain_summary[domain] = {
            **s,
            "accuracy": s["correct"] / denom,
            "topk_answer_hit_rate": s["topk_answer_hit"] / denom,
        }

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "topk_answer_hits": topk_hits,
        "topk_answer_hit_rate": topk_hits / total if total else 0.0,
        "note": "topk_answer_hit is computed as final top-k evidence -> model answer equals ground truth, because the provided ground truth only contains answer letters, not gold evidence spans.",
        "domain_summary": domain_summary,
    }
    payload = {"summary": summary, "details": details}

    json_path = output_dir / "ground_truth_comparison.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "ground_truth_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "domain",
                "prediction",
                "ground_truth",
                "answer_correct",
                "topk_answer_hit",
                "final_evidence_count",
                "domain_card_count",
            ],
        )
        writer.writeheader()
        for qid, d in details.items():
            writer.writerow({
                "qid": qid,
                "domain": d["domain"],
                "prediction": d["prediction"],
                "ground_truth": d["ground_truth"],
                "answer_correct": d["answer_correct"],
                "topk_answer_hit": d["topk_answer_hit"],
                "final_evidence_count": d["final_evidence_count"],
                "domain_card_count": d["domain_card_count"],
            })
    return summary

