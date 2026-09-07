"""
Universal answer comparison utility.

Supports loading answers from:
- CSV (answer.csv format: qid,answer)
- JSON (golden_answers.json format: {qid: {answer: ...}})
- Parsed reference JSON (references/gpt_pro/fin-c/parsed.json)

Usage:
    python -m methods._shared.eval.compare \
        --pred runs/lgr/m01_baseline_qwen/legacy_v1/output/answer.csv \
        --ref references/gpt_pro/fin-c/parsed.json \
        --domain fc
"""
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple


def load_answers_from_csv(path: str) -> Dict[str, str]:
    """Load answers from CSV file (qid,answer format)."""
    answers = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get('qid', row.get('question_id', ''))
            answer = row.get('answer', '')
            if qid:
                answers[qid] = answer.strip()
    return answers


def load_answers_from_json(path: str) -> Dict[str, str]:
    """Load answers from JSON file. Supports multiple formats."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    answers = {}
    if isinstance(data, dict):
        for qid, val in data.items():
            if isinstance(val, dict):
                answers[qid] = val.get('answer', val.get('final_answer', ''))
            elif isinstance(val, str):
                answers[qid] = val
    elif isinstance(data, list):
        for item in data:
            qid = item.get('qid', item.get('question_id', ''))
            answer = item.get('answer', item.get('final_answer', ''))
            if qid:
                answers[qid] = answer
    
    return answers


def load_answers(path: str) -> Dict[str, str]:
    """Auto-detect format and load answers."""
    p = Path(path)
    if p.suffix == '.csv':
        return load_answers_from_csv(path)
    elif p.suffix == '.json':
        return load_answers_from_json(path)
    else:
        # Try JSON first, then CSV
        try:
            return load_answers_from_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return load_answers_from_csv(path)


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    # Remove separators, sort letters
    letters = sorted(set(c for c in answer.upper() if c in 'ABCD'))
    return ''.join(letters)


def compare_answers(
    pred: Dict[str, str],
    ref: Dict[str, str],
    domain_filter: str = ""
) -> Tuple[Dict, Dict]:
    """
    Compare predicted answers against reference.
    
    Returns:
        (summary_dict, details_dict)
    """
    # Filter by domain prefix if specified
    domain_prefixes = {
        'fc': 'fc_a_',
        'fr': 'fr_a_',
        'ins': 'ins_a_',
        'reg': 'reg_a_',
        'res': 'res_a_',
    }
    
    prefix = domain_prefixes.get(domain_filter, domain_filter) if domain_filter else ""
    
    # Find common questions
    common_qids = sorted(set(pred.keys()) & set(ref.keys()))
    if prefix:
        common_qids = [q for q in common_qids if q.startswith(prefix)]
    
    correct = 0
    total = len(common_qids)
    details = {}
    
    for qid in common_qids:
        pred_norm = normalize_answer(pred[qid])
        ref_norm = normalize_answer(ref[qid])
        is_correct = (pred_norm == ref_norm)
        if is_correct:
            correct += 1
        details[qid] = {
            "pred": pred[qid],
            "ref": ref[qid],
            "pred_norm": pred_norm,
            "ref_norm": ref_norm,
            "correct": is_correct,
        }
    
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "domain_filter": domain_filter or "all",
    }
    
    return summary, details


def main():
    parser = argparse.ArgumentParser(description="Compare answers between prediction and reference")
    parser.add_argument('--pred', required=True, help='Path to prediction file (csv or json)')
    parser.add_argument('--ref', required=True, help='Path to reference file (csv or json)')
    parser.add_argument('--domain', default='', help='Domain filter: fc/fr/ins/reg/res')
    parser.add_argument('--output', default='', help='Output comparison JSON path')
    args = parser.parse_args()
    
    pred = load_answers(args.pred)
    ref = load_answers(args.ref)
    
    summary, details = compare_answers(pred, ref, args.domain)
    
    print("=" * 50)
    print(f"  Answer Comparison Report")
    print("=" * 50)
    print(f"  Domain: {summary['domain_filter']}")
    print(f"  Total: {summary['total']}")
    print(f"  Correct: {summary['correct']}")
    print(f"  Accuracy: {summary['accuracy']:.1%}")
    print("=" * 50)
    print()
    
    # Print details
    for qid, d in details.items():
        status = "✓" if d['correct'] else "✗"
        print(f"  {status} {qid}: pred={d['pred_norm']}  ref={d['ref_norm']}")
    
    if args.output:
        result = {"summary": summary, "details": details}
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
