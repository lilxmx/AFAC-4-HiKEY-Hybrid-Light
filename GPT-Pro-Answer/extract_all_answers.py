"""
Extract all GPT-Pro answers from different domains into a unified JSON file,
including all relevant attributes (question, options, answer_format, type, doc_ids, reason, etc.)

Domains:
- financial_contracts: parsed.json + raw/ text files
- financial_reports: individual JSON files (1.json ~ 20.json)
- insurance: insurance_answers.json (answers array with full attributes)
- regulatory: regulatory_answers.json (answers array with full attributes)
- research: research_answers.json (answers array with full attributes)

Output: gpt_pro_all_answers.json
"""
import json
from pathlib import Path


def main():
    base_dir = Path(__file__).parent
    all_answers = {}

    # 1. financial_contracts - from parsed.json (limited attributes)
    # Also load question file for question/options/answer_format/type/doc_ids
    fc_questions = {}
    fc_qfile = base_dir.parent / "public_dataset_upload" / "questions" / "group_a" / "financial_contracts_questions.json"
    if fc_qfile.exists():
        with open(fc_qfile, 'r', encoding='utf-8') as f:
            for q in json.load(f):
                fc_questions[q["qid"]] = q

    fc_path = base_dir / "financial_contracts" / "parsed.json"
    if fc_path.exists():
        with open(fc_path, 'r', encoding='utf-8') as f:
            fc_data = json.load(f)
        for qid, info in fc_data.items():
            entry = {
                "qid": qid,
                "domain": "financial_contracts",
                "answer": info["answer"],
                "source": info.get("source", "gpt_pro"),
            }
            # Merge question attributes if available
            if qid in fc_questions:
                q = fc_questions[qid]
                entry["question"] = q.get("question", "")
                entry["options"] = q.get("options", {})
                entry["answer_format"] = q.get("answer_format", "")
                entry["question_type"] = q.get("type", "")
                entry["doc_ids"] = q.get("doc_ids", [])
            all_answers[qid] = entry
        print(f"[financial_contracts] Loaded {len(fc_data)} answers")

    # 2. financial_reports - from individual JSON files (1.json ~ 20.json)
    # Also load question file for question/options/answer_format/type/doc_ids
    fr_questions = {}
    fr_qfile = base_dir.parent / "public_dataset_upload" / "questions" / "group_a" / "financial_reports_questions.json"
    if fr_qfile.exists():
        with open(fr_qfile, 'r', encoding='utf-8') as f:
            for q in json.load(f):
                fr_questions[q["qid"]] = q

    fr_dir = base_dir / "financial_reports"
    fr_count = 0
    for i in range(1, 21):
        fr_path = fr_dir / f"{i}.json"
        if fr_path.exists():
            with open(fr_path, 'r', encoding='utf-8') as f:
                fr_data = json.load(f)
            qid = fr_data.get("qid", f"fin_a_{i:03d}")
            entry = {
                "qid": qid,
                "domain": "financial_reports",
                "answer": fr_data.get("answer", ""),
                "source": "gpt_pro",
                "evidence_retrieval": fr_data.get("evidence_retrieval", []),
            }
            # Merge question attributes if available
            if qid in fr_questions:
                q = fr_questions[qid]
                entry["question"] = q.get("question", "")
                entry["options"] = q.get("options", {})
                entry["answer_format"] = q.get("answer_format", "")
                entry["question_type"] = q.get("type", "")
                entry["doc_ids"] = q.get("doc_ids", [])
            all_answers[qid] = entry
            fr_count += 1
    print(f"[financial_reports] Loaded {fr_count} answers")

    # 3. insurance - from insurance_answers.json
    ins_path = base_dir / "insurance" / "insurance_answers.json"
    if ins_path.exists():
        with open(ins_path, 'r', encoding='utf-8') as f:
            ins_data = json.load(f)
        answers_list = ins_data.get("answers", [])
        for item in answers_list:
            qid = item["qid"]
            entry = {
                "qid": qid,
                "domain": item.get("domain", "insurance"),
                "answer": item["answer"],
                "source": item.get("source", "gpt_pro"),
                "question": item.get("question", ""),
                "options": item.get("options", {}),
                "answer_format": item.get("answer_format", ""),
                "question_type": item.get("question_type", ""),
                "doc_ids": item.get("doc_ids", []),
                "reason": item.get("reason", ""),
                "retrieval_summary": item.get("retrieval_summary", ""),
            }
            all_answers[qid] = entry
        print(f"[insurance] Loaded {len(answers_list)} answers")

    # 4. regulatory - from regulatory_answers.json
    reg_path = base_dir / "regulatory" / "regulatory_answers.json"
    if reg_path.exists():
        with open(reg_path, 'r', encoding='utf-8') as f:
            reg_data = json.load(f)
        answers_list = reg_data.get("answers", [])
        for item in answers_list:
            qid = item["qid"]
            entry = {
                "qid": qid,
                "domain": item.get("domain", "regulatory"),
                "answer": item["answer"],
                "source": item.get("source", "gpt_pro"),
                "question": item.get("question", ""),
                "options": item.get("options", {}),
                "answer_format": item.get("answer_format", ""),
                "question_type": item.get("question_type", ""),
                "doc_ids": item.get("doc_ids", []),
                "reason": item.get("reason", ""),
                "retrieval_summary": item.get("retrieval_summary", ""),
            }
            all_answers[qid] = entry
        print(f"[regulatory] Loaded {len(answers_list)} answers")

    # 5. research - from research_answers.json
    res_path = base_dir / "research" / "research_answers.json"
    if res_path.exists():
        with open(res_path, 'r', encoding='utf-8') as f:
            res_data = json.load(f)
        answers_list = res_data.get("answers", [])
        for item in answers_list:
            qid = item["qid"]
            entry = {
                "qid": qid,
                "domain": item.get("domain", "research"),
                "answer": item["answer"],
                "source": item.get("source", "gpt_pro"),
                "question": item.get("question", ""),
                "options": item.get("options", {}),
                "answer_format": item.get("answer_format", ""),
                "question_type": item.get("question_type", ""),
                "doc_ids": item.get("doc_ids", []),
                "reason": item.get("reason", ""),
                "retrieval_summary": item.get("retrieval_summary", ""),
            }
            all_answers[qid] = entry
        print(f"[research] Loaded {len(answers_list)} answers")

    # Output unified JSON
    output_path = base_dir / "gpt_pro_all_answers.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)

    print(f"\n=== Total: {len(all_answers)} answers ===")
    print(f"Output: {output_path}")

    # Print summary of answer_format distribution
    format_counts = {}
    for entry in all_answers.values():
        fmt = entry.get("answer_format", "unknown")
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    print(f"\nAnswer format distribution: {format_counts}")


if __name__ == '__main__':
    main()
