"""
Parse GPT-Pro raw markdown answers into structured JSON.

Reads: references/gpt_pro/fin-c/raw/1..20
Writes: references/gpt_pro/fin-c/parsed.json
"""
import re
import json
from pathlib import Path


def parse_answer_from_text(text: str) -> str:
    """Extract the final answer (letter(s)) from GPT-Pro markdown output."""
    # Comprehensive patterns ordered by reliability
    # All patterns allow optional bold markers and various separators
    reliable_patterns = [
        # "最终答案是：**A、B、D**" or "最终答案：**A、B、D**"
        r'最终答案[是为]?[：:]\s*\*{0,2}([A-D](?:[、,，\s]*[A-D])*)\*{0,2}',
        # "准确选项：**A、B、D**"
        r'准确选项[：:]\s*\*{0,2}([A-D](?:[、,，\s]*[A-D])*)\*{0,2}',
        # "正确项应为：**A、B**"
        r'正确项应为[：:]\s*\*{0,2}([A-D](?:[、,，\s]*[A-D])*)\*{0,2}',
    ]
    
    for pattern in reliable_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            raw = matches[-1].group(1)
            letters = re.findall(r'[A-D]', raw)
            return ''.join(sorted(set(letters)))
    
    # Fallback: "答案：X" or "答案为X"
    fallback_patterns = [
        r'答案[是为]?[：:]\s*\*{0,2}([A-D](?:[、,，\s]*[A-D])*)\*{0,2}',
    ]
    for pattern in fallback_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            raw = matches[-1].group(1)
            letters = re.findall(r'[A-D]', raw)
            return ''.join(sorted(set(letters)))
    
    return ""


def main():
    raw_dir = Path(__file__).parent / "fin-c" / "raw"
    output_path = Path(__file__).parent / "fin-c" / "parsed.json"
    
    results = {}
    
    for i in range(1, 21):
        file_path = raw_dir / str(i)
        if not file_path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue
        
        text = file_path.read_text(encoding='utf-8')
        answer = parse_answer_from_text(text)
        qid = f"fc_a_{i:03d}"
        results[qid] = {
            "answer": answer,
            "source": "gpt_pro",
            "raw_file": str(i),
        }
        print(f"  {qid}: {answer}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nParsed {len(results)} answers -> {output_path}")


if __name__ == '__main__':
    main()
