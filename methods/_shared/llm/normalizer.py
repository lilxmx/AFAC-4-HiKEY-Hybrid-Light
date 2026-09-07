"""
Answer normalizer: extract A/B/C/D letters from raw LLM output and shape them
into a canonical answer string.

Public API:
    normalize_answer(raw_answer: str, answer_format: str) -> str
        answer_format: 'mcq' | 'multi' | 'tf'

Multi-strategy extraction (in priority order):
    1. Last few short lines that contain only A-D letters.
    2. Anchored answer patterns ("最终答案：A", "答案是 ACD"...).
    3. Pure-letter compression of the whole tail when message is short.
    4. All A-D letters from the last 5 lines as a last resort.

Always returns a non-empty answer; falls back to 'A' on total failure.
"""
import re
from typing import List


VALID_LETTERS = set("ABCD")


def normalize_answer(raw_answer: str, answer_format: str) -> str:
    """Extract a canonical answer letter string."""
    if not raw_answer or raw_answer.startswith("ERROR"):
        return _fallback(answer_format)

    # ---- Strategy 1: short trailing lines that look like an answer line ----
    lines = raw_answer.strip().split("\n")
    for back in range(min(3, len(lines))):
        line = lines[-(back + 1)].strip()
        if not line:
            continue
        letters = re.findall(r"[A-D]", line.upper())
        if letters and len(line) < 15:
            return _shape(letters, answer_format)

    # ---- Strategy 2: anchored answer patterns ----
    answer_patterns = [
        r"最终答案[是为：:]\s*([A-D]+)",
        r"答案[是为：:]\s*([A-D]+)",
        r"正确答案[是为：:]\s*([A-D]+)",
        r"选择?\s*([A-D]+)\s*$",
        r"综上[，,]?\s*(?:选择?|答案[是为]?)\s*([A-D]+)",
    ]
    for pattern in answer_patterns:
        m = re.search(pattern, raw_answer.upper(), re.MULTILINE)
        if m:
            letters = list(m.group(1))
            if letters:
                return _shape(letters, answer_format)

    # ---- Strategy 3: short whole-message compression ----
    text = raw_answer.strip()
    if len(text) < 20:
        clean = re.sub(r"[^A-D]", "", text.upper())
        if clean:
            return _shape(list(clean), answer_format)

    # ---- Strategy 4: fall back to last 5 lines ----
    last_part = "\n".join(lines[-5:]) if len(lines) >= 5 else raw_answer
    letters = re.findall(r"[A-D]", last_part.upper())
    if letters:
        # For mcq/tf prefer the LAST letter (likely the final answer line).
        if answer_format == "mcq":
            return letters[-1]
        if answer_format == "tf":
            valid = [c for c in letters if c in ("A", "B")]
            return valid[-1] if valid else "A"
        return "".join(sorted(set(letters)))

    return _fallback(answer_format)


def extract_letters(text: str) -> List[str]:
    """Helper for callers that just want the raw letter list (used by some agents)."""
    if not text:
        return []
    text = text.strip()
    # Whole-text compression for very short responses
    if len(text) < 20:
        clean = re.sub(r"[^A-D]", "", text.upper())
        if clean:
            return list(clean)
    # Anchored patterns
    for pattern in (
        r"答案[是为：:]\s*([A-D]+)",
        r"选择?\s*([A-D]+)",
        r"正确[的答案选项]*[是为：:]\s*([A-D]+)",
        r"^([A-D]+)\s*$",
        r"([A-D](?:[、,，\s]+[A-D])*)",
    ):
        m = re.search(pattern, text.upper())
        if m:
            found = re.sub(r"[^A-D]", "", m.group(1))
            if found:
                return list(found)
    standalone = re.findall(r"\b([A-D])\b", text.upper())
    if standalone:
        return standalone
    return re.findall(r"[A-D]", text.upper())


def _shape(letters: List[str], answer_format: str) -> str:
    if not letters:
        return _fallback(answer_format)
    if answer_format == "mcq":
        return letters[0]
    if answer_format == "tf":
        valid = [c for c in letters if c in ("A", "B")]
        return valid[0] if valid else "A"
    if answer_format == "multi":
        return "".join(sorted(set(letters)))
    return letters[0]


def _fallback(answer_format: str) -> str:
    return "A"
