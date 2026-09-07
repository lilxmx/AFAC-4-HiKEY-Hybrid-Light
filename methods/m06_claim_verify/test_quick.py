#!/usr/bin/env python3
"""Quick test: run 1-2 questions through m06 to verify the pipeline works.

Usage:
    python methods/m06_claim_verify/test_quick.py
"""
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m06_test")

# Load config
from methods.m06_claim_verify import config as _cfg
from methods._shared.llm.clients import make_dashscope_client
from methods._shared.retrieval import build_retriever
from methods._shared.qa_agent import BaseAgentWorkflow

# Test questions (from financial_reports)
TEST_QUESTIONS = [
    {
        "qid": "fin_a_001",
        "domain": "financial_reports",
        "split": "A",
        "question": "根据比亚迪连续两年的年度报告，下列关于公司经营业绩变化的描述中，哪些是准确的？",
        "options": {
            "A": "2025 年营业收入较 2024 年实现增长",
            "B": "2025 年归属于上市公司股东的净利润出现下滑",
            "C": "2025 年经营活动产生的现金流量净额优于 2024 年",
            "D": "2025 年研发投入占营业收入的比例较 2024 年有所下降"
        },
        "answer_format": "multi",
        "type": "财务指标对比分析",
        "doc_ids": ["annual_byd_2024_report", "annual_byd_2025_report"]
    },
    {
        "qid": "fin_a_016",
        "domain": "financial_reports",
        "split": "A",
        "question": "关于宁德时代 2024 年年度报告与美的集团 2025 年年度报告中披露的利润分配方案，以下说法正确的有？",
        "options": {
            "A": "宁德时代 2024 年拟每 10 股派发现金红利 45.53 元（含税）",
            "B": "美的集团 2025 年拟每 10 股派发现金红利 38 元（含税）",
            "C": "两家公司在对应年份均实施了资本公积金转增股本",
            "D": "美的集团 2025 年现金分红与股份回购总金额超过当年归母净利润"
        },
        "answer_format": "multi",
        "type": "财务分配政策比较",
        "doc_ids": ["annual_catl_2024_report", "annual_midea_2025_report"]
    },
]


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_state_summary(state):
    """Print a detailed summary of the QAState after workflow completes."""
    print(f"\n--- QAState Summary for {state.qid} ---")
    print(f"  Question: {state.question[:80]}...")
    print(f"  Format: {state.answer_format}")
    print(f"  Question Type: {state.question_type}")
    print(f"  Doc Scope: {state.doc_scope}")
    print(f"  Global Query: {state.global_query[:80]}...")
    print(f"  Rounds completed: {state.round_id}")
    print(f"  Tool calls: {len(state.tool_history)}")
    print(f"  Evidence pool size: {len(state.raw_evidence_pool)}")
    print(f"  Tokens: prompt={state.prompt_tokens}, completion={state.completion_tokens}, total={state.total_tokens}")

    print(f"\n  --- Evidence Groups ({len(state.evidence_groups)}) ---")
    for g in state.evidence_groups:
        print(f"    [{g.group_id}] type={g.group_type} mode={g.mode} topk={g.topk}")
        print(f"      query: {g.query[:100]}...")
        print(f"      covers: {g.covers_options}")

    print(f"\n  --- Claims ---")
    for opt_key, cs in sorted(state.claims.items()):
        print(f"    [{opt_key}] status={cs.status} confidence={cs.confidence:.2f}")
        print(f"      text: {cs.original_text[:80]}...")
        print(f"      risk_flags: {cs.risk_flags}")
        print(f"      reason: {cs.reason[:100]}" if cs.reason else "")
        print(f"      support_evidence: {len(cs.support_evidence_ids)} items")
        print(f"      counter_evidence: {len(cs.counter_evidence_ids)} items")

    print(f"\n  --- Working Memory ---")
    for opt_key, mem in sorted(state.working_memory.items()):
        print(f"    [{opt_key}] facts={len(mem.key_facts)} support={len(mem.support_evidence_ids)}")
        if mem.key_facts:
            print(f"      first_fact: {mem.key_facts[0][:120]}...")

    print(f"\n  --- Reflection Actions ({len(state.reflection_actions)}) ---")
    for a in state.reflection_actions:
        print(f"    tool={a.tool} target={a.target_option} reason={a.reason[:80]}")

    print(f"\n  --- Tool History ---")
    for t in state.tool_history:
        print(f"    R{t.round_id} {t.tool_name}: targets={t.target_options} results={t.result_count} latency={t.latency_ms:.0f}ms")

    print(f"\n  *** FINAL ANSWER: {state.final_answer} ***")
    print(f"  Aggregator reasoning: {state.aggregator_reasoning[:200]}" if state.aggregator_reasoning else "")


def main():
    print_separator("m06 Claim-Centric Verification Agent - Quick Test")
    print(f"Config: model={_cfg.MODEL_NAME}, retriever={_cfg.RETRIEVER_NAME}")
    print(f"Agent: version={_cfg.AGENT_CONFIG.version}, max_rounds={_cfg.AGENT_CONFIG.max_rounds}")
    print(f"  counter_search={_cfg.AGENT_CONFIG.enable_counter_search}")
    print(f"  option_fallback={_cfg.AGENT_CONFIG.enable_option_fallback}")
    print(f"  llm_planner={_cfg.AGENT_CONFIG.use_llm_planner}")
    print(f"  llm_verifier={_cfg.AGENT_CONFIG.use_llm_verifier}")

    # Build retriever
    print_separator("Building Retriever")
    retriever = build_retriever(
        _cfg.RETRIEVER_NAME,
        index_dir=_cfg.HIKEY_INDEX_DIR,
        max_siblings=_cfg.MAX_SIBLINGS,
        skip_doc_routing=_cfg.SKIP_DOC_ROUTING,
    )
    # Warmup for financial_reports domain
    warmup = getattr(retriever, "warmup", None)
    if callable(warmup):
        warmup(domains=["financial_reports"])
        logger.info("Retriever warmed up for financial_reports")

    # Build LLM client
    print_separator("Building LLM Client")
    client = make_dashscope_client(api_key=_cfg.DASHSCOPE_API_KEY)
    logger.info("LLM client ready: %s", _cfg.MODEL_NAME)

    # Build workflow
    workflow = BaseAgentWorkflow(
        retriever=retriever,
        llm_client=client,
        config=_cfg.AGENT_CONFIG,
    )
    logger.info("Workflow ready")

    # Run test questions
    for i, q in enumerate(TEST_QUESTIONS):
        print_separator(f"Question {i+1}/{len(TEST_QUESTIONS)}: {q['qid']}")
        print(f"  Q: {q['question']}")
        for k, v in q['options'].items():
            print(f"  {k}: {v}")
        print(f"  Format: {q['answer_format']}")
        print(f"  Docs: {q['doc_ids']}")

        t0 = time.time()
        state = workflow.run(q)
        elapsed = time.time() - t0

        print_state_summary(state)
        print(f"\n  Time elapsed: {elapsed:.1f}s")
        print(f"  Result dict: {json.dumps(state.to_result_dict(), ensure_ascii=False, indent=2)[:500]}")

    print_separator("Test Complete")


if __name__ == "__main__":
    main()
