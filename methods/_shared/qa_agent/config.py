"""Agent configuration dataclass.

All strategy parameters are centralised here so that ablation experiments
can be run by simply swapping config instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration for the Claim-Centric QA Agent workflow.

    Designed for easy ablation: toggle features on/off, adjust topk,
    control rounds, etc. Each m06/m07/m08 variant can override specific
    fields while inheriting sensible defaults.
    """

    # --- Identity ---
    version: str = "m06"

    # --- Execution control ---
    max_rounds: int = 2
    max_tool_calls: int = 12

    # --- Retrieval parameters ---
    global_topk: int = 10
    group_topk: int = 10
    option_topk: int = 10
    option_fallback_topk: int = 8
    counter_topk: int = 8
    field_topk: int = 8
    final_evidence_topk: int = 28

    # --- Round 2 retrieval (increased budget) ---
    round2_option_topk: int = 20  # 2x of option_topk for retry searches
    round2_metric_topk: int = 10  # topk for metric-focused queries

    # --- Evidence budget ---
    max_evidence_per_claim: int = 8
    max_raw_evidence_for_verifier: int = 6
    evidence_token_budget: int = 24000

    # --- Feature toggles ---
    enable_grouped_retrieval: bool = True
    enable_option_fallback: bool = True
    enable_counter_search: bool = True
    enable_field_lookup: bool = False  # m06 first version: off
    enable_calculator: bool = False  # m06 first version: off
    enable_vlm_inspect: bool = False  # future: m08

    # --- LLM usage ---
    use_llm_planner: bool = True
    use_llm_verifier: bool = True
    use_llm_aggregator: bool = False  # rule-based by default

    # --- LLM parameters ---
    model_name: str = "qwen-plus"
    temperature: float = 0.1
    enable_thinking: bool = True
    thinking_budget: int = 10000
    max_output_tokens: int = 12000
    max_output_tokens_reflection: int = 16000

    # --- Reflection triggers ---
    # Options containing these keywords trigger counter-search
    counter_keywords: List[str] = field(default_factory=lambda: [
        "不", "未", "无", "均", "全部", "所有", "任一",
        "超过", "低于", "不超过", "不少于", "高于",
        "同比增长", "同比下降", "资本公积金转增", "送红股",
    ])

    # --- Debug ---
    debug_trace: bool = True
    save_intermediate: bool = True

    # --- Domain hints (inherited from m05) ---
    domain_hints: Dict[str, str] = field(default_factory=lambda: {
        "insurance": "注意保险责任、免责条款、等待期、赔付条件、责任范围和例外条件。",
        "regulatory": '注意法规适用范围、义务强度、时限要求、例外条件，以及"应当/可以/不得"的区别。',
        "financial_contracts": "注意发行金额与注册金额、评级、期限、利率、回售赎回、违约条款和主体角色。",
        "financial_reports": "注意年份、单位、同比、每股/每10股、归母/扣非、研发投入/研发费用等口径差异。",
        "research": "注意研报中的趋势、图表数值、单位、公司/行业对比和结论适用范围。",
    })

    def with_overrides(self, **kwargs) -> "AgentConfig":
        """Return a copy with specific fields overridden (for ablation)."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)
