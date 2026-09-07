"""
Query Planner Agent for Financial Contracts domain.
Decomposes questions into atomic claims and generates structured retrieval plans.
Follows Skill v6.0: Financial_Contracts_RAG_Field_Verification_Workflow.

Migrated from Golden_label_baseline. Uses Qwen API instead of GPT-4.1.
No voting mechanism - single inference only.
"""
import json
import time
from typing import Dict, List
from openai import OpenAI

from .config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
)

# ============================================================
# Query Planner System Prompt
# ============================================================
QUERY_PLANNER_SYSTEM_PROMPT = """你是金融合同题目的 Query Planner。

任务：
将题目和选项改写为结构化检索计划。

要求：
1. 保持 doc_ids 顺序：
   第一份文档 = doc_ids[0]
   第二份文档 = doc_ids[1]
   （如有更多文档依次类推）
2. 识别题型：tf / mcq / multi
3. 从题干中抽取主题方向
4. 从选项中抽取具体字段
5. 将每个选项拆解为原子命题（atomic claims）
6. 不要为每个选项都单独检索
7. 应按 doc_id + field_cluster 聚合成字段检索包
8. 判断题应拆成每个 doc_id 的子问题，再设置最终逻辑 AND/OR
9. 对于"均/都/全部"等词，final_logic 设为 AND

字段本体参考（Field Ontology）：
- issuer: 发行人
- current_issue_amount_max: 本期发行金额上限
- registered_amount_max: 注册金额上限
- total_issue_amount: 本次发行总额
- subject_rating: 主体信用评级
- bond_rating: 债项信用评级
- rating_agency: 评级机构
- lead_underwriter: 牵头主承销商
- bookrunner: 簿记管理人
- trustee: 受托管理人
- disclosure_commitment: 信息披露义务承诺
- investor_protection: 投资者保护机制
- put_option: 回售条款
- redemption: 赎回条款
- asset_liability_ratio: 资产负债率
- net_profit: 净利润
- conversion_price: 转股价格
- conversion_period: 转股期限
- conditional_redemption: 有条件赎回
- conditional_put: 有条件回售
- listed_company: 上市公司
- transaction_counterparty: 交易对方
- target_company: 标的公司
- transaction_structure: 交易结构
- share_issuance_purchase_assets: 发行股份购买资产
- supporting_fundraising: 募集配套资金
- default_clause: 违约条款
- penalty_interest: 违约利息/罚息
- stock_code: 股票代码
- stock_name: 股票简称
- issuance_date: 发行日期
- maturity_date: 兑付日/到期日

文档类型参考：
- corporate_bond_prospectus: 公司债募集说明书
- convertible_bond_prospectus: 可转债募集说明书
- mna_restructuring_report: 重大资产重组报告书
- special_bond_prospectus: 铁路建设债券等特殊债券
- asset_purchase_agreement: 资产购买协议/补偿协议

常见字段簇（field_pack）：
- bond_issuance_elements: 发行人、发行金额、注册金额、评级、担保、中介机构
- disclosure_commitment: 信息披露义务、声明
- investor_protection: 回售、赎回、持有人会议、权利行权
- financial_metrics: 资产负债率、净利润、归母净利润、现金流
- mna_transaction_structure: 发行股份购买资产、募集配套资金、交易对方、标的公司
- convertible_bond_terms: 转股价格、转股期限、赎回条款、回售条款
- default_and_penalty: 违约条款、违约利息、罚息计算
- issuer_basic_info: 发行人类别、所属行业、主营业务、股票代码

输出格式（严格 JSON）：
{
  "qid": "题目ID",
  "answer_format": "multi/mcq/tf",
  "doc_scope": ["text01", "text02"],
  "doc_type_guess": {"text01": "类型", "text02": "类型"},
  "atomic_claims": [
    {
      "option": "A",
      "claims": [
        {
          "doc_id": "text01",
          "field": "issuer",
          "operator": "equals/contains/greater_than/less_than/exists/not_exists",
          "target": "目标值",
          "requires_explicit_evidence": true
        }
      ],
      "logic": "AND"
    }
  ],
  "field_packs": [
    {
      "doc_id": "text01",
      "field_pack": "bond_issuance_elements",
      "fields": ["issuer", "current_issue_amount_max", "subject_rating"],
      "preferred_sections": ["cover", "issuance_overview", "important_notice"],
      "query_keywords": "发行人 本期债券发行金额 不超过 主体信用评级 信用评级结果"
    }
  ],
  "final_logic": "AND/OR/INDEPENDENT"
}

注意事项：
- 对于选择题（mcq/multi），final_logic 设为 INDEPENDENT
- 对于判断题（tf），如果题干包含"均/都/全部"，final_logic 设为 AND
- preferred_sections 可选值：cover, statement, important_notice, issuance_overview, definition, risk_factors, fund_usage, issuer_info, credit_info, financial_info, investor_protection, disclosure_arrangement, trustee, institutions, transaction_overview, listed_company_info, counterparty_info, target_info, share_issuance, management_discussion
- 将指向同一 doc_id 且字段相近的选项合并到同一个 field_pack
- query_keywords 应包含字段的多种同义表达
- 只输出严格的 JSON，不要包含其它说明文本"""


class QueryPlanner:
    """Query Planner Agent that decomposes questions into retrieval plans."""

    def __init__(self):
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        self.model = MODEL_NAME

    def plan(self, question: str, options: Dict[str, str],
             answer_format: str, doc_ids: List[str],
             qid: str = "") -> Dict:
        """
        Generate a structured retrieval plan for a question.
        Single inference (no voting).
        """
        user_prompt = self._build_user_prompt(question, options, answer_format, doc_ids, qid)

        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': QUERY_PLANNER_SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                plan = json.loads(raw)
                plan = self._validate_plan(plan, question, options, answer_format, doc_ids, qid)
                # Track tokens
                plan['_usage'] = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                }
                return plan
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    print(f"  [RETRY] Query Planner JSON parse failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    print(f"  [ERROR] Query Planner failed after {retries} attempts (JSON)")
                    return self._fallback_plan(question, options, answer_format, doc_ids, qid)
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  [RETRY] Query Planner API failed: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [ERROR] Query Planner failed: {e}")
                    return self._fallback_plan(question, options, answer_format, doc_ids, qid)

    def _build_user_prompt(self, question: str, options: Dict[str, str],
                           answer_format: str, doc_ids: List[str], qid: str) -> str:
        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"

        doc_mapping = ""
        for i, doc_id in enumerate(doc_ids):
            doc_mapping += f"  第{i+1}份文档 = {doc_id}\n"

        return f"""请为以下题目生成结构化检索计划。

题目ID: {qid}
题型: {answer_format}
涉及文档:
{doc_mapping}
题目: {question}

选项:
{options_text.strip()}

请输出严格的 JSON 格式检索计划。"""

    def _validate_plan(self, plan: Dict, question: str, options: Dict[str, str],
                       answer_format: str, doc_ids: List[str], qid: str) -> Dict:
        plan.setdefault('qid', qid)
        plan.setdefault('answer_format', answer_format)
        plan.setdefault('doc_scope', doc_ids)
        plan.setdefault('atomic_claims', [])
        plan.setdefault('field_packs', [])

        if answer_format == 'tf':
            plan.setdefault('final_logic', 'AND')
        else:
            plan.setdefault('final_logic', 'INDEPENDENT')

        valid_doc_ids = set(doc_ids)
        plan['field_packs'] = [
            fp for fp in plan['field_packs']
            if fp.get('doc_id') in valid_doc_ids
        ]

        if not plan['field_packs']:
            plan['field_packs'] = self._generate_fallback_field_packs(
                question, options, doc_ids
            )

        return plan

    def _fallback_plan(self, question: str, options: Dict[str, str],
                       answer_format: str, doc_ids: List[str], qid: str) -> Dict:
        field_packs = self._generate_fallback_field_packs(question, options, doc_ids)

        atomic_claims = []
        for key, text in sorted(options.items()):
            claims = []
            for doc_id in doc_ids:
                claims.append({
                    "doc_id": doc_id,
                    "field": "general",
                    "operator": "verify",
                    "target": text,
                })
            atomic_claims.append({
                "option": key,
                "claims": claims,
                "logic": "AND"
            })

        return {
            "qid": qid,
            "answer_format": answer_format,
            "doc_scope": doc_ids,
            "atomic_claims": atomic_claims,
            "field_packs": field_packs,
            "final_logic": "AND" if answer_format == 'tf' else "INDEPENDENT",
            "_usage": {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        }

    def _generate_fallback_field_packs(self, question: str, options: Dict[str, str],
                                       doc_ids: List[str]) -> List[Dict]:
        all_text = question + " " + " ".join(options.values())

        field_packs = []
        for doc_id in doc_ids:
            keywords = []
            fields = []
            sections = ["cover", "issuance_overview", "important_notice"]

            if any(kw in all_text for kw in ["发行人", "发行主体", "公司名称"]):
                keywords.extend(["发行人", "发行主体"])
                fields.append("issuer")

            if any(kw in all_text for kw in ["发行金额", "发行规模", "不超过"]):
                keywords.extend(["发行金额", "发行规模", "本期债券", "不超过"])
                fields.append("current_issue_amount_max")

            if any(kw in all_text for kw in ["注册金额", "注册额度"]):
                keywords.extend(["注册金额", "注册额度"])
                fields.append("registered_amount_max")

            if any(kw in all_text for kw in ["主体评级", "主体信用评级", "信用评级"]):
                keywords.extend(["主体信用评级", "主体评级", "信用评级结果"])
                fields.append("subject_rating")

            if any(kw in all_text for kw in ["债项评级", "债项信用评级"]):
                keywords.extend(["债项评级", "债项信用评级"])
                fields.append("bond_rating")

            if any(kw in all_text for kw in ["受托管理人", "债券受托管理人"]):
                keywords.extend(["受托管理人", "债券受托管理人"])
                fields.append("trustee")

            if any(kw in all_text for kw in ["回售", "赎回", "投资者保护", "持有人权利"]):
                keywords.extend(["回售", "赎回", "投资者保护", "债券持有人权利"])
                fields.extend(["put_option", "redemption", "investor_protection"])
                sections.extend(["investor_protection", "issuance_overview"])

            if any(kw in all_text for kw in ["资产负债率", "净利润", "财务"]):
                keywords.extend(["资产负债率", "净利润", "财务指标"])
                fields.extend(["asset_liability_ratio", "net_profit"])
                sections.extend(["financial_info", "important_notice"])

            if any(kw in all_text for kw in ["转股", "可转换", "可转债"]):
                keywords.extend(["转股价格", "转股期限", "可转换公司债券"])
                fields.extend(["conversion_price", "conversion_period"])
                sections.extend(["issuance_overview"])

            if any(kw in all_text for kw in ["发行股份", "重组", "标的", "交易对方"]):
                keywords.extend(["发行股份购买资产", "重大资产重组", "标的公司", "交易对方"])
                fields.extend(["transaction_structure", "target_company"])
                sections.extend(["transaction_overview", "target_info"])

            if any(kw in all_text for kw in ["违约", "罚息", "赔偿", "补偿"]):
                keywords.extend(["违约", "违约利息", "罚息", "赔偿", "补偿"])
                fields.extend(["default_clause", "penalty_interest"])

            if any(kw in all_text for kw in ["信息披露", "披露义务"]):
                keywords.extend(["信息披露", "披露义务", "及时", "公平"])
                fields.append("disclosure_commitment")
                sections.extend(["statement", "disclosure_arrangement"])

            if any(kw in all_text for kw in ["股票代码", "证券代码", "股票简称"]):
                keywords.extend(["股票代码", "证券代码", "股票简称"])
                fields.extend(["stock_code", "stock_name"])

            if not fields:
                fields = ["general"]
                keywords = list(set(all_text.split()[:10]))

            field_packs.append({
                "doc_id": doc_id,
                "field_pack": "auto_detected",
                "fields": list(set(fields)),
                "preferred_sections": list(set(sections)),
                "query_keywords": " ".join(list(set(keywords))[:15]),
            })

        return field_packs
