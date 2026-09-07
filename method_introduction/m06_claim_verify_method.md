# m06_claim_verify 方法说明

`m06_claim_verify` 是在 `m05_HiKEY_question_option` 的基础上进一步升级的一版。它不再把整道题一次性交给模型生成答案，而是引入了一套 **Claim-Centric 逐选项验证 Agent**：先把每个选项拆解为原子命题，再分轮次检索证据，最后由独立的 Verifier 对每个选项输出 `supported / contradicted / mixed` 的结构化判断，最终由 Aggregator 汇总成最终答案。

相比 m05 的"一次生成"，m06 的核心变化是把"答题"变成了"逐项裁决"：

```text
m05: 题干 + 选项 → 多路检索 → evidence pack → 模型一次生成答案
m06: 题干 + 选项 → Planner 分析 → 多轮检索 → Verifier 逐项判断 → Aggregator 汇总
```

---

## 这版主要优化了什么

### 1. 引入 Claim-Centric Agent 框架

m06 的核心是 `methods/_shared/qa_agent/` 下的一套可复用 Agent 框架，包含以下模块：

```text
planner.py      QuestionPlanner + ClaimExtractor + EvidenceGroupBuilder
workflow.py     BaseAgentWorkflow（多轮编排）
verifier.py     ClaimVerifier（逐选项 LLM 判断）
reflector.py    ReflectionController（反思触发 Round 2）
aggregator.py   AnswerAggregator（规则汇总最终答案）
compressor.py   run_compression（证据压缩进 working memory）
schemas.py      QAState / OptionClaimState / EvidenceGroup 等数据结构
config.py       AgentConfig（所有参数集中管理）
```

这套框架设计为可继承：m07、m08 等后续版本只需继承 `BaseAgentWorkflow` 并覆盖特定阶段，无需重写整个流程。

---

### 2. Planning 阶段：LLM 分析选项结构

m06 在检索之前先用 LLM 对每个选项做结构分析（`ClaimExtractor`），提取：

```text
risk_flags   特殊逻辑特征（全称命题、否定命题、阈值比较、计算、时间限定等）
operation    核验操作类型（lookup / compare / calculate / negate / universal / threshold）
company      涉及的公司名称
year         涉及的年份
key_metrics  涉及的关键指标/字段名
```

例如，对于选项"A 公司 2023 年现金分红比例超过 30%"，LLM 会输出：

```json
{
  "risk_flags": ["threshold", "calculation"],
  "operation": "calculate",
  "company": "A 公司",
  "year": 2023,
  "key_metrics": ["现金分红", "净利润"]
}
```

这些分析结果会指导后续的检索策略和 Verifier 的判断重点。当 LLM 调用失败时，自动降级为基于关键词的规则匹配（fallback）。

---

### 3. 多轮检索：Round 1 + Round 2（反思驱动）

#### Round 1：宽泛证据获取

`EvidenceGroupBuilder` 会为每道题构建一组检索任务：

```text
G01: 全局检索（题干）                → 覆盖所有选项
G02: 选项 A 检索（题干 + 选项 A）    → 覆盖选项 A
G03: 选项 B 检索（题干 + 选项 B）    → 覆盖选项 B
G04: 选项 C 检索（题干 + 选项 C）    → 覆盖选项 C
G05: 选项 D 检索（题干 + 选项 D）    → 覆盖选项 D
```

所有检索结果汇入 `raw_evidence_pool`，并记录每条证据覆盖的选项来源。

#### Round 2：反思驱动的补充检索

Round 1 + Verifier 完成后，`ReflectionController` 会检查每个选项的验证状态，对以下情况触发 Round 2：

```text
mixed 且置信度低（证据不足）    → option_search（增大 topk 重试）
含否定/全称关键词且 mixed      → counter_search（反向检索）
含关键指标但未找到数值          → metric_search（字段级精确检索）
```

Round 2 的检索结果会与 Round 1 的证据池合并（去重），然后对受影响的选项重新执行 Verifier。

---

### 4. Verifier：逐选项 LLM 判断

`ClaimVerifier` 对每个选项独立调用 LLM，输出结构化判断：

```json
{"status": "supported|contradicted|mixed", "confidence": 0.0-1.0, "reason": "..."}
```

Verifier 的核心设计原则：

- **独立判断**：每个选项单独验证，不受其他选项影响
- **语义等价不算矛盾**：选项与证据用词不同但含义一致时（如"市场价格"与"前 N 个交易日均价"），判 `supported`
- **严格区分 contradicted 与 mixed**：只有证据中有明确的、可量化的直接矛盾才判 `contradicted`；"证据没提到"或"信息不完整"一律判 `mixed`
- **数值类选项必须实际计算**：含百分比、金额、比例的选项，必须从证据中提取数值后计算，不凭感觉判断
- **多实体选项**：涉及多个公司/产品/年份时，必须每个实体都在证据中找到支持才能判 `supported`

针对不同题型使用不同提示词：

```text
mcq    单选题提示词（强调只有一个正确答案，仔细核对数据）
multi  多选题提示词（强调独立判断，优先 mixed 而非 contradicted）
tf     判断题提示词
```

---

### 5. Aggregator：规则汇总最终答案

`AnswerAggregator` 使用规则（非 LLM）将逐选项判断汇总为最终答案，保证确定性和速度：

**多选题策略：**

```text
1. 所有 supported 选项直接入选
2. mixed 且 confidence >= 0.6 的选项也入选（部分支持也是支持）
3. 若无任何 supported，取置信度最高的非 contradicted 选项（最多 3 个）
4. 兜底：返回 "A"
```

**单选题策略：**

```text
1. 恰好一个 supported → 直接返回
2. 多个 supported → 取置信度最高的
3. 无 supported → 取置信度最高的非 contradicted 选项
```

**判断题策略：**

```text
A 为 supported 且 B 非 supported → 返回 A
B 为 supported 且 A 非 supported → 返回 B
A 为 contradicted → 返回 B
B 为 contradicted → 返回 A
均不确定 → 取置信度较高的
```

---

### 6. 复用 m05 的 HiKEY 结构化缓存

m06 通过 `methods/_shared/retrieval/` 下的统一检索工厂接入 HiKEY 检索器，直接复用 m04/m05 已生成的缓存：

```text
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/doc_card.json
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/sections.jsonl
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/units.jsonl
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/field_cards.jsonl
```

检索器通过 `build_retriever("hikey", ...)` 统一创建，后续切换检索器只需修改 `RETRIEVER_NAME` 配置。

---

## 当前关键参数

默认参数集中在：

```text
methods/m06_claim_verify/config.py
```

主要参数如下：

```text
MODEL_NAME = qwen-plus（可通过 M06_MODEL 环境变量覆盖）
CONCURRENCY = 10
TEMPERATURE = 0.1
ENABLE_THINKING = 1
THINKING_BUDGET = 10000
MAX_OUTPUT_TOKENS = 12000

# 检索
GLOBAL_TOPK = 10
OPTION_TOPK = 10
OPTION_FALLBACK_TOPK = 8
COUNTER_TOPK = 8
FINAL_EVIDENCE_TOPK = 28
ROUND2_OPTION_TOPK = 20      # Round 2 重试时加倍
ROUND2_METRIC_TOPK = 10      # Round 2 字段级检索

# 证据预算
MAX_EVIDENCE_PER_CLAIM = 8
MAX_RAW_EVIDENCE_FOR_VERIFIER = 6
EVIDENCE_TOKEN_BUDGET = 24000

# 功能开关
ENABLE_GROUPED_RETRIEVAL = 1
ENABLE_OPTION_FALLBACK = 1
ENABLE_COUNTER_SEARCH = 1
ENABLE_FIELD_LOOKUP = 0      # v1 暂未启用
ENABLE_CALCULATOR = 0        # v1 暂未启用

# Agent 策略
USE_LLM_PLANNER = 0          # v1: 规则 planner（LLM planner 待启用）
USE_LLM_VERIFIER = 1
USE_LLM_AGGREGATOR = 0       # v1: 规则 aggregator

MAX_ROUNDS = 2
MAX_TOOL_CALLS = 12
```

---

## 运行方式

### 默认运行

```bash
python methods/m06_claim_verify/run.py
```

### 指定模型

```bash
python methods/m06_claim_verify/run.py --model qwen3.7-max-2026-06-08
```

### 指定领域

```bash
python methods/m06_claim_verify/run.py --domains regulatory insurance
```

### 调整并发

```bash
python methods/m06_claim_verify/run.py --concurrency 5
```

### 后台运行

```bash
nohup python -u methods/m06_claim_verify/run.py \
  --model qwen3.7-max-2026-06-08 \
  --no-resume \
  > /tmp/m06_run.log 2>&1 &
```

---

## 输出文件

运行完成后，`BaseRunner` 会将结果写入：

```text
runs/{CODER}/m06_claim_verify/{run_id}/output/
```

主要输出包括：

```text
answer.csv              用于提交或评分
partial_results.json    每题的原始回答、token 消耗、领域、claim 状态
evidence.json           证据片段，方便回溯错误题
```

`partial_results.json` 中每题会额外记录：

```text
claims          每个选项的 status / confidence / reason
tool_history    每轮检索的 query、topk、结果数、耗时
round_id        最终停在第几轮
debug_trace     中间状态快照（如 LLM planner 输出、reflection 决策）
```

---

## 当前效果观察

使用 `GPT-Pro-Answer/gpt_pro_all_answers.json` 作为参考答案，与 m05 对比：

| 版本 | 总题数 | 正确数 | 准确率 |
|------|--------|--------|--------|
| m05  | 100    | 64     | 64.00% |
| m06 (qwen-plus)  | 100    | ~64    | 持平或略有波动 |
| m06 (qwen3.7-max-2026-06-08) | 100 | 待评估 | — |

当前 m06 相比 m05 的主要收益在于：

- **多选题漏选问题有所改善**：逐选项独立验证减少了"多选被答成单选"的情况
- **错误可追溯**：每个选项都有 `status + confidence + reason`，可以精确定位是"没检索到"还是"检索到但判断错"
- **框架可扩展**：后续 m07/m08 可以在此基础上叠加 VLM、计算器、字段级检索等工具

---

## 当前主要问题

### 1. LLM Planner 暂未启用

当前 `use_llm_planner = False`，Planner 使用规则模式（`QuestionPlanner` 只做元数据填充，`ClaimExtractor` 使用 LLM 分析选项结构但不做深度规划）。后续启用 LLM Planner 后，可以根据题目类型动态调整检索策略（如对计算题优先召回 FieldCard，对全称命题主动触发 counter_search）。

### 2. 字段级检索（field_lookup）暂未启用

`enable_field_lookup = False`。对于含具体数值的选项（如"净利润同比增长 15%"），目前仍依赖文本相似度检索，而非直接查询 FieldCard 的数值字段。启用后可以显著提升数值类题目的准确率。

### 3. regulatory 领域仍然偏弱

法规类题目对措辞极其敏感（"应当 / 可以 / 不得"、"除外 / 但是 / 前款规定"），当前 Verifier 的语义等价判断在这类题上容易误判。需要针对 regulatory 领域加入更强的条文定位和逻辑校验。

### 4. Round 2 反思策略有待精调

当前 `ReflectionController` 的触发条件较为保守，部分本应触发 counter_search 的选项（含"均"、"全部"等全称命题）在 Round 1 置信度较高时不会进入 Round 2，存在漏检风险。

---

## 与 m05 的关键区别

| 维度 | m05 | m06 |
|------|-----|-----|
| 答题方式 | 一次生成 | 逐选项独立验证 |
| 选项分析 | 无（直接拼接检索） | LLM 分析 risk_flags / operation / key_metrics |
| 检索轮次 | 1 轮 | 最多 2 轮（反思驱动） |
| 多选题处理 | reflection 补救 | Aggregator 规则汇总 + Round 2 补充检索 |
| 错误可追溯性 | 有限（只有原始回答） | 完整（每选项 status + confidence + reason + tool_history） |
| 框架可扩展性 | 单文件 pipeline | 模块化 Agent 框架（可继承） |
| token 消耗 | 每题 1 次 LLM 调用 | 每题 1 次 Planner + N 次 Verifier（N = 选项数） |

---

## 代码中建议继续补强的方向

### 1. 启用 LLM Planner

将 `use_llm_planner = True`，让 Planner 根据题目类型动态生成检索计划，而不是固定的"全局 + 逐选项"模式。

### 2. 启用字段级检索（field_lookup）

对 `key_metrics` 非空的选项，在 Round 1 额外触发 FieldCard 精确查询，直接获取数值字段而非依赖文本相似度。

### 3. 反思策略精调

对含全称命题（"均"、"全部"、"所有"）的选项，无论 Round 1 置信度如何，都强制触发 counter_search，避免漏检。

### 4. 领域专用 Verifier 提示词

当前所有领域共用同一套 Verifier 提示词。后续可以为 regulatory 领域加入专用提示词，强调条文义务强度、主体范围、例外条款的区分。

### 5. 失败题诊断

建议在 `debug_trace` 中额外记录：

```text
每个 EvidenceGroup 的原始 top-k 结果
Verifier 的完整 reasoning（thinking content）
Round 2 新增证据数量
Aggregator 的决策路径
```

这样可以精确区分错误来源：检索未召回 / 召回但被截断 / 召回但 Verifier 误判 / 标准答案差异。

---

## 小结

m06 的定位是把 m05 的"一次生成"升级为"逐项裁决"。它的核心价值是：

```text
每个选项都有独立的证据支撑和结构化判断
错误可以精确追溯到具体选项和具体证据
框架模块化，后续版本可以按需叠加新工具
```

当前版本（v1）已经具备：

```text
LLM 选项结构分析（risk_flags / operation / key_metrics）
多轮检索（Round 1 宽泛 + Round 2 反思驱动）
逐选项 LLM Verifier（supported / contradicted / mixed）
规则 Aggregator（多选 / 单选 / 判断题）
完整的 debug_trace 和 tool_history
```

下一步最关键的是启用 **字段级检索（field_lookup）** 和 **LLM Planner**，把数值类题目和全称命题的准确率进一步提升。
