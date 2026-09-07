# m05_HiKEY_question_option 方法说明

`m05_HiKEY_question_option` 是在 `m04_HiKEY` 的 HiKEY 文档结构化缓存基础上继续增强的一版。它不重新解析 PDF，而是直接复用 m04 已生成的 `DocCard / SectionCard / EvidenceUnit / FieldCard` 缓存，把重点从“文档解析能力”转向“选择题逐选项证据召回与答案判定”。

相比 m04 只围绕题干检索证据，m05 会同时对题干和每个选项分别检索：

```text
题干
题干 + 选项 A
题干 + 选项 B
题干 + 选项 C
题干 + 选项 D
```

然后将多路检索结果去重、重排、压缩成一个 evidence pack，再交给 Qwen 进行最终作答。

---

## 这版主要优化了什么

### 1. 复用 m04 的 HiKEY 结构化缓存

m05 不再从原始 PDF 开始解析，而是读取：

```text
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/doc_card.json
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/sections.jsonl
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/units.jsonl
methods/_shared/_HiKEY_cache/{domain}/{doc_id}/field_cards.jsonl
```

核心入口在：

```text
methods/m05_HiKEY_question_option/pipeline.py
```

其中 `HiKEYQuestionOptionIndexManager` 会加载 m04 的 `SimpleRetriever`，并继续使用 m04 中的：

```text
QueryPlanner
SimpleRetriever
COMPANY_ALIASES
METRIC_ALIASES
```

这意味着 m05 仍然保留 HiKEY 的章节路径、字段卡片、表格字段、页码和相邻上下文能力，但避免了重复解析文档的时间成本。

---

### 2. 从题干检索升级为题干 + 选项检索

m05 的核心变化是 `_retrieve_question_option_evidence()`。

它会先用题干做一次全局检索：

```text
GLOBAL_TOPK
```

再对每个选项拼接题干后分别检索：

```text
OPTION_TOPK
```

例如一道多选题会形成类似下面的检索请求：

```text
原始题干
原始题干 + A. 选项内容
原始题干 + B. 选项内容
原始题干 + C. 选项内容
原始题干 + D. 选项内容
```

这样做的原因是比赛题目中很多选项只在局部事实上有差别，仅靠题干检索容易召回“主题相关但无法验证选项”的证据。选项级检索可以把发行金额、评级、等待期、免责条款、法规义务、财务指标等细粒度事实拉进 evidence pack。

---

### 3. 多路 evidence pack 去重和重排

多路检索后，m05 使用 `_merge_evidence_packs()` 合并证据。

合并逻辑包括：

```text
按 anchor id 去重
记录每条证据命中的 query 来源
保留每条证据的最高原始分数
对被多个 query 命中的证据加权
最终截断为 FINAL_EVIDENCE_TOPK
```

重排时会给重复命中的证据额外加分：

```text
final_score = best_score + 1.25 * (命中来源数 - 1)
```

这个设计的直觉是：如果同一条证据同时被题干和多个选项命中，它往往更可能是判断该题的关键证据；但加权幅度有限，避免“命中次数”完全压过原始检索分数。

---

### 4. 保留 HiKEY-style Evidence Pack

发送给模型的证据不是简单文本片段，而是带结构的 evidence pack。

每条证据会尽量包含：

```text
命中来源：题干 / 选项A / 选项B / ...
文档 doc_id
章节路径 section_path
页码
字段名 metric
字段值 value_map
单位 unit
表格名 table_name
相邻上下文 siblings
```

对于 `FieldCard`，m05 会优先呈现指标名、单位、表格名和值映射；对于普通文本或表格单元，则呈现文本、行标题、单位和相邻内容。

这让模型不只看到“相似文本”，还可以看到证据在文档结构中的位置。

---

### 5. 按题型使用不同提示词

m05 根据 `answer_format` 区分三类题型：

```text
mcq    单选题
multi  多选题
tf     判断题
```

对应提示词分别为：

```text
QA_PROMPT_TEMPLATE
MULTI_QA_PROMPT_TEMPLATE
TF_PROMPT_TEMPLATE
```

同时还会加入领域提示：

```text
insurance
regulatory
financial_contracts
financial_reports
research
```

例如金融合同题会提醒模型注意发行金额、评级、期限、利率、回售赎回、违约条款和主体角色；监管题会提醒区分“应当 / 可以 / 不得”。

---

### 6. 多选题加入二次 reflection

当前线上和本地测试都显示，多选题最容易出现漏选。m05 针对这一点加入了一个轻量二次检查：

```text
如果多选题第一次答案长度 <= 1，则触发 MULTI_REFLECTION_PROMPT
```

reflection 会要求模型重新逐项判断 A/B/C/D，若二次答案包含更多选项，则采用二次答案。

这不是完整的 verifier，但能针对“多选题被答成单选”的常见错误做补救。

---

## 当前关键参数

默认参数集中在：

```text
methods/m05_HiKEY_question_option/config.py
```

主要参数如下：

```text
MODEL_NAME = qwen-plus
CONCURRENCY = 10
TEMPERATURE = 0.1
ENABLE_THINKING = 1
THINKING_BUDGET = 10000
MAX_OUTPUT_TOKENS = 12000
MAX_OUTPUT_TOKENS_REFLECTION = 16000

GLOBAL_TOPK = 10
OPTION_TOPK = 10
FINAL_EVIDENCE_TOPK = 28
MAX_SIBLINGS = 12
EVIDENCE_TOKEN_BUDGET = 24000
SKIP_DOC_ROUTING = 1
```

其中最影响效果和 token 消耗的是：

```text
OPTION_TOPK
FINAL_EVIDENCE_TOPK
EVIDENCE_TOKEN_BUDGET
ENABLE_THINKING
THINKING_BUDGET
```

当前这版为了提高召回，证据预算较大，因此 token 消耗会明显高于只做题干检索的方案。

---

## 运行方式

### 默认运行

```bash
python methods/m05_HiKEY_question_option/run.py
```

### 指定领域运行

```bash
python methods/m05_HiKEY_question_option/run.py --domains financial_reports insurance
```

### 调整并发

```bash
python methods/m05_HiKEY_question_option/run.py --concurrency 5
```

### 调整检索条数

```bash
python methods/m05_HiKEY_question_option/run.py \
  --global-topk 8 \
  --option-topk 8 \
  --final-topk 24
```

### 调整证据 token 预算

```bash
python methods/m05_HiKEY_question_option/run.py --evidence-budget 18000
```

### 指定模型

```bash
python methods/m05_HiKEY_question_option/run.py --model qwen-plus
```

---

## 输出文件

运行完成后，`BaseRunner` 会将结果写入：

```text
runs/{CODER}/m05_HiKEY_question_option/{run_id}/output/
```

主要输出包括：

```text
answer.csv
partial_results.json
evidence.json
```

其中 `answer.csv` 用于提交或评分，`partial_results.json` 保存每题的原始回答、token 消耗、领域和检索元信息，`evidence.json` 保存证据片段，方便回溯错误题。

---

## 当前效果观察

本地用 `GPT-Pro-Answer/gpt_pro_all_answers.json` 作为参考答案比较时，当前一次运行结果为：

```text
总题数：100
正确数：64
准确率：64.00%
```

按题号前缀拆分：

```text
fc_a:  12/20 = 60.00%
fin_a: 14/20 = 70.00%
ins_a: 15/20 = 75.00%
reg_a:  6/20 = 30.00%
res_a: 17/20 = 85.00%
```

线上最终分数为 52.71，结合本次 `summary.total_tokens = 1,248,187` 和比赛公式反推，线上准确率约为：

```text
56.98%
```

这说明当前本地参考答案和线上标准之间仍有差异；同时也说明 m05 的 token 效率仍有优化空间。

---

## 当前主要问题

### 1. 证据召回多，但逐项验证还不够硬

m05 已经把选项级证据召回做起来了，但最终判断仍然主要依赖一次模型生成。对于多选题、法规题和细粒度合同条款题，模型可能出现：

```text
看到相似证据后误判
遗漏某个正确选项
把“未提及”当成“错误”
把条件性表述当成绝对表述
```

下一步应把选项 A/B/C/D 拆成独立 claim，分别输出：

```text
supported
contradicted
insufficient
```

最后再聚合答案。

---

### 2. regulatory 领域表现偏弱

当前本地比较中 `reg_a` 只有 30.00%。这类题往往对措辞非常敏感，尤其是：

```text
应当 / 可以 / 不得
除外 / 但是 / 前款规定
期限
主体范围
监管机构职责
```

单纯相似度检索容易召回同主题但不完全对应的条款，需要更强的条文定位和逻辑校验。

---

### 3. token 消耗偏高

m05 每题会进行题干 + 多个选项检索，并把较大的 evidence pack 送给模型。当前一次运行 token 消耗为：

```text
1,248,187
```

虽然低于比赛预算 `5,000,000`，但根据公式 token 效率仍会影响最高 30% 的加权系数。后续可以考虑：

```text
降低 FINAL_EVIDENCE_TOPK
缩短 siblings
按题型动态设置 evidence budget
对 evidence pack 做本地压缩
只对困难题启用 thinking 或 reflection
```

---

## 代码中建议继续补强的方向

### 1. Option-level Verifier

这是 m05 之后最值得继续做的方向。

建议流程：

```text
选项 A/B/C/D
-> 原子命题拆解
-> 每个命题检索 FieldCard / EvidencePack
-> supported / contradicted / insufficient
-> 聚合最终答案
```

这样可以把“生成式作答”改成“逐项证据裁决”，尤其适合多选题和判断题。

---

### 2. 领域专用规则

不同领域可加入轻量规则：

```text
financial_reports: 单位、年份、同比、归母/扣非、每股/每10股
financial_contracts: 发行主体、规模、评级、期限、受托管理人、担保、回售赎回
insurance: 等待期、免责条款、赔付条件、责任范围
regulatory: 义务强度、主体范围、例外条款、期限
research: 图表数值、行业/公司对比、结论适用范围
```

规则不一定直接决定答案，但可以在 verifier 中作为校验 checklist。

---

### 3. 检索预算动态化

当前所有题使用同一套 top-k 和 token budget。后续可以改成：

```text
单选题：较小 evidence pack
多选题：更高 option_topk 和 reflection
判断题：优先召回定义、条件、例外
regulatory：提高章节邻域和条文上下文
financial_reports：优先 FieldCard
```

这样有机会在不显著降准确率的情况下减少 token。

---

### 4. 失败题回放和证据诊断

建议针对错误题保存更完整的诊断字段：

```text
每个 query 的原始 top-k
每条证据的 source query
最终 evidence pack 的截断位置
模型原始 reasoning
reflection 前后答案差异
```

这样可以区分错误来源是：

```text
检索没召回
召回了但被截断
召回了但模型误判
标准答案存在差异
```

---

## 小结

m05 的定位不是替代 m04，而是在 m04 已经完成 HiKEY 结构化索引的基础上，进一步面向比赛选择题做“选项级证据召回”。它的主要价值是把 evidence pack 从题干相关，推进到选项可验证。

当前版本已经具备：

```text
复用 HiKEY 缓存
题干 + 选项多路检索
证据去重重排
题型提示词
领域提示词
多选题 reflection
token 统计和结果落盘
```

下一步最关键的是补上真正的 `Option Verifier`，把每个选项从一次性生成判断改成可追溯的逐项证据裁决。
