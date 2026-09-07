# m05c_HiKEY_domain 方法说明

`m05c_HiKEY_domain` 是在 `m05_HiKEY_question_option` 基础上的一次结构升级：保留 HiKEY 的通用文档层级架构，同时增加五个金融领域的轻量领域卡片，并把最终作答改成更明确的逐选项裁决。

当前版本采用“法规强领域卡，其他领域弱领域卡”的策略。也就是说，`regulatory` 会强制保留一部分 DomainCard 进入最终 top-k；其他四个领域只把 DomainCard 作为弱候选参与重排，不再强制挤掉原本的 HiKEY 证据。

核心思路是：

```text
HiKEY 通用文档地图
+ DomainCard 领域事实索引
+ Option-level Verifier 逐选项判断
```

---

## 为什么要做 m05c

m05 的检索方式是题干 + 每个选项分别召回，再把去重后的 evidence pack 交给模型。这比只检索题干更好，但仍然存在一个问题：不同金融领域的“关键事实形态”差别很大。

例如：

```text
年报题看公司、年份、指标、单位、同比方向。
债券题看发行人、发行规模、评级、期限、中介机构。
保险题看责任、免责、免赔额、赔付比例、退保规则。
法规题看主体、应当/可以/不得、期限、例外、施行日期。
研报题看市场规模、预测年份、增速、CAGR、公司对比。
```

如果所有领域都只用通用 EvidenceUnit 和 FieldCard，模型容易拿到“主题相关但无法裁决选项”的证据。m05c 的目标是让召回结果更像“可验证事实”，而不是更长的文本。

---

## 共享层：继续复用 HiKEY

m05c 直接复用：

```text
methods/_shared/_HiKEY_cache
```

仍然读取每个文档的：

```text
doc_card.json
sections.jsonl
units.jsonl
field_cards.jsonl
```

也就是说，m05c 不重新解析 PDF，不改变 m04/m05 已经构建好的通用文档地图。

HiKEY 继续负责：

```text
文档 doc_id
章节路径 section_path
页码
正文片段
表格字段
相邻上下文 siblings
```

---

## 特有层：新增 DomainCard

m05c 新增：

```text
methods/m05c_HiKEY_domain/domain_cards.py
```

它会从 HiKEY 的 `units` 和 `field_cards` 中构建轻量领域卡片。

当前第一版包含：

```text
financial_reports:
  MetricYearCard
  DividendCard
  CompareCard

financial_contracts:
  BondTermCard
  PartyRoleCard
  ClauseCard

insurance:
  BenefitRuleCard
  ExclusionCard
  SurrenderRuleCard
  ClaimCalcCard

regulatory:
  ArticleCard
  ObligationCard
  DeadlineCard
  ExceptionCard

research:
  ForecastMetricCard
  MarketSizeCard
  CompanyCompareCard
```

这些卡片不是替代 HiKEY，而是给检索增加更贴近领域考点的索引入口。

从当前实验看，`regulatory` 的 DomainCard 最有效，因为法规题天然适合抽成：

```text
主体
义务强度
动作
条件
期限
例外
施行日期
```

其他四个领域的 DomainCard 第一版还比较粗，如果强制进入 top-k，反而可能挤掉 m05 原本更好的证据。因此现在默认只对 `regulatory` 使用强策略。

---

## 检索流程

m05c 使用“本地宽召回，进模型窄证据”的策略。

默认参数：

```text
QUESTION_TOPK = 6
OPTION_TOPK = 8
DOMAIN_CARD_TOPK = 10
PER_OPTION_EVIDENCE = 3
FINAL_EVIDENCE_TOPK = 12
EVIDENCE_TOKEN_BUDGET = 12000
```

领域卡策略参数：

```text
STRONG_DOMAIN_CARD_DOMAINS = regulatory
WEAK_DOMAIN_CARD_TOPK = 4
WEAK_PER_OPTION_EVIDENCE = 1
WEAK_DOMAIN_CARD_MIN_SCORE = 4.0
```

流程如下：

```text
题干 -> HiKEY 检索 + DomainCard 检索
选项 A -> HiKEY 检索 + DomainCard 检索
选项 B -> HiKEY 检索 + DomainCard 检索
选项 C -> HiKEY 检索 + DomainCard 检索
选项 D -> HiKEY 检索 + DomainCard 检索
本地合并、去重、重排、压缩
最终最多 12 条证据进入模型
```

对于 `regulatory`，最终 top-k 会预留部分 DomainCard；对于其他领域，最终 top-k 按综合分排序，不做 DomainCard 预留。这样可以保留法规领域的提升，同时减少其他领域因弱 DomainCard 干扰而降分。

---

## 作答方式

m05c 不再让模型直接“看完证据输出字母”，而是要求模型输出 JSON：

```json
{
  "option_verdicts": {
    "A": {
      "verdict": "supported",
      "evidence_ids": ["A-D1", "H3"],
      "reason": "..."
    }
  },
  "answer": "A"
}
```

其中 verdict 只能是：

```text
supported
contradicted
insufficient
```

最终答案从 `answer` 字段中抽取。如果 JSON 解析失败，则回退到通用答案归一化逻辑。

---

## 输出目录

运行输出统一写入：

```text
runs/qhl/m05c_HiKEY_domain/{run_id}/
```

主要文件：

```text
output/answer.csv
output/evidence.json
output/partial_results.json
output/ground_truth_comparison.json
output/ground_truth_comparison.csv
logs/run_summary.json
```

其中新增的两个评估文件是：

```text
ground_truth_comparison.json
ground_truth_comparison.csv
```

它们会比较 `answer.csv` 与：

```text
GPT-Pro-Answer/gpt_pro_all_answers.json
```

中的 `answer` 字段是否一致。

---

## 关于 top-k 召回率

用户要求计算“检索最终进模型的 top-k 中是否含有 ground truth”。当前 ground truth 文件只有答案字母，没有标准证据 span，因此无法判断“top-k 证据是否包含标准原文证据”。

所以 m05c 当前自动计算的是：

```text
final top-k evidence -> 模型答案 是否等于 ground truth answer
```

字段名为：

```text
topk_answer_hit
topk_answer_hit_rate
```

如果后续有 gold evidence span，可以进一步升级为真正的 evidence recall。

---

## 运行方式

默认全量运行：

```bash
python methods/m05c_HiKEY_domain/run.py
```

只跑某些领域：

```bash
python methods/m05c_HiKEY_domain/run.py --domains regulatory insurance
```

小样本调试：

```bash
python methods/m05c_HiKEY_domain/run.py --qid reg_a_001 reg_a_002 --no-resume
```

调整最终进模型证据数量：

```bash
python methods/m05c_HiKEY_domain/run.py --final-topk 10
```

调整领域卡片召回：

```bash
python methods/m05c_HiKEY_domain/run.py --domain-card-topk 12
```

---

## 当前版本定位

m05c 是领域化检索框架的第一版，重点是把代码结构搭好：

```text
可复用 HiKEY 缓存
可构建 DomainCard
可按领域召回
可压缩最终 top-k
可逐选项 verifier
可自动生成 ground truth 对比文件
```

当前策略结论：

```text
regulatory:
  强使用 DomainCard + verifier

financial_reports / financial_contracts / insurance / research:
  弱使用 DomainCard，主要保留 m05 的 HiKEY 证据优势
```

下一步如果要继续提升其他领域，需要分别加强它们的 DomainCard 抽取质量，而不是让粗卡片强行进入 top-k。
