# m07_section_routed_HiKEY 方法说明

`m07_section_routed_HiKEY` 是在当前表现最好的 `m05_HiKEY_question_option` 基础上继续优化检索召回的一版设计。核心目标不是替换 m05 的题干 + 选项检索，而是在它前面增加一个 **章节级路由 / 排序层**，让后续 BM25 检索优先发生在更可能包含答案的章节子树中。

这一版的设计参考了 `papers/IntrAgent.pdf` 中的关键思想：长文档问答不应只做平铺 chunk 相似度检索，而应先利用文档层级结构判断哪些章节最可能包含答案，再在这些章节中逐步读取和抽取细节。

---

## 1. 背景判断

当前比赛的正确性主要受两类问题制约：

```text
1. 文档处理：文档是否被解析成可定位、可追溯、可检索的结构。
2. 检索召回：正确证据是否进入 top-k evidence pack。
```

目前文档处理已经采用 HiKEY 树状层级结构：

```text
DocCard
SectionCard
EvidenceUnit
FieldCard
ancestry / siblings
```

这个方向是正确的。金融长文本中的答案通常不是孤立段落，而是依赖章节路径、条款层级、表格上下文、字段单位和相邻说明。

因此，m07 的优化重点放在检索侧。原因是：

```text
如果 top-k 中不包含正确证据，模型几乎不可能稳定答对。
如果正确证据已经进入 top-k，后续还可以通过 prompt、verifier、规则聚合继续优化。
```

m05 已经通过“题干 + 逐选项检索”显著改善了 evidence recall，但它的检索仍然主要是平铺的 BM25 / 关键词检索。m07 希望进一步利用 HiKEY 已有的章节树结构，减少正确证据被低相关段落挤出 top-k 的情况。

---

## 2. IntrAgent 对我们的启发

IntrAgent 的检索流程不是传统 RAG 的：

```text
query -> embedding / BM25 -> top-k chunks -> answer
```

而是两阶段：

```text
1. Section Ranking
   根据问题和文档章节层级，先判断哪些章节最可能包含答案。

2. Iterative Reading
   按排序后的章节依次读取，抽取细节，并做信息充分性检查。
```

论文中的关键结论是：在结构化长文档里，基于章节标题和层级的 reasoning-based section ranking 明显优于简单相似度检索。其章节定位实验中，IntrAgent 的 reasoning-based section ranking top-3 达到 94.6%，而相似度检索 top-3 只有 50.0%。

这对我们的启发是：

```text
金融文档同样是强结构文档。
不要让 BM25 在全局平铺文本中盲搜。
应先用章节结构做候选区域路由，再在候选章节内精搜证据。
```

不过，完整 IntrAgent 会引入较多 LLM 调用，包括章节排序、逐章节 detail extraction、sufficiency check 和最终聚合。对比赛来说，token 成本和运行时间都偏高。

因此 m07 只借鉴其中最适合落地的一部分：

```text
章节级路由 / 排序
+ m05 的题干 + 逐选项 BM25 精搜
+ 轻量充分性判断 / fallback
```

---

## 3. m07 核心流程

m07 的整体流程如下：

```text
HiKEY 缓存加载
-> 题目解析
-> 章节级候选召回
-> 章节级重排 / 路由
-> 在候选章节子树内执行 m05 题干 + 逐选项检索
-> 证据合并、去重、重排
-> 证据充分性轻量检查
-> 必要时全局 BM25 fallback
-> Qwen 作答
-> 答案格式校验
```

与 m05 的区别主要在检索前半段。

### m05

```text
题干
题干 + 选项 A
题干 + 选项 B
题干 + 选项 C
题干 + 选项 D
-> 全局 BM25 检索 unit / field_card
-> 合并 top-k
```

### m07

```text
题干 / 选项
-> 先找 top-M 章节或章节子树
-> 在 top-M 章节范围内执行题干 + 选项检索
-> 如果证据不足，再退回 m05 的全局检索
```

---

## 4. 章节级路由设计

章节级路由的输入：

```text
question
options
answer_format
domain
target_doc_ids
HiKEY sections
doc_card.top_sections
section_path
section title
section text_preview
```

章节级路由的输出：

```json
{
  "candidate_sections": [
    {
      "doc_id": "xxx",
      "section_id": "xxx",
      "section_path": "xxx > xxx",
      "score": 12.34,
      "source": ["question", "option_A", "domain_prior"]
    }
  ],
  "routing_mode": "section_limited | global_fallback"
}
```

### 4.1 第一阶段：规则 + BM25 章节召回

先不直接调用 LLM，而是用本地方法召回一批候选章节：

```text
query = 题干
query = 题干 + 各选项
query = 题干 + 全部选项关键词
```

检索对象只使用 section 级字段：

```text
section_path
title
text_preview
doc title
top_sections
```

打分信号包括：

```text
BM25(section title/path/preview)
指标别名命中
年份命中
公司/产品/法规名称命中
条款号命中
义务词命中
领域 prior 命中
选项关键词命中
```

这一阶段建议召回：

```text
SECTION_CANDIDATE_TOPK = 20
```

### 4.2 第二阶段：轻量章节重排

章节重排有两种实现路径。

#### 路径 A：纯本地重排

优点是零 token、速度快、稳定可复现。适合作为默认版本。

```text
section_score =
  bm25_score
  + metric_boost
  + year_boost
  + entity_boost
  + domain_prior_boost
  + option_hit_boost
```

#### 路径 B：Qwen 章节 reasoning rerank

只把候选章节的标题、路径和很短 preview 给 Qwen，让它判断哪些章节最可能包含答案。

输入控制在较小范围：

```text
题干
选项
候选章节列表 top-20
每个章节只给 section_path/title/text_preview 前 100-200 字
```

输出：

```text
按相关性排序的 section_id 列表
```

建议只在高风险题触发：

```text
regulatory 题
multi 题
题干/选项包含 否定、除外、不得、全部、均、至少、超过、不低于 等词
BM25 章节分数分散、top1 不明显
```

这样可以借鉴 IntrAgent 的 reasoning-based section ranking，同时控制 token 成本。

---

## 5. 章节子树内精搜

确定 top-M 章节后，在这些章节及其子节点中执行 m05 的原有检索：

```text
题干
题干 + 选项 A
题干 + 选项 B
题干 + 选项 C
题干 + 选项 D
```

但检索范围限制为：

```text
section_id in routed_sections
or section_path startswith routed_section_path
```

检索对象仍然包括：

```text
section
unit
field_card
```

这样可以保留 m05 的核心优势：选项级 evidence recall。

建议参数：

```text
ROUTED_SECTION_TOPM = 5
ROUTED_GLOBAL_TOPK = 8
ROUTED_OPTION_TOPK = 8
ROUTED_FINAL_EVIDENCE_TOPK = 24
ROUTED_MAX_SIBLINGS = 8
```

如果章节路由置信度较低，则提高 top-M：

```text
ROUTED_SECTION_TOPM = 8 或 10
```

---

## 6. 证据合并与重排

m07 继续复用 m05 的多路合并思想：

```text
按 anchor id 去重
记录命中来源：题干 / 选项A / 选项B / ...
多 query 命中的证据加权
```

但新增章节路由信号：

```text
final_score =
  anchor_best_score
  + query_hit_boost
  + section_rank_boost
  + field_card_boost
  + exact_numeric_hit_boost
```

其中：

```text
section_rank_boost = 章节排名越靠前，加分越高
field_card_boost = 对 financial_reports / financial_contracts 中的字段卡加分
exact_numeric_hit_boost = 选项中的数字、年份、百分比在证据中精确命中时加分
```

---

## 7. 轻量充分性检查

IntrAgent 的 iterative reading 里有 sufficiency check。m07 不建议完整复刻，但可以做轻量版本。

在本地层面检查：

```text
1. 每个选项是否至少命中 1 条 option-source evidence。
2. 题干中的年份、公司、产品、法规名称是否出现在 evidence pack。
3. 选项中的关键数字、百分比、期限是否出现在 evidence pack。
4. 多选题是否存在只有单个选项有证据的情况。
5. regulatory 题是否命中义务词和例外词所在句。
```

如果不满足，则触发 fallback：

```text
1. 扩大章节 top-M
2. 增加 option_topk
3. 回退到 m05 全局 BM25
4. 对高风险题触发 Qwen section rerank
```

这一步的目的不是判断答案，而是判断“证据是否可能足够”。它不应直接替代模型推理。

---

## 8. 分领域策略

### 8.1 financial_reports

优先路径：

```text
章节路由 -> FieldCard 精确召回 -> unit/table_row 补上下文
```

重点信号：

```text
营业收入
归母净利润
扣非归母净利润
经营现金流
研发投入 / 研发费用
现金分红 / 每10股
同比 / 增长 / 下降
单位
年份
```

### 8.2 regulatory

优先路径：

```text
章节 / 条款路由 -> 条文句子召回 -> 例外和前后款补充
```

重点信号：

```text
应当 / 可以 / 不得 / 禁止 / 应
除外 / 但是 / 前款规定 / 另有规定
期限
主体范围
监管机构
条 / 款 / 项 / 章
```

regulatory 是当前 m05 明显弱项，m07 应优先在这个领域验证章节路由收益。

### 8.3 insurance

优先路径：

```text
产品/责任章节路由 -> 责任、免责、等待期、给付规则精搜
```

重点信号：

```text
保险责任
责任免除
等待期
给付比例
身故保险金
现金价值
退保
领取规则
```

### 8.4 financial_contracts

优先路径：

```text
发行条款 / 评级 / 募集资金 / 违约责任章节路由
-> 字段和条款精搜
```

重点信号：

```text
发行规模
发行人
主体评级 / 债项评级
期限
利率
回售 / 赎回
担保
受托管理人
募集资金用途
```

### 8.5 research

优先路径：

```text
结论 / 图表 / 公司比较 / 趋势章节路由
-> 图表标题、图注、表格行和结论句精搜
```

重点信号：

```text
公司名
行业名
指标名
图 / 表
同比 / 环比 / CAGR
预测 / 假设
结论适用范围
```

---

## 9. 建议实现模块

建议新增方法目录：

```text
methods/m07_section_routed_HiKEY/
```

文件结构：

```text
config.py
pipeline.py
run.py
section_router.py
```

### 9.1 section_router.py

负责：

```text
加载 HiKEY sections
构建 section-level docs
执行章节 BM25 / 规则召回
可选 Qwen rerank
输出 candidate_sections
```

核心接口：

```python
class SectionRouter:
    def route(self, q: dict, retrievers: dict, doc_cards: dict) -> SectionRouteResult:
        ...
```

### 9.2 pipeline.py

基于 m05 改造：

```text
process_question()
  -> section_router.route()
  -> _retrieve_question_option_evidence(..., routed_sections=...)
  -> sufficiency_check()
  -> fallback if needed
  -> Qwen answer
```

### 9.3 config.py

新增参数：

```text
M07_ENABLE_SECTION_ROUTING = 1
M07_ENABLE_LLM_SECTION_RERANK = 0
M07_SECTION_CANDIDATE_TOPK = 20
M07_ROUTED_SECTION_TOPM = 5
M07_ROUTE_CONFIDENCE_THRESHOLD = 0.15
M07_ENABLE_GLOBAL_FALLBACK = 1
M07_ENABLE_LIGHT_SUFFICIENCY_CHECK = 1
```

---

## 10. 评估方式

m07 的评估不应只看最终 accuracy，还要单独评估检索召回。

建议记录：

```text
section_route_topm
section_route_scores
section_route_source
是否触发 llm_rerank
是否触发 global_fallback
每个 query 的原始 top-k
最终 evidence pack 中每条证据来源
每个选项是否有 evidence 命中
```

如果有参考答案或人工标注证据，应统计：

```text
正确证据是否在 top-5 / top-10 / top-20 / top-50
正确证据是否在 routed_sections 中
错误题中有多少是路由失败
错误题中有多少是章节命中但 unit/field_card 排名失败
错误题中有多少是证据命中但模型误判
```

这能帮助判断下一步该优化：

```text
section routing
unit ranking
field_card recall
prompt / verifier
```

---

## 11. 风险与控制

### 风险 1：章节路由错误导致召回范围过窄

控制方式：

```text
默认保留 global fallback
top-M 不宜太小
章节分数不明显时自动扩大 top-M
高风险题触发 Qwen rerank
```

### 风险 2：LLM section rerank 增加 token 消耗

控制方式：

```text
默认关闭 LLM rerank
只给 section_path/title/short preview
只在 regulatory / multi / low-confidence routing 等场景触发
```

### 风险 3：章节标题不规范

控制方式：

```text
不仅看 title，也看 section_path 和 text_preview
保留 BM25 全局 fallback
使用 doc_card.top_sections 辅助召回
```

---

## 12. MVP 建议

第一版 m07 不需要完整 agent loop。建议先做最小可验证版本：

```text
1. 复制 m05 为 m07。
2. 新增 SectionRouter，本地 BM25 + 规则召回 top-M section。
3. 修改 SimpleRetriever / pack_query 支持 section_id 或 section_path 过滤。
4. 在 routed_sections 中执行题干 + 选项检索。
5. 若最终证据条数太少或覆盖不足，回退 m05 全局检索。
6. 记录 route_meta 到 partial_results.json。
7. 先在 regulatory 和 financial_reports 上对比 m05。
```

预期收益：

```text
提高正确证据进入 top-k 的概率
减少无关章节噪声
改善 regulatory / insurance 这类强章节结构题
保持 m05 的选项级召回优势
token 成本可控
```

---

## 13. 小结

m07 的定位是：

```text
m05 的结构感知检索增强版。
```

它不推翻 m05，而是在 m05 的题干 + 逐选项 BM25 之前增加章节级路由，让检索从：

```text
全局平铺搜索
```

升级为：

```text
先找可能章节，再在章节子树中精搜证据。
```

这正好利用了当前 HiKEY 文档处理已经建立好的树状层级结构，也吸收了 IntrAgent 中最适合比赛落地的部分：**section ranking before detail retrieval**。

如果 m07 成功，后续可以继续演化为：

```text
m08: section-routed + option-level verifier
m09: section-routed + field lookup + calculator
m10: section-routed + adaptive token budget
```

