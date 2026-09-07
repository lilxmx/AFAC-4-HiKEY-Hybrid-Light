HiKEY 可以理解成：**把 PDF 先整理成“目录树”，再按“文档 → 章节 → 证据块”逐级缩小范围，最后把答案所需的局部证据打包给 Reader。**

普通 RAG 是：

```text
PDF → 切 chunk → embedding 检索 top-k chunk → LLM 回答
```

HiKEY 是：

```text
PDF → 恢复文档层级树
→ 建 Doc_card / Sec_card
→ 先找相关文档
→ 再找相关章节和证据单元
→ 按层级关系打包证据
→ Reader 回答
```

论文认为普通 chunk RAG 的两个核心问题是：**routing failure**，也就是在大量文档中先找错文档/章节；以及 **evidence fragmentation**，也就是答案证据散落在正文、表格、图片、标题路径、脚注里，top-k chunk 拼起来仍然不完整。HiKEY 的核心不是简单加 VLM，而是把“文档层级”当成检索信号本身。

---

## 0. 先把几个名词说清楚

HiKEY 里最关键的对象有 6 个：

| 名词                | 含义                 | 在年报赛题里的类比                         |
| ----------------- | ------------------ | --------------------------------- |
| `document d`      | 一份 PDF             | 比亚迪 2025 年报                       |
| `T(d)`            | 这份 PDF 的层级树        | 年报目录树                             |
| `unit c`          | 最小证据单元，可以是文本、表格、图片 | 一段正文、一张表、一行表格、一个图                 |
| `section_path(c)` | 某个证据单元所属的标题路径      | `第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标` |
| `Doc_card`        | 文档级检索卡片            | 用来判断“这是不是我要找的年报”                  |
| `Sec_card`        | 章节级检索卡片            | 用来判断“这份年报里哪一节最相关”                 |

论文里的定义是：每份文档会被解析成一棵 hierarchy tree，每个 section 下面有一组 multimodal units，包括 text、table、image；同时建立两类 index card：`Doc_card` 表示全局文档上下文，`Sec_card` 表示章节内的多模态证据单元。

---

# 第一步：离线阶段，给 PDF 建“层级树 + 检索卡片”

这一步不是回答问题时做的，而是**预处理阶段**做的。论文叫：

```text
Offline Hierarchy-Aware Graph and Index Construction
```

也就是：先把原始 PDF 转成结构化图和索引。

## 1.1 DHP：把 PDF 解析成树

HiKEY 用一个叫 **DHP，Document Hierarchical Parsing** 的模块，把 PDF 里的 layout blocks 解析成层级树。layout blocks 包括：

```text
标题 heading
段落 paragraph
caption
表格 table
图片 figure
```

每个 block 都变成树上的一个节点，节点之间有父子关系。比如一份年报可以抽象成：

```text
比亚迪股份有限公司 2025 年年度报告
├── 第一节 重要提示、目录和释义
├── 第二节 公司简介和主要财务指标
│   ├── 一、公司信息
│   ├── 六、主要会计数据和财务指标
│   │   ├── 表格：营业收入、净利润、现金流量净额
│   │   └── 表格脚注/单位说明
├── 第三节 管理层讨论与分析
│   ├── 一、报告期内公司所处行业情况
│   ├── 二、报告期内公司从事的主要业务
└── 第八节 财务报告
    ├── 合并资产负债表
    ├── 合并利润表
    └── 合并现金流量表
```

论文明确说，每个 block，例如 text、table、figure，都会成为节点，通过 parent-child tree edges 连接，并获得一个 ancestor section path，例如 `Title > Section 5 > 5.3`。

这一步的意义是：后面检索到一个数字时，系统不只知道“这个数字在哪一页”，还知道它属于哪个文档、哪个章节、哪个表格、哪个标题路径。

---

## 1.2 给每个证据单元挂 section_path

HiKEY 不把标题路径当成普通 metadata 随便存一下，而是把它作为后续检索和打包的核心字段。

它的做法是：对每个 evidence unit，从当前节点往树上找最近的 Title 或 Section Header，把从根节点到这个 header 的标题序列串起来，形成：

```text
section_path(c) = Title > Sec > Subsec > ...
```

论文说这个 section path 会作为 structural metadata 存储，并用于 hierarchy field index 和 graph traversal。

用年报举例：

```json
{
  "unit_id": "midea_2025_p9_table_row_4",
  "type": "table_row",
  "content": "经营活动产生的现金流量净额 53,345,930 60,511,572 -11.84%",
  "section_path": "美的集团股份有限公司2025年年度报告 > 第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标",
  "page": 9
}
```

普通 chunk RAG 可能只保存：

```text
经营活动产生的现金流量净额 53,345,930 60,511,572 -11.84%
```

HiKEY 会保存：

```text
这是哪个公司、哪份年报、哪一节、哪张表、哪个字段、上下文是什么
```

这就是它和普通 chunking 最大的区别之一。

---

## 1.3 建 Doc_card：用于“先找对文档”

`Doc_card` 是文档级索引卡片。它不是全文，也不是摘要，而是一个轻量的文档路由表示。

论文里的理论定义大致是：

```text
Doc_card = 文档标题 + 所有 section paths
```

实际构造时，论文说它会拼接：检测到的 Title、高层级 section headers，通常是 1–2 级标题，以及可识别的目录 ToC entries；最后截断到固定长度，避免 Doc_card 太长。这样做的目的，是让 Stage-1 routing 有全局主题信号，又不被全文 OCR 噪声和局部无关细节干扰。

年报场景里，一个 Doc_card 可以长这样：

```text
doc_id: byd_2025

title:
比亚迪股份有限公司 2025 年年度报告

hierarchy:
第一节 重要提示、目录和释义
第二节 公司简介和主要财务指标
第三节 管理层讨论与分析
第四节 公司治理、环境和社会
第五节 重要事项
第六节 股份变动及股东情况
第七节 债券相关情况
第八节 财务报告
```

它的作用是回答：

```text
这道题应该去哪个 PDF 里找？
```

例如问题是：

```text
比亚迪 2025 年营业收入是多少？
```

Stage-1 不应该在所有 chunk 里直接搜“营业收入”，而是先通过 `比亚迪 + 2025 + 年度报告 + 章节路径` 找到 `byd_2025` 这份文档。

---

## 1.4 建 Sec_card：用于“在文档里找对章节”

`Sec_card` 是章节级索引卡片。它包含某个 section 下的正文、表格、图片等证据单元。

论文说，`Sec_card` 是 section-level card，里面包含该 section 的 text 和 non-text units，也就是 tables/figures，作为 searchable items。

年报场景里，一个 Sec_card 可以长这样：

```json
{
  "doc_id": "midea_2025",
  "section_id": "sec_2_6",
  "section_path": "第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标",
  "units": [
    {
      "type": "text",
      "content": "公司是否需追溯调整或重述以前年度会计数据..."
    },
    {
      "type": "table",
      "table_name": "主要会计数据和财务指标",
      "content": "| 项目 | 2025年 | 2024年 | 本年比上年增减 | 2023年 | ..."
    }
  ]
}
```

它的作用是回答：

```text
在这份 PDF 里，哪一节最可能包含答案？
```

---

## 1.5 给表格/图片补 Upper Context

这是 HiKEY 很实用的细节。

表格或图片经常没有清晰 caption。比如一个表格本身只写：

```text
2025年 2024年 本年比上年增减
456,451,731 407,149,600 12.11%
```

如果没有上方标题，你不知道这是营业收入、净利润还是总资产。

HiKEY 的做法是：对于缺 caption 的 Table/Image，沿着 DHP 树往上或往前找一个逻辑上位的文本节点，作为 `ctx(c)`，也就是 Upper Context，用来降低视觉/表格单元歧义。论文明确提到，对缺 caption 的 visual units，会通过 graph traversal 找 logically preceding textual node 作为 Upper Context。

在你的年报系统里，可以把它实现成：

```text
表格 ctx =
最近的章节标题
+ 最近的表格标题
+ 表格前 1–2 行说明
+ 单位行
+ 脚注
```

例如：

```json
{
  "type": "table",
  "content": "...经营活动产生的现金流量净额...",
  "ctx": "第二节 公司简介和主要财务指标；六、主要会计数据和财务指标；单位：千元"
}
```

这一步对财报题特别重要，因为很多错误不是“找不到数”，而是“数找到了，但单位、年份、口径错了”。

---

# 第二步：在线阶段，先找文档，再找章节/证据单元

这一步叫：

```text
Online Hierarchical Coarse-to-Fine Retrieval
```

也就是分两级检索：

```text
Stage-1：Document Routing
Stage-2：Section / Unit Retrieval
```

核心思想是：**不要一上来就在全库 chunk 里搜答案。先缩小文档范围，再在候选文档里细搜。**

---

## 2.1 Stage-1：文档级路由 Doc Routing

输入一个问题 `q`，HiKEY 先用 `Doc_card` 找候选文档。

论文里的文档级得分是：

```text
Sdoc(d, q)
= α · lexical_score(Doc_card, q)
+ (1 - α) · dense_text_score(Doc_card, q)
```

论文实际写法还包括 min-max normalization，即对每个 query 的分数做归一化。这个阶段通过 Doc_card 中的 hierarchy paths 过滤掉无关文档。

简单说就是：

```text
Doc_card 和问题做 BM25 / 关键词匹配
+
Doc_card 和问题做文本向量匹配
→ 得到候选文档
```

例如问题：

```text
美的集团 2025 年经营活动产生的现金流量净额同比是否下降？
```

Stage-1 的目标不是直接回答，而是得到：

```text
候选文档：
1. midea_2025
2. midea_2024
...
```

其中 `美的集团` 和 `2025 年年度报告` 更应该作为过滤/路由信号，而不是把全库中所有出现“经营活动”的 chunk 都拉出来。

---

## 2.2 Stage-2：章节级和证据单元级检索

Stage-1 找到候选文档后，HiKEY 再在这些文档内部检索 `Sec_card`。

这里不是简单给整个 section 打一个 embedding 分数，而是先给 section 里的每个 unit 打分。

论文定义了 type-specific unit score：

```text
如果 unit 是 Text：
s(c, q) = 文本混合检索分数

如果 unit 是 Table / Image：
s(c, q) = γ · 视觉相似度
        + (1 - γ) · Upper Context 文本相似度
```

然后 section 的分数不是平均值，而是取它下面最强 evidence unit 的分数：

```text
Ssec(s, q) = max over units in section
```

最后 section 总分融合文档级分数和章节级分数：

```text
Sfinal(s, q)
= λ · Sdoc(doc(s), q)
+ (1 - λ) · Ssec(s, q)
```

论文把这叫 Hierarchical Section MaxSim Scoring：在候选文档中，对所有 Sec_cards 做细粒度排序，并用不同 encoder/score 处理 text、table、image。

翻译成你的年报赛题逻辑就是：

```text
问题：
美的集团 2025 年经营活动产生的现金流量净额同比是否下降？

Stage-1：
找文档：美的集团 2025 年报

Stage-2：
找章节：
第二节 公司简介和主要财务指标
第八节 财务报告 > 合并现金流量表
第三节 管理层讨论与分析 > 现金流说明

找 unit：
主要会计数据表里的“经营活动产生的现金流量净额”这一行
合并现金流量表里的对应项目
```

注意，HiKEY 的 Stage-2 关注的是：

```text
哪个 section 下面的哪个 unit 最像答案证据？
```

不是：

```text
哪个 chunk 整体最像问题？
```

这是很关键的区别。

---

## 2.3 为什么要 Doc → Sec，而不是直接 Sec 全库搜？

因为直接在全库 section/chunk 搜，会有很多高频词干扰。

年报里所有公司都有这些词：

```text
营业收入
净利润
现金流量净额
报告期
年度报告
人民币
董事会
公司简介
财务报告
```

如果你直接全库搜“经营活动现金流量净额”，很容易把其他公司、其他年份、其他报表口径也搜出来。

HiKEY 的 ablation 显示，`Doc → Sec` 的 coarse-to-fine 路由优于单独 Doc-only 或 Sec-only；同时，把 hierarchy 作为单独字段，比只用正文或简单拼接标题更有效。论文报告的 Avg R@10 中，Body-only 是 81.2，Concat Title/Header 是 84.6，Field-separated Hierarchy 是 88.6；Doc-only 是 87.7，Sec-only 是 76.1，Doc → Sec 是 88.6。

对你的赛题，这意味着：

```text
公司名、年份、报告类型 → 先用于文档过滤
指标名、表格名、章节名 → 再用于章节/字段检索
```

不要让“营业收入”这种全库高频词一开始就主导检索。

---

# 第三步：不是把 top-k 原样塞给模型，而是组装 evidence subgraph

这是 HiKEY 最容易被忽略，但对 QA 很重要的一步。

普通 RAG 通常是：

```text
取 top-k chunk
按分数排序
拼进 prompt
```

HiKEY 是：

```text
取 top-ranked sections
找到 anchor unit
带上 ancestor headers
补 sibling units
再补 semantic associates
在 token budget 内序列化成证据子图
```

论文把这一步叫：

```text
Hierarchical Subgraph Assembly
```

目标是：**在有限 token budget 下，把最有解释力、最不容易误读的证据放进去。**

---

## 3.1 Anchor Unit：先放“命中答案的证据块”

Anchor Unit 是某个 section 里获得 MaxSim 分数的 unit。

比如问题是：

```text
美的集团 2025 年经营活动产生的现金流量净额同比是否下降？
```

Anchor Unit 可能是：

```text
表格中“经营活动产生的现金流量净额”这一行
```

而不是整页，也不是整张大表的全部内容。

论文说，对于每个 selected section，会插入 Anchor Unit，也就是取得 MaxSim 分数的 unit。

---

## 3.2 Governing Headers / Ancestry Context：必须带上标题路径

光有一行表格还不够。模型需要知道这行属于哪份文档、哪个章节、哪个表。

所以 HiKEY 会把 Anchor Unit 的 ancestral section titles 一起塞进去，叫 Governing Headers 或 Ancestry Context。论文说这样做是为了 preserve logical coherence，避免 scope misinterpretation。

对年报来说，证据包应该长这样：

```text
Document:
美的集团股份有限公司 2025 年年度报告

Section Path:
第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标

Table:
主要会计数据和财务指标

Unit:
经营活动产生的现金流量净额：
2025 年 53,345,930
2024 年 60,511,572
本年比上年增减 -11.84%

Unit:
金额单位：千元
```

这比只给模型：

```text
经营活动产生的现金流量净额 53,345,930 60,511,572 -11.84%
```

稳定得多。

---

## 3.3 Sibling Units：再补同一父章节下的兄弟证据

HiKEY 接下来优先补 **Sibling Units**，也就是和 Anchor Unit 共享同一 parent section 的证据单元。

论文说，Sibling Units 是 structural association，优先加入和 anchor 处于同一 parent section 的 tables 和 figures，用来保留局部上下文和 visual-to-text alignment。

在年报里，Sibling Units 可以是：

```text
同一张表的表头
同一张表的单位行
同一张表的上一年列
同一张表的本年比上年增减列
同一小节的脚注
同一小节的说明段落
```

例如你问同比下降，不能只给 2025 年值，还要给：

```text
2024 年值
同比增减百分比
单位
表格标题
```

所以 Sibling Units 对财务题非常重要。

---

## 3.4 Semantic Associates：预算够时，再补跨章节相关证据

Sibling Units 解决“局部同一小节”的证据，但有些问题需要跨章节。

比如：

```text
公司经营活动现金流下降的原因是什么？
```

答案可能需要：

```text
第二节主要财务指标里的现金流数据
+
第三节管理层讨论与分析里的原因解释
+
第八节现金流量表里的明细
```

这时 HiKEY 会加入 **Semantic Associates**：和 anchor 在语义向量上相似、但不一定在同一章节的证据单元。

论文说，如果 token budget 允许，会继续加入与 anchor 有高向量相似度的 non-local visual units，以捕获跨章节依赖；structural association 和 semantic association 是互补的，前者恢复同一 subtree 下的共同引用证据，后者恢复布局没有显式连接但主题相关的证据。

用年报说：

```text
Anchor:
第二节主要财务指标中的“经营活动产生的现金流量净额”表格行

Sibling:
同表的 2025 年、2024 年、同比变化、单位

Semantic Associate:
第三节管理层讨论与分析中关于现金流变动原因的段落
第八节合并现金流量表中的经营活动现金流明细
```

这就是 HiKEY 的“证据子图”。

---

## 3.5 最终序列化给 Reader 的格式

论文附录说，每个 selected evidence unit 会带着：

```text
Ancestry Context:
文档标题 + governing headers

Unit Metadata:
unit id、unit type、source page

Content:
OCR 文本、linearized table、caption

Visual Crop:
Table/Figure 的图像裁剪，如果 LVLM budget 允许
```

并且它不是 naive greedy packing，而是先放高分 anchor，再强制附上 ancestry，然后扩展 sibling units 和 semantic associates，在固定 token budget 内形成 coherent subgraph。

对你的比赛，即使不能用 VLM，也可以把 Visual Crop 去掉，保留：

```text
文档标题
章节路径
页码
表格 markdown
表头
单位
行名
数值
脚注
相邻行
跨章节相关段落
```

也就是说，**packing 思想完全可以文本化实现**。

---

# 用一个完整例子串起来

问题：

```text
美的集团 2025 年经营活动产生的现金流量净额同比是否下降？
```

普通 RAG 可能这样做：

```text
query embedding
→ 全库搜 top-k chunk
→ 找到几个包含“经营活动产生的现金流量净额”的片段
→ LLM 自己判断
```

HiKEY 风格会这样做：

## 离线已建好

```json
{
  "doc_id": "midea_2025",
  "doc_card": {
    "title": "美的集团股份有限公司 2025 年年度报告",
    "hierarchy": [
      "第二节 公司简介和主要财务指标",
      "第三节 管理层讨论与分析",
      "第八节 财务报告"
    ]
  }
}
```

```json
{
  "sec_id": "midea_2025_sec_2_6",
  "section_path": "第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标",
  "units": [
    {
      "type": "table",
      "table_name": "主要会计数据和财务指标",
      "content": "经营活动产生的现金流量净额：2025年 53,345,930；2024年 60,511,572；本年比上年增减 -11.84%",
      "unit": "千元"
    }
  ]
}
```

## 在线检索

```text
Stage-1 Doc Routing:
问题命中 美的集团 + 2025 年年度报告
→ 候选文档 midea_2025

Stage-2 Section Scoring:
问题命中 经营活动产生的现金流量净额 + 同比
→ 候选章节 第二节 公司简介和主要财务指标
→ Anchor Unit = 主要会计数据表中的对应表格行
```

## Evidence Packing

```text
Anchor:
经营活动产生的现金流量净额这一行

Ancestry:
美的集团股份有限公司 2025 年年度报告
第二节 公司简介和主要财务指标
六、主要会计数据和财务指标

Sibling:
表头：2025 年、2024 年、本年比上年增减
单位：千元

Semantic Associate:
如果题目还问原因，再加入管理层讨论中的现金流解释段落
```

## Reader 得到的上下文

```text
[Evidence 1]
doc_id: midea_2025
page: 9
section_path: 第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标
table_name: 主要会计数据和财务指标
unit: 千元
content:
经营活动产生的现金流量净额：
2025 年 53,345,930；
2024 年 60,511,572；
本年比上年增减 -11.84%。

[Answer Logic]
53,345,930 < 60,511,572，且同比为 -11.84%，所以同比下降。
```

这就是 HiKEY 的真正价值：**不是多找几个 chunk，而是把“答案行 + 表头 + 单位 + 章节路径 + 相关说明”作为一个证据结构交给模型。**

---

# 和普通 RAG 的差异

| 环节     | 普通 chunk RAG     | HiKEY                                              |
| ------ | ---------------- | -------------------------------------------------- |
| PDF 表示 | 平铺 chunk         | 文档层级树                                              |
| 最小检索单元 | chunk / page     | text/table/image block                             |
| 文档选择   | 全库 chunk 相似度隐式决定 | 先用 Doc_card 显式路由                                   |
| 章节选择   | top-k chunk      | Sec_card + unit MaxSim                             |
| 表格处理   | 表格经常被切碎          | 表格作为 first-class unit                              |
| 上下文    | chunk 前后窗口       | section_path + governing headers                   |
| 证据打包   | top-k 贪心拼接       | anchor + ancestry + siblings + semantic associates |
| 目标     | 找相似文本            | 找可解释、可组合的证据子图                                      |

论文也总结了这些差异：HiKEY 用 block-level units、hierarchical routing 和 ancestry-aware packing 解决 chunk RAG 文档级线索弱、page-level RAG 噪声大、text GraphRAG 无法把表格/图片当一等证据的问题。

---

# 对你的赛题，最推荐借鉴的不是 VLM，而是这 4 个结构

## 1. 年报目录树

先把每份年报转成：

```text
doc_id
company
year
section_path
page
block_type
text/table
```

这是 HiKEY 的 DHP 思想。

## 2. Doc_card

用于先找文档：

```json
{
  "doc_id": "byd_2025",
  "company": "比亚迪",
  "year": 2025,
  "title": "比亚迪股份有限公司 2025 年年度报告",
  "hierarchy": [
    "第二节 公司简介和主要财务指标",
    "第三节 管理层讨论与分析",
    "第八节 财务报告"
  ]
}
```

## 3. Sec_card / Field_card

用于找章节和字段：

```json
{
  "doc_id": "byd_2025",
  "section_path": "第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标",
  "table_name": "主要会计数据和财务指标",
  "units": [
    {
      "row_header": "营业收入",
      "col_header": "2025年",
      "value": "...",
      "unit": "千元"
    }
  ]
}
```

## 4. Evidence Pack

最终给模型的证据不要是裸文本，而是：

```text
doc_id:
company:
year:
page:
section_path:
table_name:
unit:
anchor:
siblings:
semantic_associates:
```

这就是文本版 HiKEY。

---

# HiKEY 的局限也要注意

HiKEY 依赖上游 OCR 和 layout parsing。如果初始解析把标题识别错、层级挂错，section_path 也会错，后续 routing 和 packing 都可能受影响。论文也明确说，初始解析错误会传播到错误 section paths，低质量扫描件、手写、异常版式会影响鲁棒性；而且 HiKEY 更适合有显式标题层级的文档，对收据、扁平文本、非结构化 slides 这类弱层级文档收益会下降。

不过年报天然有目录、章节、表格标题和财务报表，所以你的赛题非常适合做“文本版 HiKEY”。论文还强调 DHP 是离线一次性 indexing cost，不是在每次 query 时执行；建好树和 section paths 后，线上只需要 Stage-1 routing、Stage-2 scoring 和 Reader inference。

---

## 一句话总结

HiKEY 的三步可以这样记：

```text
第一步：把 PDF 变成树
文档 → 章节 → 小节 → 文本/表格/图片证据单元

第二步：按树检索
先用 Doc_card 找文档，再用 Sec_card 找章节和证据块

第三步：按树打包
不是 top-k chunk，而是 anchor + 标题路径 + 同章节兄弟证据 + 跨章节语义证据
```

对你的年报 QA 来说，最值得抄的是：

```text
Doc_card 做文档路由
Sec_card / Field_card 做字段检索
section_path 防止口径错
anchor + siblings 防止表头、单位、年份丢失
semantic associates 支持跨章节解释题
```

即使比赛不允许 VLM，这套结构仍然可以用纯文本、OCR、表格 markdown、BM25、文本 embedding 和规则计算器实现。
