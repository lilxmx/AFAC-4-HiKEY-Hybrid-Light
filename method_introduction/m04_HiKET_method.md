已经把 `hikey_financial_parser.py` 升级成 **HiKEY-Finance v2.0 比赛增强版**，并重新解析了 `annual_cscec_2025_report.pdf`。我直接改进了能在当前环境完成的部分；需要服务器/API/更强解析器的地方，已经在代码里用 `@TODO(server)` 标注。

下载：

[更新后的 hikey_financial_parser.py](sandbox:/mnt/data/hikey_financial_parser.py)

[完整 v2 代码与中国建筑 2025 解析结果包](sandbox:/mnt/data/hikey_finance_v2_cscec_2025_package.zip)

---

## 这版主要优化了什么

相比上一版，重点补强了你提到的“当前版本限制”：

### 1. 加入 VLM-ready 路径

代码新增：

```bash
prepare-vlm
render-pages
page_image_stub
```

它会自动标记适合 VLM fallback 的页面，并可以渲染成图片，生成给 Qwen-plus / Qwen-VL 使用的 JSONL manifest。

示例已生成：

[annual_cscec_2025_report_vlm_manifest.jsonl](sandbox:/mnt/data/hikey_cscec_2025_output_v2/annual_cscec_2025_report_vlm_manifest.jsonl)

示例渲染页面也已经放入输出包中，例如第 13、14、132 页，用于你后续接 Qwen-plus 做表格/图像抽取。

代码里已标注：

```python
@TODO(server): call Qwen-plus/Qwen-VL with these image paths and prompts,
then merge the returned table JSON or visual captions back into EvidenceUnits / FieldCards.
```

---

### 2. 新增 FieldCard 缓存

这是最重要的比赛级增强。

现在除了：

```text
DocCard
SectionCard
EvidenceUnit
```

还新增了：

```text
FieldCard
```

也就是把财务字段进一步结构化成：

```json
{
  "metric": "营业收入",
  "value_map": {
    "v1": "2,082,141,811",
    "v2": "2,187,334,286",
    "v3": "2,187,147,839",
    "v4": "-4.8",
    "v5": "2,265,529,244"
  },
  "unit": "千元",
  "source_page": 13,
  "section_path": "第二节 公司简介和主要财务指标 > 七、近三年主要会计数据和财务指标 > （二）主要财务指标"
}
```

输出文件：

[FieldCards JSONL](sandbox:/mnt/data/hikey_cscec_2025_output_v2/annual_cscec_2025_report_field_cards.jsonl)

[FieldCards CSV](sandbox:/mnt/data/hikey_cscec_2025_output_v2/annual_cscec_2025_report_field_cards.csv)

这一步的意义是：比赛中的选择题/判断题，不应该全部走自由文本 RAG，而应该优先走：

```text
题目解析 → metric 映射 → FieldCard 精确核验 → 必要时 evidence pack
```

---

### 3. 加强 Query Planner

代码新增 `QueryPlanner`，现在会从问题中抽取：

```text
company
year
metric
operation
answer_format
```

例如：

```text
中国建筑2025年营业收入是多少
```

会抽出：

```json
{
  "companies": ["中国建筑股份有限公司"],
  "years": [2025],
  "metrics": ["营业收入"],
  "operations": []
}
```

这个信息会用于检索加权，不再只是 BM25 硬搜。

---

### 4. 加强金融指标同义词体系

代码里现在有 `METRIC_ALIASES`，包括：

```text
营业收入 / 营业总收入 / 营业额
归母净利润 / 归属于上市公司股东的净利润
扣非归母净利润
经营活动产生的现金流量净额 / 经营现金流
现金红利 / 现金分红 / 每10股派 / 股息
资本公积金转增股本
新签合同额
资产负债率
研发投入
发行规模
利率
到期日
```

这对比赛选项核验很重要，因为题干和年报原文经常用不同表述。

---

### 5. 新增 HiKEY-style Evidence Pack

现在检索可以返回：

```text
anchor
+ ancestry
+ siblings
```

也就是：

```text
命中的核心 unit
+ 所在 section_path
+ 同一 section 下的表头、单位、相邻字段、说明文本
```

示例输出：

[营业收入 evidence pack](sandbox:/mnt/data/hikey_cscec_2025_output_v2/search_pack_revenue.json)

[现金红利 evidence pack](sandbox:/mnt/data/hikey_cscec_2025_output_v2/search_pack_dividend.json)

运行方式：

```bash
python hikey_financial_parser.py search \
  --sections hikey_cscec_2025_output_v2/annual_cscec_2025_report_sections.jsonl \
  --units hikey_cscec_2025_output_v2/annual_cscec_2025_report_units.jsonl \
  --field_cards hikey_cscec_2025_output_v2/annual_cscec_2025_report_field_cards.jsonl \
  --query "中国建筑2025年营业收入是多少" \
  --topk 5 \
  --pack
```

---

## 重新解析结果

本地环境下，为了避免 `pdfplumber` 全量表格解析超时，我重新解析时使用了：

```bash
--no_tables
```

但代码本身仍然支持 `pdfplumber` 表格解析。服务器上建议去掉 `--no_tables` 运行，或者接入更强表格解析器。

当前 v2 解析结果：

```text
doc_id: annual_cscec_2025_report
company: 中国建筑股份有限公司
year: 2025
page_count: 370

sections: 399
units: 2097
field_cards: 454
```

unit 类型：

```text
text_block: 783
heading: 590
financial_field: 456
page_image_stub: 268
```

解析报告：

[annual_cscec_2025_report_parse_report.md](sandbox:/mnt/data/hikey_cscec_2025_output_v2/annual_cscec_2025_report_parse_report.md)

---

## 服务器上建议怎么跑

### 快速文本版

```bash
python hikey_financial_parser.py parse annual_cscec_2025_report.pdf \
  --out_dir ./hikey_cscec_2025_output_v2 \
  --no_tables
```

### 加表格解析版

```bash
python hikey_financial_parser.py parse annual_cscec_2025_report.pdf \
  --out_dir ./hikey_cscec_2025_output_v2
```

### 准备 VLM 页面

```bash
python hikey_financial_parser.py prepare-vlm annual_cscec_2025_report.pdf \
  --units ./hikey_cscec_2025_output_v2/annual_cscec_2025_report_units.jsonl \
  --out_dir ./hikey_cscec_2025_output_v2
```

### 指定渲染页面

```bash
python hikey_financial_parser.py render-pages annual_cscec_2025_report.pdf \
  --out_dir ./vlm_pages \
  --pages 13,14,132 \
  --zoom 2.0
```

---

## 代码中已标注的核心 TODO

主要包括四类：

```text
@TODO(server): 接 Qwen-plus / Qwen-VL API
@TODO(server): 替换轻量 lexical retriever 为 embedding + reranker
@TODO(server): 接入 MinerU / PaddleOCR PP-Structure / Docling / Camelot 等表格解析
@TODO(server): 加 option-level verifier，把选择题拆成 claim 后逐项核验
```

下一步最值得做的是 **Option Verifier**，因为比赛最终是单选/多选/判断。当前代码已经有 FieldCard 和 EvidencePack，后面只差：

```text
选项 A/B/C/D
→ 原子命题拆解
→ 每个命题检索 FieldCard / EvidencePack
→ supported / contradicted / insufficient
→ 聚合答案
```

这一步会比继续盲目优化 PDF parsing 更直接提升分数。
