# AFAC-4 项目协作手册

## 项目概述

AFAC2026 Task4 金融问答竞赛项目。本项目采用 **方法编号化 + 运行产物隔离 + 三维身份标识（coder / method / run_id）** 的工程框架，支持多人协作开发。

---

## 目录结构

```
AFAC-4/
├── run.sh                          # 统一启动入口
├── .env                            # API Keys（不入 git）
├── .gitignore
│
├── data/                           # 数据入口（软链接）
│   ├── raw/                        → public_dataset_upload/raw
│   └── questions/                  → public_dataset_upload/questions
│
├── methods/                        # 所有算法方案代码
│   ├── __init__.py                 # 使 methods.* 可导入
│   ├── _shared/                    # ⭐ 跨方法共享基础设施
│   │   ├── config_base.py          # 通用配置（PROJECT_ROOT, QUESTION_FILES, proxy, TOKEN_BUDGET...）
│   │   ├── parsers/                # 文档解析（PDF/HTML/TXT + 共享磁盘缓存）
│   │   │   ├── doc_resolver.py     # doc_id → file path 映射
│   │   │   ├── pdf_parser.py       # PDF 解析（pdfplumber + PyMuPDF fallback）
│   │   │   ├── html_parser.py      # HTML 解析（BeautifulSoup）
│   │   │   ├── txt_parser.py       # TXT 解析（按章节切分）
│   │   │   └── full_text.py        # 统一入口：parse_document_blocks / parse_document_fulltext
│   │   ├── indexer/                # 索引构建
│   │   │   ├── chunk.py            # Chunk 数据结构 + create_chunks()
│   │   │   └── bm25_index.py       # BM25 索引（jieba + rank_bm25）
│   │   ├── llm/                    # LLM 工具
│   │   │   ├── clients.py          # OpenAI client 工厂（dashscope/azure/openrouter）
│   │   │   └── normalizer.py       # 答案字母提取 + 后处理（normalize_answer）
│   │   ├── pipeline/               # 推理框架
│   │   │   ├── base_runner.py      # BaseRunner（并发推理 + resume + 进度 + 输出）
│   │   │   └── output.py           # answer.csv / evidence.json / run_summary.json 写入
│   │   ├── eval/compare.py         # 通用答案对比工具
│   │   ├── io_utils/runner.py      # RunContext + manifest 生成器
│   │   └── _cache/                 # 跨方法共享的解析缓存（自动生成）
│   │       ├── fulltext/           # 全文 txt 缓存
│   │       └── blocks/             # 结构化 blocks JSON 缓存
│   ├── m01_baseline_qwen/          # 方法 01：Qwen + BM25 + FC Skill v6
│   ├── m02_gpt41_golden/           # 方法 02：GPT-4.1 + BM25+Embedding RRF + 投票 + FC Skill v6
│   ├── m03_fulltext_baseline/      # 方法 03：全文输入 + Qwen-plus（无检索）
│   └── m04_xxx/                    # 新方法直接新建目录
│
├── runs/                           # 所有运行产物（与代码隔离）
│   └── {coder}/
│       └── {method_id}/
│           └── {run_id}/           # 每次运行的独立目录
│               ├── manifest.json   # 运行身份卡
│               ├── config.snapshot.json
│               ├── logs/
│               ├── intermediate/
│               └── output/
│
├── cache/                          # 跨 run 共享的重计算产物
│   ├── bm25_index/                 # BM25 索引
│   ├── embeddings/                 # Embedding 缓存
│   └── chunks/                     # 文档切分缓存
│
├── references/                     # 外部参考答案
│   └── gpt_pro/
│       ├── fin-c/parsed.json       # 结构化解析结果
│       └── parse_answers.py        # 解析脚本
│
├── analysis/                       # 一次性分析脚本（按日期归档）
│   └── {YYYYMMDD}_{主题}/
│
└── archive/                        # 被淘汰的旧代码/旧结果
```

---

## 命名规范

| 对象 | 格式 | 示例 |
|------|------|------|
| 方法目录 | `m{编号:02d}_{方法关键词}` | `m03_hybrid_rerank` |
| coder 短码 | 2-4 个小写字母 | `lgr`, `zw`, `lm` |
| run_id | `{YYYYMMDD-HHMMSS}_{coder}_{描述}` | `20260610-143000_lgr_fc-full` |
| 配置文件 | `config.yaml`（默认）/ `config.{实验}.yaml` | `config.no_rerank.yaml` |
| 分析脚本 | `analysis/{YYYYMMDD}_{主题}/` | `analysis/20260610_baseline_vs_gptpro/` |

---

## 快速开始

### 1. 环境准备

```bash
cd AFAC-4
pip install -r methods/m01_baseline_qwen/requirements.txt

# 确认 .env 中有 DASHSCOPE_API_KEY
cat .env
```

### 2. 运行已有方法

```bash
# 前台运行 m01（Qwen baseline）
./run.sh m01 --inference-only

# 后台运行 m02（GPT-4.1 golden）
./run.sh m02 --domains financial_contracts --bg

# 对比答案
./run.sh eval \
    --pred runs/lgr/m01_baseline_qwen/legacy_v1/output/answer.csv \
    --ref references/gpt_pro/fin-c/parsed.json \
    --domain fc
```

### 3. 查看运行结果

每次运行会在 `runs/{coder}/{method_id}/{run_id}/` 下生成：

```
manifest.json           ← 身份卡（who/when/what/model/config）
config.snapshot.json    ← 本次实际生效的配置快照
logs/stdout.log         ← 标准输出
logs/stderr.log         ← 错误输出
intermediate/           ← 中间产物（query plan、evidence）
output/answer.csv       ← 最终提交格式
output/answers.json     ← 结构化答案
```

---

## 开发新方法

### Step 1：创建方法目录

```bash
# 确定编号（查看 methods/ 下已有的最大编号 +1）
mkdir -p methods/m04_hybrid_rerank
```

### Step 2：编写代码（复用 _shared 基础设施）

方法目录最小结构（只需 3 个文件即可跑通）：

```
methods/m04_hybrid_rerank/
├── config.py               # 方法配置（复用 _shared.config_base）
├── pipeline.py             # 主流程（继承 BaseRunner）
└── run.py                  # 入口脚本
```

#### config.py 模板

```python
"""m04 配置 - 复用 _shared.config_base 的所有通用常量。"""
import os
import sys
from pathlib import Path

# 让 methods.* 可导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 继承所有通用配置（PROJECT_ROOT, RAW_BASE, QUESTION_FILES, DOMAINS, proxy, TOKEN_BUDGET...）
from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import PROJECT_BM25_INDEX_DIR  # noqa: F401

# ---- 方法特有配置 ----
METHOD_NAME = "m04_hybrid_rerank"
MODEL_NAME = "qwen-plus"  # 或 "gpt-4.1" 等
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

METHOD_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = METHOD_ROOT / "output"
LOGS_DIR = METHOD_ROOT / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = 5
MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.1
```

#### pipeline.py 模板

```python
"""m04 pipeline - 继承 BaseRunner，只需实现 process_question。"""
from typing import Dict

from methods._shared.llm import make_dashscope_client, normalize_answer
from methods._shared.parsers import parse_document_fulltext  # 或 parse_document_blocks
from methods._shared.indexer import BM25Index, Chunk, create_chunks
from methods._shared.pipeline import BaseRunner

from config import (
    CONCURRENCY, DASHSCOPE_API_KEY, LOGS_DIR, METHOD_NAME,
    MODEL_NAME, OUTPUT_DIR, MAX_OUTPUT_TOKENS, TEMPERATURE,
)


class MyPipeline(BaseRunner):
    method_name = METHOD_NAME
    model_name = MODEL_NAME

    def __init__(self):
        super().__init__(
            output_dir=OUTPUT_DIR,
            logs_dir=LOGS_DIR,
            concurrency=CONCURRENCY,
        )
        self.client = make_dashscope_client(api_key=DASHSCOPE_API_KEY)

    def setup(self, domains):
        """可选：加载索引、初始化 retriever 等。"""
        pass

    def process_question(self, q: Dict) -> Dict:
        """核心逻辑：接收一道题，返回答案 + token 统计。"""
        qid = q["qid"]
        # ... 你的检索 + 推理逻辑 ...
        return {
            "qid": qid,
            "answer": "A",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
```

#### run.py 模板

```python
#!/usr/bin/env python3
import argparse, logging, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (_HERE.parent.parent, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import DOMAINS, LOGS_DIR, METHOD_NAME
import config

def main():
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME}")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    if args.concurrency:
        config.CONCURRENCY = args.concurrency

    logging.basicConfig(level=logging.INFO)

    from pipeline import MyPipeline
    pipeline = MyPipeline()
    pipeline.run(domains=args.domains or DOMAINS, resume=not args.no_resume)
    pipeline.save_results()

if __name__ == "__main__":
    main()
```

### Step 3：注册到 run.sh

编辑项目根目录的 `run.sh`，在 `case` 语句中添加：

```bash
m04|m04_hybrid_rerank)
    METHOD_DIR="${PROJECT_ROOT}/methods/m04_hybrid_rerank"
    ;;
```

### Step 4：运行并评估

```bash
# 前台运行
./run.sh m04 --inference-only

# 对比 GPT-Pro 参考答案
./run.sh eval \
    --pred runs/lgr/m04_hybrid_rerank/{run_id}/output/answer.csv \
    --ref references/gpt_pro/fin-c/parsed.json \
    --domain fc
```

### Step 5：记录配置快照（推荐）

在 `run.py` 中接入 `RunContext`，自动生成 manifest：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from methods._shared.io_utils import create_run

ctx = create_run(
    coder="lgr",
    method_id="m04_hybrid_rerank",
    desc="fc-full",
    model_llm="qwen-plus",
    notes="first attempt with hybrid retrieval"
)

# 你的 pipeline 逻辑...
# results = pipeline.run(...)

# 运行结束后
ctx.finish(metrics={"fc_a": {"correct": 15, "total": 20, "acc": 0.75}})
```

---

## manifest.json 说明

每次运行的身份卡，自动记录关键信息：

```json
{
  "run_id": "20260610-143000_lgr_fc-full",
  "coder": "lgr",
  "method_id": "m03_hybrid_rerank",
  "git_commit": "a1b2c3d",
  "started_at": "2026-06-10T14:30:00+08:00",
  "finished_at": "2026-06-10T15:42:11+08:00",
  "status": "success",
  "model": {
    "llm": "qwen-plus",
    "embedding": "text-embedding-v3"
  },
  "config_hash": "sha256:3f2a...",
  "metrics": {
    "fc_a": {"correct": 15, "total": 20, "acc": 0.75}
  },
  "notes": "first attempt with hybrid retrieval"
}
```

---

## 共享资源

以下资源跨方法共享，**不要在方法目录中重复存放**：

| 路径 | 说明 |
|------|------|
| `data/raw/` | 原始文档（PDF/HTML/TXT），只读 |
| `data/questions/` | 题目 JSON 文件，只读 |
| `cache/bm25_index/` | BM25 索引（按 domain 分 pkl 文件，m01 构建，m01/m02 共用） |
| `cache/embeddings/` | Embedding 向量缓存（按 model 分子目录） |
| `methods/_shared/_cache/fulltext/` | PDF/HTML/TXT 全文解析缓存（自动生成，跨方法复用） |
| `methods/_shared/_cache/blocks/` | 结构化 blocks JSON 缓存（自动生成，跨方法复用） |
| `references/gpt_pro/` | GPT-Pro 参考答案 |
| `.env` | API Keys |

---

## _shared 基础设施速查

新方法可直接 import 以下模块，**不需要自己重新实现**：

| 模块 | 导入方式 | 提供的能力 |
|------|----------|------------|
| **config_base** | `from methods._shared.config_base import *` | PROJECT_ROOT, RAW_BASE, QUESTION_FILES, DOMAINS, TOKEN_BUDGET, compute_token_score, infer_domain_from_qid, proxy 自动配置 |
| **parsers** | `from methods._shared.parsers import ...` | `resolve_doc_path(doc_id, domain)` → 文件路径<br>`parse_document_blocks(doc_id, domain)` → List[{text, page_num, section_title}]<br>`parse_document_fulltext(doc_id, domain)` → 完整文本字符串<br>所有解析结果自动缓存到 `_shared/_cache/` |
| **indexer** | `from methods._shared.indexer import Chunk, create_chunks, BM25Index` | Chunk 数据结构 + 固定大小切分 + BM25 索引（build/save/load/search） |
| **llm.clients** | `from methods._shared.llm import make_dashscope_client, make_azure_client, make_openrouter_client` | OpenAI-compatible client 工厂（自动读 .env） |
| **llm.normalizer** | `from methods._shared.llm import normalize_answer` | 从 LLM 原始输出中提取 A/B/C/D 字母，支持 mcq/multi/tf 三种题型 |
| **pipeline.BaseRunner** | `from methods._shared.pipeline import BaseRunner` | 继承后只需实现 `process_question(q) → dict`，自动获得：并发推理、tqdm 进度、partial_results resume、answer.csv + evidence.json + run_summary.json 输出 |
| **pipeline.output** | `from methods._shared.pipeline import save_answer_csv, save_run_summary` | 单独调用输出函数（不继承 BaseRunner 时使用） |
| **eval** | `from methods._shared.eval.compare import compare_answers` | 答案对比 |
| **io_utils** | `from methods._shared.io_utils import create_run` | RunContext + manifest 生成 |

### BaseRunner 生命周期

```
pipeline.run(domains, resume=True)
    ├── setup(domains)              ← 可选 hook：加载索引、初始化 retriever
    ├── load_questions(domains)     ← 自动从 QUESTION_FILES 加载
    ├── _load_partial()             ← 自动恢复上次中断的进度
    └── ThreadPoolExecutor
        └── process_question(q)     ← 你的核心逻辑（必须实现）
            返回 {qid, answer, prompt_tokens, completion_tokens, total_tokens, ...}

pipeline.save_results()
    ├── answer.csv                  ← 提交格式
    ├── evidence.json               ← 可选
    └── run_summary.json            ← token 统计 + 分 domain 统计
```

---

## 答案对比工具

内置通用对比工具，支持 CSV 和 JSON 格式：

```bash
# 基本用法
./run.sh eval --pred <预测文件> --ref <参考文件> --domain <领域>

# 示例
./run.sh eval \
    --pred runs/lgr/m01_baseline_qwen/legacy_v1/output/answer_v2.csv \
    --ref references/gpt_pro/fin-c/parsed.json \
    --domain fc

# 输出 JSON 报告
./run.sh eval \
    --pred output.csv \
    --ref references/gpt_pro/fin-c/parsed.json \
    --domain fc \
    --output analysis/20260610_xxx/report.json
```

领域代码：`fc`（金融合同）、`fr`（财报）、`ins`（保险）、`reg`（监管）、`res`（研报）

---

## Git 规范

### 提交到 git 的内容

✅ `methods/` 下的源代码和 `config.yaml`  
✅ `references/*/parsed.json`（结构化参考答案）  
✅ `.gitignore`、`.env`、本文档  
✅ `analysis/` 下的分析脚本  

### 不提交的内容（已在 .gitignore 中配置）

❌ `runs/`（运行产物，每次自动生成）  
❌ `cache/`（大文件：索引、embedding）  
❌ `archive/`（旧代码备份）  
❌ `data/raw/`（原始数据集）  
❌ `*.pkl`、`*.npy`、`*.log`、`__pycache__/`  

### Commit 格式

```
[coder] type: description

# type: feat / fix / refactor / exp / doc
# 示例:
[lgr] feat: add m03_hybrid_rerank method
[lgr] exp: tune chunk_size from 800 to 1200
[zw] fix: multi-select answer normalization
```

---

## 注意事项

1. **不要在方法目录中存放运行产物**（logs/output/indices），统一走 `runs/` 和 `cache/`
2. **不要创建临时分析脚本散落在代码目录**，统一放 `analysis/{日期}_{主题}/`
3. **新增 coder** 只需在运行时传入新的 coder 短码，目录会自动创建
4. **新增方法** 只需新建 `methods/m{N}_{name}/` + 注册到 `run.sh`
5. **Token 预算很重要**，每次运行后检查 manifest 中的 token 统计
6. **答案格式严格**，提交前务必用 eval 工具验证
