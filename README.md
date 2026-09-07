# AFAC-4

An engineering-oriented retrieval-augmented generation (RAG) solution for the AFAC Task 4 competition. The repository contains several baselines and HiKEY-style variants for document parsing, indexing, evidence retrieval, reasoning, and answer validation across financial and regulatory domains.

## Repository layout

- `methods/` — runnable baselines and method variants (`m01`–`m08`) plus shared infrastructure.
- `method_introduction/` — short descriptions of the implemented methods.
- `GPT-Pro-Answer/` — structured reference-answer utilities and examples.
- `COLLABORATION.md` — detailed development and experiment workflow.
- `run.sh` — unified launcher for inference and evaluation.
- `比赛说明.md` — competition task and submission requirements.

Large datasets, generated outputs, caches, and credentials are intentionally excluded from this public snapshot.

## Quick start

```bash
git clone https://github.com/lilxmx/AFAC-4-HiKEY-Hybrid-Light.git
cd AFAC-4-HiKEY-Hybrid-Light
python -m venv .venv
source .venv/bin/activate
pip install -r methods/m01_baseline_qwen/requirements.txt
cp .env.example .env
```

Fill in only the providers you plan to use. API keys must be supplied through environment variables; no credential is required to inspect the code or run non-networking utilities.

Example commands:

```bash
./run.sh m01 --inference-only
./run.sh eval --pred <prediction.csv> --ref <reference.json> --domain fc
```

See `COLLABORATION.md` for method registration, cache layout, output format, and evaluation details.

## Reproducibility and scope

This is a cleaned public release of a competition codebase. Exact leaderboard reproduction may require the original competition data, provider access, and locally generated indexes, which are not redistributed here. Please verify the competition rules and dataset licenses before using any included reference material.

## Security

Read [SECURITY.md](SECURITY.md) before configuring credentials. Never commit `.env` or paste an API key into source code, comments, notebooks, logs, or issue reports.

## License

No license has been declared yet. Until one is added, treat the code as “all rights reserved” and obtain permission before redistributing it.
