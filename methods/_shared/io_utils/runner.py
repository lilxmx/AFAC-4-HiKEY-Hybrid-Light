"""
Unified Run Context Manager.

Generates unique run_id, creates output directories, writes manifest.json,
and redirects stdout/stderr to log files.

Usage:
    from methods._shared.io_utils import create_run

    ctx = create_run(
        coder="lgr",
        method_id="m01_baseline_qwen",
        desc="fc-full-v3",
        model_llm="qwen-plus",
        model_embedding="text-embedding-v3",
        notes="first run with new retriever"
    )

    # ctx.logs_dir, ctx.output_dir, ctx.intermediate_dir are ready
    # ctx.manifest_path points to manifest.json
    # Call ctx.finish(metrics={...}) when done
"""
import json
import os
import sys
import subprocess
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# Project root: AFAC-4/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"

CST = timezone(timedelta(hours=8))


@dataclass
class RunContext:
    """Holds all paths and metadata for a single run."""
    run_id: str
    coder: str
    method_id: str
    run_dir: Path
    logs_dir: Path
    output_dir: Path
    intermediate_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any] = field(default_factory=dict)

    def finish(self, metrics: Optional[Dict] = None, status: str = "success", notes: str = ""):
        """Finalize the run: update manifest with end time and metrics."""
        self.manifest["finished_at"] = datetime.now(CST).isoformat()
        self.manifest["status"] = status
        if metrics:
            self.manifest["metrics"] = metrics
        if notes:
            self.manifest["notes"] = notes
        self._write_manifest()

    def fail(self, error_msg: str = ""):
        """Mark run as failed."""
        self.manifest["finished_at"] = datetime.now(CST).isoformat()
        self.manifest["status"] = "failed"
        self.manifest["error"] = error_msg
        self._write_manifest()

    def _write_manifest(self):
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)


def _get_git_commit() -> str:
    """Try to get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_config_hash(config_dict: Optional[Dict]) -> str:
    """Compute a short hash of the config for reproducibility tracking."""
    if not config_dict:
        return "none"
    raw = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def create_run(
    coder: str,
    method_id: str,
    desc: str = "",
    model_llm: str = "",
    model_embedding: str = "",
    config_snapshot: Optional[Dict] = None,
    notes: str = "",
    data_info: Optional[Dict] = None,
) -> RunContext:
    """
    Create a new run directory with all subdirectories and manifest.

    Args:
        coder: Short code for the person running (e.g. "lgr")
        method_id: Method directory name (e.g. "m01_baseline_qwen")
        desc: Short description for the run_id suffix
        model_llm: LLM model name
        model_embedding: Embedding model name
        config_snapshot: Dict of config to snapshot
        notes: Free-text notes
        data_info: Dict describing data sources

    Returns:
        RunContext with all paths ready to use.
    """
    now = datetime.now(CST)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{coder}_{desc}" if desc else f"{timestamp}_{coder}"

    # Create directory structure
    run_dir = RUNS_ROOT / coder / method_id / run_id
    logs_dir = run_dir / "logs"
    output_dir = run_dir / "output"
    intermediate_dir = run_dir / "intermediate"

    for d in [logs_dir, output_dir, intermediate_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = {
        "run_id": run_id,
        "coder": coder,
        "method_id": method_id,
        "git_commit": _get_git_commit(),
        "started_at": now.isoformat(),
        "finished_at": None,
        "status": "running",
        "model": {
            "llm": model_llm,
            "embedding": model_embedding,
        },
        "data": data_info or {},
        "config_hash": _compute_config_hash(config_snapshot),
        "metrics": {},
        "notes": notes,
    }

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Save config snapshot if provided
    if config_snapshot:
        config_path = run_dir / "config.snapshot.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    ctx = RunContext(
        run_id=run_id,
        coder=coder,
        method_id=method_id,
        run_dir=run_dir,
        logs_dir=logs_dir,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    return ctx


def setup_logging(ctx: RunContext):
    """Redirect stdout/stderr to log files (for background runs)."""
    stdout_log = open(ctx.logs_dir / "stdout.log", 'a', encoding='utf-8')
    stderr_log = open(ctx.logs_dir / "stderr.log", 'a', encoding='utf-8')
    sys.stdout = stdout_log
    sys.stderr = stderr_log
    return stdout_log, stderr_log
