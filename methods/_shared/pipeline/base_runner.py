"""
BaseRunner: shared concurrent inference loop with resume support.

Subclass usage:

    class MyRunner(BaseRunner):
        method_name = "m99_xxx"
        model_name  = "qwen-plus"

        def process_question(self, q: dict) -> dict:
            ...
            return {
                "qid": q["qid"],
                "answer": "A",
                "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...,
                # optional: "evidence", "raw_answer", "reasoning", "domain"
            }

    runner = MyRunner(output_dir=..., logs_dir=..., concurrency=5)
    runner.run(domains=["insurance"])
    runner.save_results()

The base class handles:
- loading questions from QUESTION_FILES,
- writing partial_results.json for crash recovery,
- ThreadPoolExecutor concurrency with tqdm progress,
- writing answer.csv / evidence.json / run_summary.json on save_results().
- unique run_id generation and output isolation via RunContext.
"""
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from ..config_base import DOMAINS, QUESTION_FILES, infer_domain_from_qid
from ..io_utils.runner import create_run, RunContext
from .output import save_answer_csv, save_evidence_json, save_run_summary

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Optional helper dataclass; methods may also return plain dicts."""
    qid: str
    answer: str = "A"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_answer: str = ""
    evidence: list = field(default_factory=list)
    domain: str = ""

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "answer": self.answer,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "raw_answer": self.raw_answer,
            "evidence": self.evidence,
            "domain": self.domain,
        }


class BaseRunner:
    """Skeleton for a Q&A inference pipeline. Subclasses override process_question."""

    method_name: str = "base"
    model_name: str = "unknown"

    # Submission CSV delimiter. Official format uses comma; m03 used tab — keep comma default.
    answer_csv_delimiter: str = ","

    def __init__(
        self,
        output_dir: Path,
        logs_dir: Path,
        concurrency: int = 10,
        save_partial_every: int = 10,
        save_evidence: bool = True,
        coder: str = None,
        run_desc: str = "",
    ):
        # These are only used as fallback defaults before run() creates the RunContext
        self._method_output_dir = Path(output_dir)
        self._method_logs_dir = Path(logs_dir)

        self.concurrency = concurrency
        self.save_partial_every = save_partial_every
        self.save_evidence = save_evidence

        # RunContext will be created in run() to generate unique run_id
        self._coder = coder or os.getenv("CODER", "lgr")
        self._run_desc = run_desc
        self.run_ctx: Optional[RunContext] = None

        # These will be set to the run-specific dirs once run() is called
        self.output_dir = self._method_output_dir
        self.logs_dir = self._method_logs_dir

        self.results: Dict[str, Dict] = {}
        self.token_stats = {"total_prompt": 0, "total_completion": 0, "total_tokens": 0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------
    def process_question(self, q: Dict) -> Dict:
        """Override in subclass. Must return a dict containing at least
        keys: qid, answer, prompt_tokens, completion_tokens, total_tokens."""
        raise NotImplementedError

    def setup(self, domains: List[str]) -> None:
        """Optional: build indices, load embeddings, etc. Override as needed."""
        pass

    # ------------------------------------------------------------
    # Question loading
    # ------------------------------------------------------------
    def load_questions(self, domains: Optional[List[str]] = None) -> List[Dict]:
        domains = domains or DOMAINS
        all_q: List[Dict] = []
        for domain in domains:
            qfile = QUESTION_FILES.get(domain)
            if qfile is None or not qfile.exists():
                logger.warning(f"Question file not found for {domain}")
                continue
            with open(qfile, "r", encoding="utf-8") as f:
                qs = json.load(f)
            for q in qs:
                q.setdefault("domain", domain)
            all_q.extend(qs)
            logger.info(f"Loaded {len(qs)} questions for {domain}")
        return all_q

    # ------------------------------------------------------------
    # Resume / partial-save bookkeeping
    # ------------------------------------------------------------
    @property
    def partial_path(self) -> Path:
        return self.output_dir / "partial_results.json"



    def _load_partial(self) -> None:
        """Load partial results from run-specific dir."""
        if self.partial_path.exists():
            target = self.partial_path
        else:
            return

        try:
            with open(target, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.results = saved.get("results", {})
            self.token_stats = saved.get("token_stats", self.token_stats)
            logger.info(f"Resumed: {len(self.results)} questions already done (from {target})")
        except Exception as e:
            logger.warning(f"Failed to load partial results: {e} (starting fresh)")

    def _save_partial(self) -> None:
        """Save partial results to run-specific dir only."""
        payload = {"results": self.results, "token_stats": self.token_stats}
        try:
            with open(self.partial_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save partial results: {e}")

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def run(self, domains: Optional[List[str]] = None, resume: bool = True, qid_filter: Optional[List[str]] = None) -> None:
        domains = domains or DOMAINS

        # Create a unique RunContext for this run
        self.run_ctx = create_run(
            coder=self._coder,
            method_id=self.method_name,
            desc=self._run_desc,
            model_llm=self.model_name,
        )
        # Use run-specific output/logs directories
        self.output_dir = self.run_ctx.output_dir
        self.logs_dir = self.run_ctx.logs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Run ID: {self.run_ctx.run_id}")
        logger.info(f"Run output dir: {self.output_dir}")

        # Allow subclasses to build/load indices etc.
        self.setup(domains)

        if resume:
            self._load_partial()

        all_questions = self.load_questions(domains)

        # Apply qid filter if specified
        if qid_filter:
            qid_set = set(qid_filter)
            all_questions = [q for q in all_questions if q["qid"] in qid_set]
            logger.info(f"QID filter applied: {len(all_questions)} questions selected from {len(qid_filter)} requested IDs")

        pending = [q for q in all_questions if q["qid"] not in self.results]
        skipped = len(all_questions) - len(pending)
        logger.info(f"Total: {len(all_questions)}, Pending: {len(pending)}, Skipped: {skipped}")

        if not pending:
            logger.info("All questions already answered.")
            return

        detail_log_path = self.logs_dir / "question_details.jsonl"
        completed = [0]

        def worker(q: Dict):
            try:
                return q["qid"], q, self.process_question(q)
            except Exception as e:
                logger.exception(f"[{q.get('qid')}] process_question failed: {e}")
                return q["qid"], q, {
                    "qid": q["qid"],
                    "answer": "A",
                    "raw_answer": f"ERROR: {e}",
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "domain": q.get("domain") or infer_domain_from_qid(q["qid"]),
                }

        logger.info(f"Starting concurrent inference with {self.concurrency} workers...")
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(worker, q): q for q in pending}
            pbar = tqdm(total=len(pending), desc=f"{self.method_name}")
            for fut in as_completed(futures):
                qid, q, result = fut.result()
                # Ensure required fields
                result.setdefault("qid", qid)
                result.setdefault("domain", q.get("domain") or infer_domain_from_qid(qid))

                with self._lock:
                    self.results[qid] = result
                    self.token_stats["total_prompt"] += int(result.get("prompt_tokens", 0) or 0)
                    self.token_stats["total_completion"] += int(result.get("completion_tokens", 0) or 0)
                    self.token_stats["total_tokens"] += int(result.get("total_tokens", 0) or 0)
                    completed[0] += 1

                    # per-question detail log
                    try:
                        detail_entry = {
                            "qid": qid,
                            "domain": result.get("domain"),
                            "answer_format": q.get("answer_format", "mcq"),
                            "answer": result.get("answer"),
                            "prompt_tokens": result.get("prompt_tokens", 0),
                            "completion_tokens": result.get("completion_tokens", 0),
                            "total_tokens": result.get("total_tokens", 0),
                            "evidence_count": len(result.get("evidence", []) or []),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(detail_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(detail_entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                    if self.save_partial_every and completed[0] % self.save_partial_every == 0:
                        self._save_partial()
                        logger.info(
                            f"Progress: {completed[0]+skipped}/{len(all_questions)} "
                            f"tokens={self.token_stats['total_tokens']:,}"
                        )
                pbar.update(1)
            pbar.close()

        self._save_partial()
        logger.info(
            f"Inference complete. Done={completed[0]}, Skipped={skipped}, "
            f"Total tokens={self.token_stats['total_tokens']:,}"
        )

    # ------------------------------------------------------------
    # Final result writer
    # ------------------------------------------------------------
    def save_results(self, extra_summary: Optional[Dict] = None) -> Dict:
        # Save to run-specific output directory
        answer_csv_path = self.output_dir / "answer.csv"
        save_answer_csv(
            answer_csv_path,
            self.results,
            self.token_stats,
            delimiter=self.answer_csv_delimiter,
        )
        logger.info(f"Saved answer.csv: {answer_csv_path}")

        evidence_json_path = self.output_dir / "evidence.json"
        if self.save_evidence:
            save_evidence_json(evidence_json_path, self.results)
            logger.info(f"Saved evidence.json: {evidence_json_path}")

        summary_path = self.logs_dir / "run_summary.json"
        summary = save_run_summary(
            summary_path,
            method_name=self.method_name,
            model_name=self.model_name,
            results=self.results,
            token_stats=self.token_stats,
            extra=extra_summary,
        )
        logger.info(f"Saved run_summary.json: {summary_path}")



        # Finalize RunContext
        if self.run_ctx:
            self.run_ctx.finish(metrics=summary.get("domain_stats", {}))
            logger.info(f"Run finalized: {self.run_ctx.run_id}")

        return summary
