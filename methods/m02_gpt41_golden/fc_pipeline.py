"""
Financial Contracts Pipeline - Specialized pipeline for financial_contracts domain.
Implements the full Skill v6.0 workflow:
1. Query Planner: decompose question into atomic claims + field packs
2. FC Retriever: per-doc_id isolated retrieval with section priority
3. FC Answer Agent: structured field verification with three-value judgment
"""
import json
import time
import threading
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    QUESTION_FILES, OUTPUT_DIR, LOGS_DIR,
    BASELINE_INDICES_DIR, CONCURRENCY, VOTING_ROUNDS,
    BM25_TOP_K
)
from methods._shared.indexer import BM25Index, Chunk
from embedding_index import EmbeddingIndex
from query_planner import QueryPlanner
from fc_retriever import FCRetriever, format_evidence_by_doc
from fc_answer_agent import FCAnswerAgent

logger = logging.getLogger(__name__)


class FCPipeline:
    """
    Financial Contracts specialized pipeline.
    Implements the full Skill v6.0 workflow for maximum accuracy.
    """

    def __init__(self):
        self.bm25_index: Optional[BM25Index] = None
        self.embedding_index: Optional[EmbeddingIndex] = None
        self.retriever: Optional[FCRetriever] = None
        self.query_planner: Optional[QueryPlanner] = None
        self.answer_agent: Optional[FCAnswerAgent] = None
        self.results: Dict[str, Dict] = {}

    def setup(self, bm25_index: BM25Index, embedding_index: EmbeddingIndex):
        """
        Setup the FC pipeline with pre-loaded indices.

        Args:
            bm25_index: Pre-loaded BM25 index for financial_contracts
            embedding_index: Pre-loaded embedding index for financial_contracts
        """
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.retriever = FCRetriever(bm25_index, embedding_index)
        self.query_planner = QueryPlanner()
        self.answer_agent = FCAnswerAgent()
        logger.info("[FC Pipeline] Setup complete with specialized components")

    def process_question(self, q: Dict) -> Dict:
        """
        Process a single financial_contracts question through the full Skill v6.0 workflow.

        Args:
            q: Question dict with keys: qid, question, options, answer_format, doc_ids

        Returns:
            Result dict with answer, confidence, evidence, reasoning
        """
        qid = q['qid']
        question = q['question']
        options = q['options']
        answer_format = q['answer_format']
        doc_ids = q.get('doc_ids', [])

        logger.debug(f"[FC] Processing {qid}: {question[:50]}...")

        # ============================================================
        # Step 1: Query Planning
        # ============================================================
        try:
            plan = self.query_planner.plan(
                question=question,
                options=options,
                answer_format=answer_format,
                doc_ids=doc_ids,
                qid=qid,
            )
        except Exception as e:
            logger.warning(f"[FC] Query Planner failed for {qid}: {e}, using fallback")
            plan = self.query_planner._fallback_plan(
                question, options, answer_format, doc_ids, qid
            )

        # ============================================================
        # Step 2: Retrieval (per doc_id isolation)
        # ============================================================
        try:
            doc_evidence = self.retriever.retrieve_by_plan(plan)
        except Exception as e:
            logger.warning(f"[FC] Retrieval failed for {qid}: {e}, using backoff")
            doc_evidence = {}

        # Backoff: if any doc_id has no evidence, do fallback retrieval
        for doc_id in doc_ids:
            if doc_id not in doc_evidence or not doc_evidence[doc_id]:
                logger.debug(f"[FC] Backoff retrieval for {doc_id} in {qid}")
                try:
                    backoff_chunks = self.retriever.retrieve_with_backoff(
                        doc_id=doc_id,
                        question=question,
                        options=options,
                    )
                    doc_evidence[doc_id] = backoff_chunks
                except Exception as e:
                    logger.warning(f"[FC] Backoff retrieval failed for {doc_id}: {e}")
                    doc_evidence[doc_id] = []

        # ============================================================
        # Step 2.5: Evidence Sufficiency Check + Targeted Supplementary Retrieval
        # ============================================================
        doc_evidence = self._check_and_supplement_evidence(
            doc_evidence, question, options, doc_ids, plan
        )

        # ============================================================
        # Step 3: Format evidence
        # ============================================================
        # Calculate per-doc char limit based on number of docs
        n_docs = max(len(doc_ids), 1)
        max_chars_per_doc = 5000 // n_docs + 1500  # More generous allocation
        evidence_text = format_evidence_by_doc(doc_evidence, max_chars_per_doc=max_chars_per_doc)

        # ============================================================
        # Step 4: Answer with structured verification
        # ============================================================
        atomic_claims = plan.get('atomic_claims', [])

        try:
            result = self.answer_agent.answer(
                question=question,
                options=options,
                evidence_text=evidence_text,
                answer_format=answer_format,
                doc_ids=doc_ids,
                atomic_claims=atomic_claims,
                voting_rounds=VOTING_ROUNDS,
            )
        except Exception as e:
            logger.error(f"[FC] Answer Agent failed for {qid}: {e}")
            result = {
                'answer': 'A',
                'raw_answers': [],
                'all_votes': [],
                'confidence': 0,
                'reasoning': f'ERROR: {str(e)}',
            }

        # ============================================================
        # Step 5: Build output
        # ============================================================
        evidence_info = []
        for doc_id, chunks in doc_evidence.items():
            for chunk in chunks[:5]:  # Limit stored evidence
                evidence_info.append({
                    'doc_id': chunk.doc_id,
                    'page_num': chunk.page_num,
                    'section_title': chunk.section_title,
                    'text_snippet': chunk.text[:300],
                })

        output = {
            'qid': qid,
            'domain': 'financial_contracts',
            'answer_format': answer_format,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'all_votes': result.get('all_votes', []),
            'evidence': evidence_info,
            'reasoning': result.get('reasoning', '')[:2000],
            'plan': {
                'field_packs': plan.get('field_packs', []),
                'atomic_claims': plan.get('atomic_claims', []),
            },
            'pipeline': 'fc_skill_v6',  # Mark as processed by FC pipeline
        }

        return output

    def _check_and_supplement_evidence(
        self, doc_evidence: Dict[str, List[Chunk]],
        question: str, options: Dict[str, str],
        doc_ids: List[str], plan: Dict
    ) -> Dict[str, List[Chunk]]:
        """
        Check if retrieved evidence covers all options' key fields.
        If any option's keywords are completely missing from evidence,
        trigger targeted supplementary retrieval.

        This addresses the common failure mode where Query Planner groups
        fields into one pack but the retriever only returns chunks for
        some fields (e.g., returns issuance amount but misses net profit).
        """
        # Build a combined evidence text per doc for keyword checking
        doc_texts = {}
        for doc_id in doc_ids:
            chunks = doc_evidence.get(doc_id, [])
            doc_texts[doc_id] = " ".join(c.text for c in chunks)

        # Define option-level keyword groups for sufficiency checking
        # Each entry: (keywords_to_check, supplementary_query, target_sections)
        FIELD_KEYWORD_MAP = {
            "净利润": (
                ["净利润", "年均可分配利润", "归属于母公司所有者的净利润", "归母净利润"],
                "净利润 归属于母公司所有者的净利润 年均可分配利润 最近三年 最近三个会计年度 损益表",
                ["important_notice", "financial_info", "management_discussion"]
            ),
            "资产负债率": (
                ["资产负债率"],
                "资产负债率 合并口径 母公司口径 财务指标 主要财务数据",
                ["important_notice", "financial_info", "management_discussion"]
            ),
            "发行金额": (
                ["发行金额", "发行规模", "不超过"],
                "本期债券发行金额 发行规模 不超过 发行总额",
                ["cover", "issuance_overview", "important_notice"]
            ),
            "注册金额": (
                ["注册金额", "注册额度"],
                "注册金额 注册额度 不超过",
                ["cover", "issuance_overview", "important_notice"]
            ),
            "回售": (
                ["回售"],
                "回售 回售条款 回售选择权 有条件回售 附加回售 投资者回售",
                ["investor_protection", "issuance_overview"]
            ),
            "赎回": (
                ["赎回"],
                "赎回 赎回条款 有条件赎回 到期赎回",
                ["investor_protection", "issuance_overview"]
            ),
            "受托管理人": (
                ["受托管理人"],
                "受托管理人 债券受托管理人 发行有关机构",
                ["cover", "institutions", "definition"]
            ),
            "信息披露": (
                ["信息披露", "披露义务"],
                "信息披露 披露义务 及时 公平 发行人承诺",
                ["statement", "disclosure_arrangement"]
            ),
            "转股": (
                ["转股"],
                "转股价格 转股期限 转股起始日 初始转股价",
                ["issuance_overview"]
            ),
            "交易结构": (
                ["发行股份购买", "交易结构", "交易方案", "标的"],
                "发行股份购买资产 募集配套资金 交易结构 交易方案 标的公司 标的资产",
                ["transaction_overview", "target_info", "cover"]
            ),
            "违约": (
                ["违约", "罚息", "补偿"],
                "违约 违约事件 违约情形 违约利息 罚息 惩罚 赔偿 补偿 违约责任",
                ["issuance_overview", "investor_protection"]
            ),
            "控股股东": (
                ["控股股东", "实际控制人"],
                "控股股东 实际控制人 资产负债率 负债规模 股权结构",
                ["target_info", "important_notice", "issuer_info"]
            ),
        }

        # Check each option's text for field keywords
        all_option_text = question + " " + " ".join(options.values())
        supplementary_needed = []  # List of (doc_id, query, sections)

        for field_name, (check_keywords, supp_query, supp_sections) in FIELD_KEYWORD_MAP.items():
            # Is this field mentioned in the question/options?
            if not any(kw in all_option_text for kw in check_keywords):
                continue

            # For each doc_id, check if evidence already contains this field
            for doc_id in doc_ids:
                evidence_text = doc_texts.get(doc_id, "")
                # Check if ANY of the field keywords appear in the evidence
                field_found = any(kw in evidence_text for kw in check_keywords)

                if not field_found:
                    # This field is mentioned in options but NOT in evidence for this doc
                    supplementary_needed.append((doc_id, supp_query, supp_sections))

        # Execute supplementary retrieval
        if supplementary_needed:
            logger.debug(f"[FC] Evidence sufficiency check: {len(supplementary_needed)} supplementary retrievals needed")

        for doc_id, supp_query, supp_sections in supplementary_needed:
            try:
                supp_chunks = self.retriever._retrieve_for_doc(
                    doc_id=doc_id,
                    queries=[supp_query],
                    preferred_sections=supp_sections,
                    top_k_per_query=BM25_TOP_K,
                )
                if supp_chunks:
                    if doc_id not in doc_evidence:
                        doc_evidence[doc_id] = []
                    doc_evidence[doc_id].extend(supp_chunks)
                    # Re-deduplicate
                    doc_evidence[doc_id] = self.retriever._deduplicate_chunks(
                        doc_evidence[doc_id]
                    )
            except Exception as e:
                logger.warning(f"[FC] Supplementary retrieval failed for {doc_id}: {e}")

        return doc_evidence

    def run_batch(self, questions: List[Dict], resume_results: Dict = None) -> Dict[str, Dict]:
        """
        Run batch processing for financial_contracts questions.

        Args:
            questions: List of question dicts
            resume_results: Optional dict of already-processed results

        Returns:
            Dict mapping qid -> result
        """
        if resume_results:
            self.results = resume_results.copy()

        # Filter pending
        pending = [q for q in questions if q['qid'] not in self.results]
        logger.info(f"[FC Pipeline] Total: {len(questions)}, Pending: {len(pending)}, Done: {len(self.results)}")

        if not pending:
            return self.results

        lock = threading.Lock()
        completed = [0]

        def worker(q: Dict) -> tuple:
            qid = q['qid']
            try:
                result = self.process_question(q)
                return qid, result
            except Exception as e:
                logger.error(f"[FC] Worker failed for {qid}: {e}")
                return qid, {
                    'qid': qid,
                    'domain': 'financial_contracts',
                    'answer_format': q['answer_format'],
                    'answer': 'A',
                    'confidence': 0,
                    'all_votes': [],
                    'evidence': [],
                    'reasoning': f'ERROR: {str(e)}',
                    'pipeline': 'fc_skill_v6',
                }

        logger.info(f"[FC Pipeline] Starting with {CONCURRENCY} workers...")

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(worker, q): q for q in pending}

            pbar = tqdm(total=len(pending), desc="FC Pipeline (Skill v6.0)")
            for future in as_completed(futures):
                qid, result = future.result()

                with lock:
                    self.results[qid] = result
                    completed[0] += 1

                    if completed[0] % 3 == 0:
                        avg_conf = sum(
                            r.get('confidence', 0) for r in self.results.values()
                        ) / len(self.results)
                        logger.info(
                            f"  [FC] Progress: {completed[0]}/{len(pending)} "
                            f"(avg confidence: {avg_conf:.2f})"
                        )

                pbar.update(1)
            pbar.close()

        logger.info(f"[FC Pipeline] Complete! {completed[0]} questions processed.")
        return self.results
