"""
m04_HiKEY pipeline - HiKEY-style hierarchical retrieval + evidence packing for financial QA.

Core idea from HiKET paper:
    PDF → Hierarchy Tree → DocCard / SectionCard / EvidenceUnit / FieldCard
    → Query Planning → Coarse-to-Fine Retrieval → Evidence Pack Assembly → LLM Reader

This pipeline integrates the standalone hikey_financial_parser.py into the shared BaseRunner
framework, enabling concurrent inference with resume support and standardized output.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods._shared.llm import make_dashscope_client, normalize_answer
from methods._shared.parsers import resolve_doc_path, parse_document_blocks
from methods._shared.pipeline import BaseRunner
from methods._shared.config_base import infer_domain_from_qid

from config import (
    CONCURRENCY,
    DASHSCOPE_API_KEY,
    DOMAIN_HINTS,
    ENABLE_THINKING,
    EVIDENCE_TOKEN_BUDGET,
    FALLBACK_DOMAINS,
    FIELDCARD_FIRST,
    FORMAT_HINTS,
    HIKEY_DOMAINS,
    HIKEY_INDEX_DIR,
    MAX_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS_REFLECTION,
    MAX_SIBLINGS,
    METHOD_NAME,
    METHOD_ROOT,
    MODEL_NAME,
    MULTI_QA_PROMPT_TEMPLATE,
    MULTI_REFLECTION_PROMPT,
    PROJECT_BM25_INDEX_DIR,
    QA_PROMPT_TEMPLATE,
    RETRIEVAL_TOPK,
    SKIP_DOC_ROUTING,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_REFLECTION,
    TEMPERATURE,
    TF_PROMPT_TEMPLATE,
    THINKING_BUDGET,
)

# Import HiKEY components from the parser module
from hikey_financial_parser import (
    METRIC_ALIASES,
    COMPANY_ALIASES,
    QueryPlanner,
    SimpleRetriever,
    HiKEYFinanceParser,
    normalize_text,
    compact,
    metric_aliases_for_query,
)

logger = logging.getLogger(__name__)


class HiKEYIndexManager:
    """Manages pre-built HiKEY indices for all documents in a domain.

    For financial_reports domain, uses the full HiKEY hierarchical parsing.
    For other domains, falls back to standard block-based retrieval.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.retrievers: Dict[str, SimpleRetriever] = {}
        self.doc_cards: Dict[str, Dict[str, Any]] = {}
        self._loaded_domains: set = set()

    def ensure_domain_loaded(self, domain: str) -> None:
        """Load or build HiKEY indices for all documents in a domain."""
        if domain in self._loaded_domains:
            return

        domain_index_dir = self.index_dir / domain
        if not domain_index_dir.exists():
            domain_index_dir.mkdir(parents=True, exist_ok=True)

        # Check if pre-built indices exist
        manifest_path = domain_index_dir / "manifest.json"
        if manifest_path.exists():
            self._load_domain_from_manifest(domain, manifest_path)
        else:
            logger.info(f"No pre-built HiKEY index for domain={domain}. "
                        f"Run `python pipeline.py --build-index --domains {domain}` first.")

        self._loaded_domains.add(domain)

    def _load_domain_from_manifest(self, domain: str, manifest_path: Path) -> None:
        """Load pre-built indices from manifest."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for doc_entry in manifest.get("documents", []):
            doc_id = doc_entry["doc_id"]
            doc_dir = self.index_dir / domain / doc_id

            sections_path = doc_dir / "sections.jsonl"
            units_path = doc_dir / "units.jsonl"
            field_cards_path = doc_dir / "field_cards.jsonl"

            if not sections_path.exists() or not units_path.exists():
                logger.warning(f"Missing index files for {doc_id}, skipping")
                continue

            try:
                retriever = SimpleRetriever(
                    str(sections_path),
                    str(units_path),
                    str(field_cards_path) if field_cards_path.exists() else None,
                )
                self.retrievers[doc_id] = retriever

                doc_card_path = doc_dir / "doc_card.json"
                if doc_card_path.exists():
                    with open(doc_card_path, "r", encoding="utf-8") as f:
                        self.doc_cards[doc_id] = json.load(f)

                logger.info(f"Loaded HiKEY index: {doc_id} "
                            f"(sections={len(retriever.sections)}, "
                            f"units={len(retriever.units)}, "
                            f"field_cards={len(retriever.field_cards)})")
            except Exception as e:
                logger.error(f"Failed to load index for {doc_id}: {e}")

    def get_retriever(self, doc_id: str) -> Optional[SimpleRetriever]:
        """Get the retriever for a specific document."""
        return self.retrievers.get(doc_id)

    def route_document(self, query: str, domain: str) -> List[str]:
        """Stage-1 Document Routing: find the most relevant documents for a query.

        Uses DocCard matching (company + year + section hierarchy).
        """
        plan = QueryPlanner.plan(query)
        candidates = []

        for doc_id, doc_card in self.doc_cards.items():
            score = 0.0

            # Company match (strongest signal)
            doc_company = doc_card.get("company", "")
            for company in plan.companies:
                aliases = COMPANY_ALIASES.get(company, [company])
                if doc_company == company or any(a in doc_company for a in aliases):
                    score += 10.0
                    break

            # Year match
            doc_year = doc_card.get("year")
            if doc_year and doc_year in plan.years:
                score += 5.0

            # Section path match (weaker signal for topic routing)
            top_sections = doc_card.get("top_sections", [])
            section_text = " ".join(top_sections)
            for metric in plan.metrics:
                aliases = METRIC_ALIASES.get(metric, [metric])
                if any(a in section_text for a in aliases):
                    score += 1.0

            if score > 0:
                candidates.append((doc_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in candidates[:3]]

    def search_with_routing(self, query: str, domain: str, topk: int = 8,
                            target_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Full HiKEY retrieval: Doc Routing → Section/Unit Retrieval → Evidence Pack.

        Args:
            target_doc_ids: If provided (Phase A), skip Stage-1 DocCard routing and
                            search directly within these documents.
        """
        if target_doc_ids and SKIP_DOC_ROUTING:
            # Phase A: question already specifies target documents, skip DocCard routing
            candidate_docs = target_doc_ids
        else:
            # Phase B / open-domain: use Stage-1 Document routing
            candidate_docs = self.route_document(query, domain)

        if not candidate_docs:
            # Fallback: search all loaded retrievers
            candidate_docs = list(self.retrievers.keys())

        # Stage-2: Search within candidate documents
        all_results = []
        for doc_id in candidate_docs:
            retriever = self.retrievers.get(doc_id)
            if not retriever:
                continue
            results = retriever.search(query, topk=topk)
            for r in results:
                r["doc_id"] = doc_id
            all_results.extend(results)

        # Re-rank by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:topk]

    def pack_evidence(self, query: str, domain: str, topk: int = 5,
                      target_doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Full HiKEY evidence packing with ancestry + siblings.

        Args:
            target_doc_ids: If provided (Phase A), skip Stage-1 DocCard routing and
                            pack evidence directly from these documents.

        When multiple target documents are specified, evidence is gathered from ALL
        documents (distributing topk evenly), then merged and re-ranked by score.
        This is critical for multi-select questions that reference multiple docs.
        """
        if target_doc_ids and SKIP_DOC_ROUTING:
            # Phase A: use specified documents directly
            candidate_docs = target_doc_ids
        else:
            # Phase B: use DocCard routing
            candidate_docs = self.route_document(query, domain)

        if not candidate_docs:
            candidate_docs = list(self.retrievers.keys())

        # Gather evidence from ALL candidate documents (not just the first one)
        all_evidence_items = []
        routed_docs = []
        # Distribute topk across documents, ensuring at least 3 per doc
        per_doc_topk = max(3, topk // max(len(candidate_docs), 1))

        for doc_id in candidate_docs:
            retriever = self.retrievers.get(doc_id)
            if not retriever:
                continue
            pack = retriever.pack_evidence(query, topk=per_doc_topk, max_siblings=MAX_SIBLINGS)
            items = pack.get("evidence_pack", [])
            # Tag each item with its source doc_id
            for item in items:
                item["source_doc_id"] = doc_id
            all_evidence_items.extend(items)
            routed_docs.append(doc_id)

        if not all_evidence_items:
            return {"query": query, "evidence_pack": [], "routed_doc": None}

        # Re-rank all evidence items by score and take top-k
        all_evidence_items.sort(
            key=lambda x: x.get("anchor", {}).get("score", 0), reverse=True
        )
        final_items = all_evidence_items[:topk]

        return {
            "query": query,
            "evidence_pack": final_items,
            "routed_doc": routed_docs[0] if routed_docs else None,
            "routed_docs": routed_docs,
        }


def format_evidence_for_prompt(evidence_pack: Dict[str, Any], token_budget: int = 6000) -> str:
    """Serialize HiKEY evidence pack into a readable prompt string.

    Follows the HiKEY principle: anchor + ancestry + siblings, within token budget.
    """
    items = evidence_pack.get("evidence_pack", [])
    if not items:
        return "(无相关证据)"

    lines = []
    char_budget = token_budget * 2  # rough CJK char-to-token ratio
    used = 0

    for i, item in enumerate(items, 1):
        anchor = item.get("anchor", {})
        ancestry = item.get("ancestry", {})
        siblings = item.get("siblings", [])

        block = []
        block.append(f"[证据 {i}]")

        # Ancestry context (section path + page)
        section_path = ancestry.get("section_path") or anchor.get("section_path", "")
        if section_path:
            block.append(f"章节路径: {section_path}")

        page = anchor.get("source_page") or anchor.get("page") or ancestry.get("start_page")
        if page:
            block.append(f"页码: {page}")

        # Anchor content
        obj_type = anchor.get("object_type", "")
        if obj_type == "field_card":
            metric = anchor.get("metric", "")
            value_map = anchor.get("value_map", {})
            unit = anchor.get("unit", "")
            block.append(f"指标: {metric}")
            if unit:
                block.append(f"单位: {unit}")
            block.append(f"数值: {json.dumps(value_map, ensure_ascii=False)}")
            table_name = anchor.get("table_name", "")
            if table_name:
                block.append(f"表格: {table_name}")
        else:
            text = anchor.get("text", "")
            if text:
                block.append(f"内容: {text[:800]}")
            row_header = anchor.get("row_header", "")
            if row_header:
                block.append(f"行标题: {row_header}")
            unit = anchor.get("unit", "")
            if unit:
                block.append(f"单位: {unit}")
            table_name = anchor.get("table_name", "")
            if table_name:
                block.append(f"表格: {table_name}")

        # Sibling context (abbreviated)
        if siblings:
            sibling_texts = []
            for s in siblings[:3]:
                s_text = s.get("text", "")[:200]
                if s_text:
                    sibling_texts.append(s_text)
            if sibling_texts:
                block.append(f"相关上下文: {' | '.join(sibling_texts)}")

        block_text = "\n".join(block)
        if used + len(block_text) > char_budget:
            break
        lines.append(block_text)
        used += len(block_text)

    return "\n\n".join(lines)


def format_options(question_data: Dict[str, Any]) -> str:
    """Format question options for the prompt."""
    options = question_data.get("options", {})
    if not options:
        return ""
    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)


class HiKEYPipeline(BaseRunner):
    """HiKEY-style hierarchical retrieval pipeline for financial QA.

    Implements the full HiKEY flow:
    1. Query Planning: extract company/year/metric/operation from question
    2. Document Routing: find relevant documents via DocCard matching
    3. Section/Unit Retrieval: BM25 + domain boost within candidate docs
    4. Evidence Packing: anchor + ancestry + siblings assembly
    5. LLM Reader: generate answer with structured evidence context
    """

    method_name = METHOD_NAME
    model_name = MODEL_NAME

    def __init__(self):
        super().__init__(
            output_dir=METHOD_ROOT / "output",
            logs_dir=METHOD_ROOT / "logs",
            concurrency=CONCURRENCY,
        )
        self.client = make_dashscope_client(api_key=DASHSCOPE_API_KEY)
        self.index_manager = HiKEYIndexManager(HIKEY_INDEX_DIR)

        # Fallback: standard block-based retrieval for non-HiKEY domains
        self._fallback_indices: Dict[str, Any] = {}

    def setup(self, domains: List[str]) -> None:
        """Load HiKEY indices for requested domains."""
        for domain in domains:
            # Try to load HiKEY index for all domains
            self.index_manager.ensure_domain_loaded(domain)
            # Also setup fallback for domains that might not have full HiKEY coverage
            if domain in FALLBACK_DOMAINS:
                self._setup_fallback_domain(domain)

    def _setup_fallback_domain(self, domain: str) -> None:
        """Setup standard BM25 retrieval for non-HiKEY domains."""
        # Use the shared BM25 index if available
        from methods._shared.indexer import BM25Index
        index_path = PROJECT_BM25_INDEX_DIR / f"{domain}.pkl"
        if index_path.exists():
            try:
                self._fallback_indices[domain] = BM25Index.load(str(index_path))
                logger.info(f"Loaded fallback BM25 index for {domain}")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index for {domain}: {e}")

    def process_question(self, q: Dict) -> Dict:
        """Core logic: receive a question, return answer + token stats.

        For financial_reports domain: full HiKEY pipeline
        For other domains: fallback to standard retrieval + LLM
        """
        qid = q["qid"]
        domain = q.get("domain") or infer_domain_from_qid(qid)
        question_text = q.get("question", "")
        answer_format = q.get("answer_format", "mcq")

        try:
            # Use HiKEY retrieval for all domains (indices pre-built)
            evidence_text = self._retrieve_hikey(question_text, domain, q)

            # Build prompt with format hint and domain hint
            options_text = format_options(q)
            format_hint = FORMAT_HINTS.get(answer_format, FORMAT_HINTS.get('mcq', ''))
            domain_hint = DOMAIN_HINTS.get(domain, '')

            # Select prompt template based on answer format
            if answer_format == "multi":
                prompt = MULTI_QA_PROMPT_TEMPLATE.format(
                    question=question_text,
                    options=options_text,
                    evidence=evidence_text,
                    format_hint=format_hint,
                    domain_hint=domain_hint,
                )
            elif answer_format == "tf":
                prompt = TF_PROMPT_TEMPLATE.format(
                    question=question_text,
                    options=options_text,
                    evidence=evidence_text,
                    format_hint=format_hint,
                    domain_hint=domain_hint,
                )
            else:
                prompt = QA_PROMPT_TEMPLATE.format(
                    question=question_text,
                    options=options_text,
                    evidence=evidence_text,
                    format_hint=format_hint,
                    domain_hint=domain_hint,
                )

            # Call LLM (with thinking mode: model reasons internally, outputs concise answer)
            extra_body = {}
            if ENABLE_THINKING:
                extra_body['enable_thinking'] = True
                if THINKING_BUDGET:
                    extra_body['thinking_budget'] = THINKING_BUDGET

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                **({"extra_body": extra_body} if extra_body else {}),
            )

            msg = response.choices[0].message
            raw_answer = msg.content or ""
            # Capture reasoning_content if available (thinking mode)
            reasoning_text = getattr(msg, 'reasoning_content', None) or ""
            answer = normalize_answer(raw_answer, answer_format=answer_format)

            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0

            # Multi-select reflection: if only 1 option selected, do a second pass
            if answer_format == "multi" and len(answer) <= 1:
                reflection_prompt = MULTI_REFLECTION_PROMPT.format(
                    previous_answer=answer,
                    evidence=evidence_text,
                    question=question_text,
                    options=options_text,
                    domain_hint=domain_hint,
                )
                try:
                    reflection_extra_body = {}
                    if ENABLE_THINKING:
                        reflection_extra_body['enable_thinking'] = True
                        if THINKING_BUDGET:
                            reflection_extra_body['thinking_budget'] = THINKING_BUDGET

                    reflection_response = self.client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_REFLECTION},
                            {"role": "user", "content": reflection_prompt},
                        ],
                        max_tokens=MAX_OUTPUT_TOKENS_REFLECTION,
                        temperature=TEMPERATURE,
                        **({"extra_body": reflection_extra_body} if reflection_extra_body else {}),
                    )
                    reflection_msg = reflection_response.choices[0].message
                    reflection_raw = reflection_msg.content or ""
                    reflection_reasoning = getattr(reflection_msg, 'reasoning_content', None) or ""
                    reflection_answer = normalize_answer(reflection_raw, answer_format="multi")
                    # Only accept reflection if it found more options
                    if len(reflection_answer) > len(answer):
                        answer = reflection_answer
                        raw_answer = raw_answer + "\n[REFLECTION]\n" + reflection_raw
                        reasoning_text = reasoning_text + "\n[REFLECTION_THINKING]\n" + reflection_reasoning
                    # Accumulate tokens
                    prompt_tokens += reflection_response.usage.prompt_tokens if reflection_response.usage else 0
                    completion_tokens += reflection_response.usage.completion_tokens if reflection_response.usage else 0
                    total_tokens += (reflection_response.usage.total_tokens if reflection_response.usage else 0)
                except Exception as e:
                    logger.warning(f"[{qid}] Reflection failed: {e}")

            return {
                "qid": qid,
                "answer": answer,
                "raw_answer": raw_answer,
                "reasoning": reasoning_text[:3000] if reasoning_text else "",
                "domain": domain,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "evidence": [evidence_text[:2000]],
            }

        except Exception as e:
            logger.error(f"[{qid}] Error: {e}")
            return {
                "qid": qid,
                "answer": "A",
                "raw_answer": f"ERROR: {e}",
                "domain": domain,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def _get_target_doc_ids(self, q: Dict) -> List[str]:
        """Extract target document IDs from question metadata.

        In Phase A, each question specifies which document(s) to search.
        Returns a list of doc_ids, or empty list if not specified.
        """
        doc_ids = q.get("doc_ids", [])
        if not doc_ids:
            doc_id = q.get("doc_id", "")
            if doc_id:
                doc_ids = [doc_id]
        return doc_ids

    def _retrieve_hikey(self, query: str, domain: str, q: Dict) -> str:
        """HiKEY-style retrieval: DocRouting → Section/Unit search → Evidence Pack.

        When SKIP_DOC_ROUTING=True (Phase A), skips Stage-1 DocCard routing and
        directly searches within the document(s) specified by the question.
        """
        target_doc_ids = self._get_target_doc_ids(q) if SKIP_DOC_ROUTING else None

        # Try FieldCard-first strategy
        if FIELDCARD_FIRST:
            plan = QueryPlanner.plan(query)
            if plan.metrics:
                # Direct FieldCard lookup
                field_results = self.index_manager.search_with_routing(
                    query, domain, topk=RETRIEVAL_TOPK,
                    target_doc_ids=target_doc_ids,
                )
                field_cards = [r for r in field_results if r.get("object_type") == "field_card"]
                if field_cards:
                    # Pack evidence around field cards
                    pack = self.index_manager.pack_evidence(
                        query, domain, topk=RETRIEVAL_TOPK,
                        target_doc_ids=target_doc_ids,
                    )
                    return format_evidence_for_prompt(pack, EVIDENCE_TOKEN_BUDGET)

        # Full evidence pack
        pack = self.index_manager.pack_evidence(
            query, domain, topk=RETRIEVAL_TOPK,
            target_doc_ids=target_doc_ids,
        )
        evidence_text = format_evidence_for_prompt(pack, EVIDENCE_TOKEN_BUDGET)

        if not evidence_text or evidence_text == "(无相关证据)":
            # Fallback to raw document blocks
            evidence_text = self._retrieve_fallback(query, domain, q)

        return evidence_text

    def _retrieve_fallback(self, query: str, domain: str, q: Dict) -> str:
        """Standard BM25 retrieval for non-HiKEY domains."""
        doc_ids = q.get("doc_ids", [])
        if not doc_ids:
            # Try to get doc_id from question metadata
            doc_id = q.get("doc_id", "")
            if doc_id:
                doc_ids = [doc_id]

        # Use shared parser to get document blocks
        all_blocks = []
        for doc_id in doc_ids[:3]:  # Limit to 3 documents
            try:
                blocks = parse_document_blocks(doc_id, domain)
                all_blocks.extend(blocks)
            except Exception as e:
                logger.warning(f"Failed to parse {doc_id}: {e}")

        if not all_blocks:
            return "(无法获取文档内容)"

        # Simple keyword matching as fallback
        from hikey_financial_parser import simple_tokens
        q_tokens = set(simple_tokens(query))
        scored_blocks = []
        for block in all_blocks:
            text = block.get("text", "")
            b_tokens = set(simple_tokens(text))
            overlap = len(q_tokens & b_tokens)
            if overlap > 0:
                scored_blocks.append((overlap, block))

        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        top_blocks = scored_blocks[:5]

        if not top_blocks:
            # Just use first few blocks
            top_blocks = [(0, b) for b in all_blocks[:5]]

        lines = []
        char_budget = EVIDENCE_TOKEN_BUDGET * 2
        used = 0
        for i, (score, block) in enumerate(top_blocks, 1):
            text = block.get("text", "")[:1000]
            page = block.get("page_num", "?")
            section = block.get("section_title", "")
            entry = f"[证据 {i}] (p.{page}) {section}\n{text}"
            if used + len(entry) > char_budget:
                break
            lines.append(entry)
            used += len(entry)

        return "\n\n".join(lines) if lines else "(无相关证据)"


def build_hikey_index(domains: List[str]) -> None:
    """Pre-build HiKEY indices for all documents in specified domains.

    This should be run once before inference to create the hierarchical index.
    """
    import shutil
    from methods._shared.parsers import get_all_doc_paths

    for domain in domains:
        if domain not in HIKEY_DOMAINS:
            logger.info(f"Skipping non-HiKEY domain: {domain}")
            continue

        logger.info(f"Building HiKEY index for domain: {domain}")
        domain_index_dir = HIKEY_INDEX_DIR / domain
        domain_index_dir.mkdir(parents=True, exist_ok=True)

        # get_all_doc_paths returns List[Tuple[str, Path, str]] = [(doc_id, path, file_type), ...]
        try:
            doc_entries = get_all_doc_paths(domain)
        except Exception as e:
            logger.error(f"Failed to get doc paths for {domain}: {e}")
            continue

        manifest_docs = []
        for doc_id, file_path, file_type in doc_entries:
            if file_type != "pdf":
                logger.info(f"Skipping non-PDF: {doc_id} (type={file_type})")
                continue

            doc_dir = domain_index_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Check if already built
            if (doc_dir / "sections.jsonl").exists() and (doc_dir / "units.jsonl").exists():
                logger.info(f"Index already exists for {doc_id}, skipping")
                manifest_docs.append({"doc_id": doc_id, "path": str(doc_dir)})
                continue

            logger.info(f"Parsing {doc_id} from {file_path}...")
            try:
                parser = HiKEYFinanceParser(str(file_path), str(doc_dir))
                paths = parser.parse(with_tables=True, add_vlm_stubs=False)

                # Rename outputs to standard names (parser uses doc_id prefix)
                for src_name, dst_name in [
                    (f"{doc_id}_sections.jsonl", "sections.jsonl"),
                    (f"{doc_id}_units.jsonl", "units.jsonl"),
                    (f"{doc_id}_field_cards.jsonl", "field_cards.jsonl"),
                    (f"{doc_id}_doc_card.json", "doc_card.json"),
                ]:
                    src = doc_dir / src_name
                    dst = doc_dir / dst_name
                    if src.exists() and src != dst:
                        shutil.move(str(src), str(dst))

                manifest_docs.append({"doc_id": doc_id, "path": str(doc_dir)})
                logger.info(f"Built index for {doc_id}")
            except Exception as e:
                logger.error(f"Failed to build index for {doc_id}: {e}")

        # Write manifest
        manifest = {
            "domain": domain,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "documents": manifest_docs,
        }
        with open(domain_index_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Domain {domain}: indexed {len(manifest_docs)} documents")
