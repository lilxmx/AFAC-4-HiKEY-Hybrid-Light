"""
Financial Contracts specialized retriever.
Implements doc_id-isolated retrieval with field-pack-based query strategy.
Follows Skill v6.0: per-doc_id retrieval with section priority and synonym expansion.
"""
from typing import List, Dict, Tuple, Optional

from config import (
    BM25_TOP_K, EMBEDDING_TOP_K, RRF_K, FINAL_TOP_K, MAX_EVIDENCE_CHARS
)
from methods._shared.indexer import BM25Index, Chunk
from embedding_index import EmbeddingIndex


# Section keyword mapping for preferred_sections filtering
SECTION_KEYWORDS = {
    "cover": ["封面", "首页", "发行人", "发行金额", "信用评级"],
    "statement": ["声明", "发行人承诺", "信息披露义务"],
    "important_notice": ["重大事项提示", "重大事项", "重要提示"],
    "issuance_overview": ["发行概况", "发行条款", "本期债券主要条款", "发行要素"],
    "definition": ["释义", "定义"],
    "risk_factors": ["风险因素", "风险提示"],
    "fund_usage": ["募集资金用途", "募集资金运用"],
    "issuer_info": ["发行人基本情况", "发行人概况", "公司概况"],
    "credit_info": ["资信状况", "信用情况", "信用评级"],
    "financial_info": ["财务会计信息", "财务数据", "主要财务指标", "财务报表"],
    "investor_protection": ["投资者保护", "债券持有人权利", "持有人会议"],
    "disclosure_arrangement": ["信息披露安排", "信息披露事务"],
    "trustee": ["受托管理人", "债券受托管理人"],
    "institutions": ["发行有关机构", "相关证券服务机构", "中介机构"],
    "transaction_overview": ["本次交易概况", "交易方案", "交易结构"],
    "listed_company_info": ["上市公司基本情况", "上市公司概况"],
    "counterparty_info": ["交易对方基本情况", "交易对方"],
    "target_info": ["交易标的", "标的公司", "标的资产"],
    "share_issuance": ["发行股份情况", "股份发行"],
    "management_discussion": ["管理层讨论与分析", "经营情况讨论"],
}


class FCRetriever:
    """
    Financial Contracts specialized retriever.
    Key differences from generic HybridRetriever:
    1. Retrieves per doc_id in isolation (no cross-doc contamination)
    2. Uses field-pack-based queries instead of raw question
    3. Applies section priority filtering
    4. Returns structured evidence packs per doc_id
    """

    def __init__(self, bm25_index: BM25Index, embedding_index: EmbeddingIndex):
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        # Build lookup dict
        self._chunk_lookup: Dict[tuple, int] = {}
        for i, c in enumerate(embedding_index.chunks):
            self._chunk_lookup[(c.doc_id, c.chunk_id)] = i

    def retrieve_by_plan(self, plan: Dict) -> Dict[str, List[Chunk]]:
        """
        Retrieve evidence based on a structured retrieval plan from QueryPlanner.

        Args:
            plan: Retrieval plan with field_packs

        Returns:
            Dict mapping doc_id -> list of evidence chunks
        """
        doc_evidence: Dict[str, List[Chunk]] = {}

        for field_pack in plan.get('field_packs', []):
            doc_id = field_pack.get('doc_id')
            if not doc_id:
                continue

            # Build queries from field pack
            queries = self._build_queries_from_field_pack(field_pack)
            preferred_sections = field_pack.get('preferred_sections', [])

            # Retrieve for this doc_id only
            chunks = self._retrieve_for_doc(
                doc_id=doc_id,
                queries=queries,
                preferred_sections=preferred_sections,
                top_k_per_query=BM25_TOP_K,
            )

            if doc_id not in doc_evidence:
                doc_evidence[doc_id] = []
            doc_evidence[doc_id].extend(chunks)

        # Deduplicate and rank per doc_id
        for doc_id in doc_evidence:
            doc_evidence[doc_id] = self._deduplicate_chunks(doc_evidence[doc_id])

        return doc_evidence

    def retrieve_with_backoff(self, doc_id: str, question: str,
                              options: Dict[str, str]) -> List[Chunk]:
        """
        Backoff retrieval: when field-pack retrieval is insufficient,
        fall back to question+option based retrieval for a specific doc_id.

        Args:
            doc_id: Document ID to search within
            question: Original question text
            options: Option dict

        Returns:
            List of evidence chunks
        """
        queries = [question]
        for opt_key, opt_text in sorted(options.items()):
            queries.append(f"{question} {opt_text}")

        return self._retrieve_for_doc(
            doc_id=doc_id,
            queries=queries,
            preferred_sections=[],
            top_k_per_query=BM25_TOP_K,
        )

    def _retrieve_for_doc(self, doc_id: str, queries: List[str],
                          preferred_sections: List[str],
                          top_k_per_query: int = 10) -> List[Chunk]:
        """
        Retrieve chunks for a single doc_id using hybrid BM25 + Embedding.

        Args:
            doc_id: Target document ID
            queries: List of query strings
            preferred_sections: Preferred section names for boosting
            top_k_per_query: Top-k per individual query

        Returns:
            Ranked list of chunks from this doc_id only
        """
        doc_ids_filter = [doc_id]

        # Collect BM25 rankings
        bm25_rankings = []
        for query in queries:
            results = self.bm25_index.search(
                query, top_k=top_k_per_query, doc_ids=doc_ids_filter
            )
            ranked_indices = []
            for chunk, score in results:
                idx = self._chunk_lookup.get((chunk.doc_id, chunk.chunk_id))
                if idx is not None:
                    ranked_indices.append(idx)
            bm25_rankings.append(ranked_indices)

        # Collect Embedding rankings
        embedding_rankings = []
        for query in queries:
            results = self.embedding_index.search(
                query, top_k=top_k_per_query, doc_ids=doc_ids_filter
            )
            ranked_indices = [idx for idx, score in results]
            embedding_rankings.append(ranked_indices)

        # RRF fusion
        all_rankings = bm25_rankings + embedding_rankings
        fused_scores = self._rrf_fusion(all_rankings)

        # Apply section boost
        if preferred_sections:
            fused_scores = self._apply_section_boost(fused_scores, preferred_sections)

        # Sort and get top chunks
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        # More generous top-k per doc (will be truncated later)
        top_indices = [idx for idx, score in sorted_items[:FINAL_TOP_K + 4]]

        evidence_chunks = []
        for idx in top_indices:
            if 0 <= idx < len(self.embedding_index.chunks):
                chunk = self.embedding_index.chunks[idx]
                if chunk.doc_id == doc_id:  # Safety check
                    evidence_chunks.append(chunk)

        return evidence_chunks

    def _build_queries_from_field_pack(self, field_pack: Dict) -> List[str]:
        """
        Build multiple query strings from a field pack.

        Args:
            field_pack: Field pack dict with query_keywords and fields

        Returns:
            List of query strings
        """
        queries = []

        # Primary query from keywords
        keywords = field_pack.get('query_keywords', '')
        if keywords:
            queries.append(keywords)

        # Build field-specific queries
        fields = field_pack.get('fields', [])
        field_query_map = {
            'issuer': '发行人 发行主体 公司名称',
            'current_issue_amount_max': '本期债券发行金额 发行规模 不超过',
            'registered_amount_max': '注册金额 注册额度 不超过',
            'total_issue_amount': '本次发行总额 发行总额',
            'subject_rating': '主体信用评级 主体评级 信用评级结果',
            'bond_rating': '债项信用评级 债项评级',
            'trustee': '受托管理人 债券受托管理人',
            'lead_underwriter': '牵头主承销商 主承销商',
            'bookrunner': '簿记管理人',
            'disclosure_commitment': '发行人承诺 及时 公平 信息披露义务',
            'investor_protection': '投资者保护 债券持有人权利 持有人会议',
            'put_option': '回售 回售条款 回售选择权 有条件回售 附加回售',
            'redemption': '赎回 赎回条款 有条件赎回 到期赎回',
            'asset_liability_ratio': '资产负债率 合并口径 母公司口径',
            'net_profit': '净利润 归属于母公司 年均可分配利润',
            'conversion_price': '转股价格 初始转股价 转股价格向下修正',
            'conversion_period': '转股期限 转股起始日',
            'conditional_redemption': '有条件赎回 赎回条款',
            'conditional_put': '有条件回售 回售条款',
            'transaction_structure': '发行股份购买资产 募集配套资金 交易结构 交易方案',
            'target_company': '标的公司 标的资产 交易标的',
            'transaction_counterparty': '交易对方 交易对手方',
            'default_clause': '违约 违约事件 违约情形',
            'penalty_interest': '违约利息 罚息 赔偿 补偿 惩罚系数',
            'stock_code': '股票代码 证券代码',
            'stock_name': '股票简称 证券简称',
            'issuance_date': '发行日期 发行公告日',
            'maturity_date': '兑付日 到期日 偿还日',
        }

        # Group fields into a combined query
        field_keywords = []
        for field in fields:
            if field in field_query_map:
                field_keywords.append(field_query_map[field])

        if field_keywords and not keywords:
            # If no explicit keywords, use field-based query
            queries.append(" ".join(field_keywords))

        # Also add individual field queries for better recall
        for field in fields[:3]:  # Limit to top 3 fields
            if field in field_query_map:
                queries.append(field_query_map[field])

        return queries if queries else ["发行人 发行金额 信用评级"]

    def _rrf_fusion(self, rankings: List[List[int]]) -> Dict[int, float]:
        """Apply Reciprocal Rank Fusion."""
        scores = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                if idx not in scores:
                    scores[idx] = 0.0
                scores[idx] += 1.0 / (RRF_K + rank + 1)
        return scores

    def _apply_section_boost(self, scores: Dict[int, float],
                             preferred_sections: List[str]) -> Dict[int, float]:
        """
        Boost scores for chunks that match preferred sections.

        Args:
            scores: Current RRF scores
            preferred_sections: List of preferred section identifiers

        Returns:
            Boosted scores
        """
        # Collect section keywords
        boost_keywords = set()
        for section in preferred_sections:
            if section in SECTION_KEYWORDS:
                boost_keywords.update(SECTION_KEYWORDS[section])

        if not boost_keywords:
            return scores

        boosted = {}
        for idx, score in scores.items():
            if 0 <= idx < len(self.embedding_index.chunks):
                chunk = self.embedding_index.chunks[idx]
                # Check if chunk's section_title matches any boost keyword
                section_match = False
                if chunk.section_title:
                    for kw in boost_keywords:
                        if kw in chunk.section_title:
                            section_match = True
                            break
                # Also check page_num (page 1-2 often = cover)
                if chunk.page_num <= 2 and "cover" in preferred_sections:
                    section_match = True

                if section_match:
                    boosted[idx] = score * 1.5  # 50% boost for section match
                else:
                    boosted[idx] = score
            else:
                boosted[idx] = score

        return boosted

    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks based on (doc_id, chunk_id)."""
        seen = set()
        result = []
        for chunk in chunks:
            key = (chunk.doc_id, chunk.chunk_id)
            if key not in seen:
                seen.add(key)
                result.append(chunk)
        return result


def format_evidence_by_doc(doc_evidence: Dict[str, List[Chunk]],
                           max_chars_per_doc: int = 4000) -> str:
    """
    Format evidence organized by document.

    Args:
        doc_evidence: Dict mapping doc_id -> list of chunks
        max_chars_per_doc: Max characters per document's evidence

    Returns:
        Formatted evidence string
    """
    if not doc_evidence:
        return "（未找到相关证据）"

    parts = []
    for doc_id in sorted(doc_evidence.keys()):
        chunks = doc_evidence[doc_id]
        if not chunks:
            continue

        doc_parts = [f"═══ 文档: {doc_id} ═══"]
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            header = f"[{doc_id}-证据{i}]"
            if chunk.section_title:
                header += f" 章节: {chunk.section_title}"
            if chunk.page_num > 0:
                header += f" | 页码: {chunk.page_num}"

            text = chunk.text
            if total_chars + len(text) > max_chars_per_doc:
                remaining = max_chars_per_doc - total_chars
                if remaining > 200:
                    text = text[:remaining] + "..."
                else:
                    break

            doc_parts.append(f"{header}\n{text}")
            total_chars += len(text)

        parts.append("\n\n".join(doc_parts))

    return "\n\n" + "─" * 40 + "\n\n".join(parts)
