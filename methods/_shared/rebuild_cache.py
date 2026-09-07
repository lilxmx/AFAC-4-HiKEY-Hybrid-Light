#!/usr/bin/env python3
"""
Rebuild the shared document cache with the new human-readable directory structure.

New layout:
    _cache/{domain}/{doc_id}/
        fulltext.txt          - concatenated full text
        page_001.txt          - block for page 1 (or section 1)
        page_002.txt          - block for page 2
        ...
        metadata.json         - block metadata (section_title per page)

Usage:
    python -m methods._shared.rebuild_cache [--domain DOMAIN] [--clear]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from methods._shared.config_base import (
    DOMAINS,
    QUESTION_FILES,
    SHARED_CACHE_ROOT,
)
from methods._shared.parsers.doc_resolver import resolve_doc_path, get_all_doc_paths
from methods._shared.parsers.full_text import (
    clear_cache,
    parse_document_blocks,
    parse_document_fulltext,
)


def collect_doc_ids_from_questions(domain: str) -> set:
    """Collect all doc_ids referenced in question files for a domain."""
    qfile = QUESTION_FILES.get(domain)
    if not qfile or not qfile.exists():
        return set()

    with open(qfile, "r", encoding="utf-8") as f:
        questions = json.load(f)

    doc_ids = set()
    for q in questions:
        for doc_id in q.get("doc_ids", []):
            doc_ids.add(doc_id)
    return doc_ids


def rebuild_domain(domain: str, verbose: bool = True) -> dict:
    """Rebuild cache for a single domain. Returns stats."""
    stats = {"domain": domain, "total": 0, "success": 0, "failed": 0, "skipped": 0}

    # Collect doc_ids from questions
    doc_ids = collect_doc_ids_from_questions(domain)

    # Also include all docs found on disk (in case some aren't in questions)
    all_docs = get_all_doc_paths(domain)
    for doc_id, _, _ in all_docs:
        doc_ids.add(doc_id)

    stats["total"] = len(doc_ids)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Domain: {domain} ({len(doc_ids)} documents)")
        print(f"{'='*60}")

    for i, doc_id in enumerate(sorted(doc_ids), 1):
        file_path, file_type = resolve_doc_path(doc_id, domain)
        if file_path is None or not file_path.exists():
            if verbose:
                print(f"  [{i}/{len(doc_ids)}] SKIP (not found): {doc_id}")
            stats["skipped"] += 1
            continue

        try:
            # Parse blocks (this also writes the cache)
            blocks = parse_document_blocks(doc_id, domain, use_cache=True)
            if not blocks:
                if verbose:
                    print(f"  [{i}/{len(doc_ids)}] WARN (empty): {doc_id}")
                stats["failed"] += 1
                continue

            # Generate fulltext (this also writes the cache)
            fulltext = parse_document_fulltext(doc_id, domain, use_cache=True)

            if verbose:
                text_len = len(fulltext)
                print(f"  [{i}/{len(doc_ids)}] OK: {doc_id} "
                      f"({len(blocks)} blocks, {text_len:,} chars)")
            stats["success"] += 1

        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(doc_ids)}] ERROR: {doc_id} -> {e}")
            stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rebuild document cache")
    parser.add_argument("--domain", type=str, default=None,
                        help="Only rebuild a specific domain")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing cache before rebuilding")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-document output")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS

    # Validate domains
    for d in domains:
        if d not in DOMAINS:
            print(f"ERROR: Unknown domain '{d}'. Valid: {DOMAINS}")
            sys.exit(1)

    if args.clear:
        print("Clearing existing cache...")
        for d in domains:
            clear_cache(d)
        print("Done.")

    print(f"\nRebuilding cache for domains: {domains}")
    print(f"Cache root: {SHARED_CACHE_ROOT}")

    start_time = time.time()
    all_stats = []

    for domain in domains:
        stats = rebuild_domain(domain, verbose=not args.quiet)
        all_stats.append(stats)

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total_docs = sum(s["total"] for s in all_stats)
    total_success = sum(s["success"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_skipped = sum(s["skipped"] for s in all_stats)

    for s in all_stats:
        print(f"  {s['domain']:25s}: {s['success']}/{s['total']} OK, "
              f"{s['failed']} failed, {s['skipped']} skipped")

    print(f"\n  Total: {total_success}/{total_docs} OK, "
          f"{total_failed} failed, {total_skipped} skipped")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Cache location: {SHARED_CACHE_ROOT}")


if __name__ == "__main__":
    main()
