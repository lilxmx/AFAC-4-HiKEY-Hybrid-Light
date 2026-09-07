#!/usr/bin/env python3
"""
build_hikey_cache.py - Concurrent HiKEY index builder for ALL PDF documents.

Scans all PDFs under public_dataset_upload/raw/ and builds HiKEY hierarchical
indices (sections, units, field_cards, doc_card) in parallel.

Output structure (mirrors _shared/_cache):
    _shared/_HiKEY_cache/{domain}/{doc_id}/
        ├── doc_card.json
        ├── sections.jsonl
        ├── units.jsonl
        ├── field_cards.jsonl
        ├── field_cards.csv
        ├── table_rows.csv
        └── parse_report.md

Usage:
    # Build all domains (default 4 workers)
    python build_hikey_cache.py

    # Build specific domain with more workers
    python build_hikey_cache.py --domains financial_reports --workers 8

    # Force rebuild (ignore existing)
    python build_hikey_cache.py --force
"""
import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from methods._shared.config_base import RAW_BASE
from methods._shared.parsers import get_all_doc_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Output root
HIKEY_CACHE_ROOT = _PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"

DOMAINS_WITH_PDF = ["financial_reports", "financial_contracts", "insurance", "research", "regulatory"]


def build_single_doc(args_tuple):
    """Build HiKEY index for a single PDF document. Designed for ProcessPoolExecutor."""
    doc_id, pdf_path, domain, output_dir, force = args_tuple

    # Lazy import inside worker process
    import sys as _sys
    _sys.path.insert(0, str(_HERE))
    _sys.path.insert(0, str(_PROJECT_ROOT))
    from hikey_financial_parser import HiKEYFinanceParser

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already built (unless force)
    if not force and (output_dir / "sections.jsonl").exists() and (output_dir / "units.jsonl").exists():
        return {"doc_id": doc_id, "domain": domain, "status": "skipped", "msg": "already exists"}

    try:
        t0 = time.time()
        parser = HiKEYFinanceParser(str(pdf_path), str(output_dir))
        parser.parse(with_tables=True, add_vlm_stubs=False)
        elapsed = time.time() - t0

        # Rename outputs: parser uses {doc_id}_xxx.jsonl, we want standard names
        renames = [
            (f"{doc_id}_doc_card.json", "doc_card.json"),
            (f"{doc_id}_sections.jsonl", "sections.jsonl"),
            (f"{doc_id}_units.jsonl", "units.jsonl"),
            (f"{doc_id}_field_cards.jsonl", "field_cards.jsonl"),
            (f"{doc_id}_field_cards.csv", "field_cards.csv"),
            (f"{doc_id}_table_rows.csv", "table_rows.csv"),
            (f"{doc_id}_parse_report.md", "parse_report.md"),
        ]
        for src_name, dst_name in renames:
            src = output_dir / src_name
            dst = output_dir / dst_name
            if src.exists() and src != dst:
                shutil.move(str(src), str(dst))

        # Count results
        n_sections = sum(1 for _ in open(output_dir / "sections.jsonl", encoding="utf-8")) if (output_dir / "sections.jsonl").exists() else 0
        n_units = sum(1 for _ in open(output_dir / "units.jsonl", encoding="utf-8")) if (output_dir / "units.jsonl").exists() else 0
        n_fields = sum(1 for _ in open(output_dir / "field_cards.jsonl", encoding="utf-8")) if (output_dir / "field_cards.jsonl").exists() else 0

        return {
            "doc_id": doc_id,
            "domain": domain,
            "status": "success",
            "elapsed": round(elapsed, 1),
            "sections": n_sections,
            "units": n_units,
            "field_cards": n_fields,
        }
    except Exception as e:
        return {"doc_id": doc_id, "domain": domain, "status": "error", "msg": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Build HiKEY cache for all PDF documents (concurrent)")
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to build (default: all with PDFs)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    domains = args.domains or DOMAINS_WITH_PDF
    HIKEY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect all PDF tasks
    tasks = []
    for domain in domains:
        doc_entries = get_all_doc_paths(domain)
        pdf_entries = [(doc_id, path, ftype) for doc_id, path, ftype in doc_entries if ftype == "pdf"]
        logger.info(f"Domain [{domain}]: found {len(pdf_entries)} PDFs")

        for doc_id, pdf_path, _ in pdf_entries:
            output_dir = HIKEY_CACHE_ROOT / domain / doc_id
            tasks.append((doc_id, str(pdf_path), domain, str(output_dir), args.force))

    logger.info(f"Total tasks: {len(tasks)}, Workers: {args.workers}")
    if not tasks:
        logger.info("No PDFs found. Exiting.")
        return

    # Execute concurrently
    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_single_doc, task): task[0] for task in tasks}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            results.append(result)
            status = result["status"]
            doc_id = result["doc_id"]
            domain = result["domain"]
            if status == "success":
                logger.info(
                    f"[{done_count}/{len(tasks)}] ✓ {domain}/{doc_id} "
                    f"({result['elapsed']}s, sec={result['sections']}, "
                    f"units={result['units']}, fields={result['field_cards']})"
                )
            elif status == "skipped":
                logger.info(f"[{done_count}/{len(tasks)}] ⊘ {domain}/{doc_id} (skipped)")
            else:
                logger.error(f"[{done_count}/{len(tasks)}] ✗ {domain}/{doc_id}: {result.get('msg', 'unknown error')}")

    elapsed_total = time.time() - t_start

    # Write manifest per domain
    for domain in domains:
        domain_results = [r for r in results if r["domain"] == domain and r["status"] in ("success", "skipped")]
        manifest = {
            "domain": domain,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_docs": len(domain_results),
            "documents": [{"doc_id": r["doc_id"], "path": str(HIKEY_CACHE_ROOT / domain / r["doc_id"])} for r in domain_results],
        }
        manifest_path = HIKEY_CACHE_ROOT / domain / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    logger.info(f"\n{'='*60}")
    logger.info(f"BUILD COMPLETE in {elapsed_total:.1f}s")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Errors:  {errors}")
    logger.info(f"  Output:  {HIKEY_CACHE_ROOT}")
    logger.info(f"{'='*60}")

    if errors:
        logger.info("\nFailed documents:")
        for r in results:
            if r["status"] == "error":
                logger.info(f"  {r['domain']}/{r['doc_id']}: {r.get('msg', '')}")


if __name__ == "__main__":
    main()
