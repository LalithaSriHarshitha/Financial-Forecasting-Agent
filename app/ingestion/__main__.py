"""Ingestion CLI.

    python -m app.ingestion              # download documents + build the transcript index
    python -m app.ingestion --skip-download
    python -m app.ingestion --force-download
"""

import argparse
import logging
import sys

from app.config import get_settings
from app.ingestion.fetch import download_all
from app.ingestion.index import build_index
from app.ingestion.manifest import load_manifest
from app.logging_config import setup_logging

logger = logging.getLogger("ingestion")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TCS documents and build the RAG index.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    setup_logging(get_settings().log_level)

    if not args.skip_download:
        logger.info("Step 1/2: downloading documents listed in data/manifest.json")
        download_all(force=args.force_download)

    missing = [d for d in load_manifest() if not d.exists]
    if missing:
        logger.warning("Missing files: %s", ", ".join(d.filename for d in missing))

    logger.info("Step 2/2: building the transcript vector index (%s embeddings: %s)",
                get_settings().llm_provider, get_settings().embedding_model_name)
    try:
        n = build_index()
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        return 1
    logger.info("Done. %d chunks indexed. Start the API with: uvicorn app.main:app", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
