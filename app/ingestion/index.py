"""Build and query the Chroma vector index of earnings-call transcripts."""

import logging
import re
import shutil
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.manifest import documents_of_type
from app.ingestion.pdf import read_pages
from app.llm import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION = "tcs_transcripts"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_transcript_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Document] = []
    for doc in documents_of_type("earnings_call_transcript"):
        if not doc.exists:
            logger.warning("Skipping %s: file not found at %s", doc.id, doc.path)
            continue
        for page_no, text in enumerate(read_pages(doc.path), start=1):
            text = _clean(text)
            if not text:
                continue
            for i, piece in enumerate(splitter.split_text(text)):
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "doc_id": doc.id,
                            "quarter": doc.quarter,
                            "period_end": doc.period_end,
                            "page": page_no,
                            "chunk": i,
                            "source": doc.title,
                        },
                    )
                )
    return chunks


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(get_settings().chroma_path),
    )


def build_index(rebuild: bool = True) -> int:
    """(Re)build the transcript index. Returns the number of chunks indexed."""
    path = get_settings().chroma_path
    if rebuild and path.exists():
        shutil.rmtree(path)
    chunks = build_transcript_chunks()
    if not chunks:
        raise RuntimeError("No transcript text found. Download the transcripts first.")
    store = get_vector_store()
    ids = [f"{c.metadata['doc_id']}-p{c.metadata['page']}-c{c.metadata['chunk']}" for c in chunks]
    per_minute = get_settings().embed_requests_per_minute
    batch = max(1, min(20, per_minute))
    window_start, used = time.monotonic(), 0
    for start in range(0, len(chunks), batch):
        part = slice(start, start + batch)
        n = len(chunks[part])
        # Pace requests: each chunk counts as one embed request on the free tier.
        if used + n > per_minute:
            wait = 60 - (time.monotonic() - window_start)
            if wait > 0:
                logger.info("Pausing %.0fs to stay under %d embeddings/minute ...", wait, per_minute)
                time.sleep(wait + 1)
            window_start, used = time.monotonic(), 0
        _add_with_retry(store, chunks[part], ids[part])
        used += n
        logger.info("Embedded %d/%d chunks", min(start + batch, len(chunks)), len(chunks))
    logger.info("Indexed %d transcript chunks into %s", len(chunks), path)
    return len(chunks)


def _retry_delay(exc: Exception, default: float = 30.0) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(m.group(1)) + 2 if m else default


def _add_with_retry(store: Chroma, docs: list[Document], ids: list[str], attempts: int = 5) -> None:
    for attempt in range(1, attempts + 1):
        try:
            store.add_documents(docs, ids=ids)
            return
        except Exception as exc:
            if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc) or attempt == attempts:
                raise
            delay = _retry_delay(exc)
            logger.warning("Rate limited by Gemini (attempt %d/%d); retrying in %.0fs", attempt, attempts, delay)
            time.sleep(delay)
