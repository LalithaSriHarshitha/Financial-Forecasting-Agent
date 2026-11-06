"""Build and query the Chroma vector index of earnings-call transcripts."""

import logging
import re
import shutil
import threading
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion.manifest import documents_of_type
from app.ingestion.pdf import read_pages
from app.ingestion.transcripts import chunk_transcript
from app.llm import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION = "tcs_transcripts"


def chunk_id(meta: dict) -> str:
    return f"{meta['doc_id']}-t{meta.get('turn', 0)}-c{meta.get('chunk', 0)}"


def build_transcript_chunks() -> list[Document]:
    """Speaker-aware chunks for every transcript in the manifest."""
    chunks: list[Document] = []
    for doc in documents_of_type("earnings_call_transcript"):
        if not doc.exists:
            logger.warning("Skipping %s: file not found at %s", doc.id, doc.path)
            continue
        pages = read_pages(doc.path, layout=False)
        doc_chunks = chunk_transcript(pages, doc.id, doc.quarter, doc.period_end, doc.title)
        logger.info("%s: %d chunks from %d speaker turns", doc.id, len(doc_chunks),
                    len({c.metadata["turn"] for c in doc_chunks}))
        chunks.extend(doc_chunks)
    return chunks


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(get_settings().chroma_path),
    )


_shared_store: Chroma | None = None
_store_lock = threading.Lock()


def get_shared_vector_store() -> Chroma:
    """Process-wide Chroma client for queries.

    The agent runs tool calls in parallel threads, and creating several Chroma
    clients at the same moment is not thread-safe, so the client is created once
    under a lock and reused.
    """
    global _shared_store
    if _shared_store is None:
        with _store_lock:
            if _shared_store is None:
                _shared_store = get_vector_store()
    return _shared_store


def reset_shared_vector_store() -> None:
    global _shared_store, _shared_retriever
    with _store_lock:
        _shared_store = None
        _shared_retriever = None


# ---------------- Hybrid retrieval (BM25 + vectors, reciprocal rank fusion) ----------------

_STOPWORDS = set(
    "a an and are as at be but by for from has have i in is it its of on or our so that the their "
    "there these this to was we were what which will with you your".split()
)
RRF_K = 60  # standard reciprocal-rank-fusion constant


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS]


class HybridRetriever:
    """Keyword (BM25) and semantic (vector) search, merged with reciprocal rank fusion.

    BM25 catches exact terms (names, "MHP", "furlough", "TCV"); vectors catch paraphrases.
    """

    def __init__(self, store: Chroma):
        self.store = store
        data = store.get(include=["documents", "metadatas"])
        self.docs = [Document(page_content=d, metadata=m) for d, m in zip(data["documents"], data["metadatas"])]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs]) if self.docs else None

    @staticmethod
    def _matches(meta: dict, quarter: str | None, role: str | None) -> bool:
        return (quarter is None or meta.get("quarter") == quarter) and (role is None or meta.get("role") == role)

    def search(self, query: str, vector: list[float], k: int, quarter: str | None = None,
               role: str | None = None, pool: int = 12) -> list[Document]:
        conditions = [{"quarter": quarter}] if quarter else []
        if role:
            conditions.append({"role": role})
        flt = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
        vector_hits = self.store.similarity_search_by_vector(vector, k=pool, filter=flt)

        keyword_hits: list[Document] = []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenize(query))
            ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
            keyword_hits = [self.docs[i] for i in ranked
                            if scores[i] > 0 and self._matches(self.docs[i].metadata, quarter, role)][:pool]

        fused: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for hits in (vector_hits, keyword_hits):
            for rank, d in enumerate(hits):
                cid = chunk_id(d.metadata)
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                by_id.setdefault(cid, d)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [by_id[c] for c in best]


_shared_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _shared_retriever
    if _shared_retriever is None:
        store = get_shared_vector_store()
        with _store_lock:
            if _shared_retriever is None:
                _shared_retriever = HybridRetriever(store)
    return _shared_retriever


def build_index(rebuild: bool = True) -> int:
    """(Re)build the transcript index. Returns the number of chunks indexed."""
    path = get_settings().chroma_path
    if rebuild and path.exists():
        shutil.rmtree(path)
    chunks = build_transcript_chunks()
    if not chunks:
        raise RuntimeError("No transcript text found. Download the transcripts first.")
    store = get_vector_store()
    ids = [chunk_id(c.metadata) for c in chunks]
    s = get_settings()
    # Gemini free tier counts every chunk as a request; local Ollama needs no pacing.
    per_minute = s.embed_requests_per_minute if s.llm_provider == "gemini" else 10**9
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
            logger.warning("Rate limited by the embedding API (attempt %d/%d); retrying in %.0fs", attempt, attempts, delay)
            time.sleep(delay)
