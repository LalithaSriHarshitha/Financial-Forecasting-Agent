"""Build and query the Chroma vector index of earnings-call transcripts."""

import logging
import re
import shutil

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
    batch = 50  # keep embedding requests small for free-tier rate limits
    for start in range(0, len(chunks), batch):
        store.add_documents(chunks[start : start + batch], ids=ids[start : start + batch])
    logger.info("Indexed %d transcript chunks into %s", len(chunks), path)
    return len(chunks)
