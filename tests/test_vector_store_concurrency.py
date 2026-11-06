"""Retrieval tests: thread safety (parallel tool calls) and per-quarter coverage."""

import importlib
import json
import threading

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.ingestion import index

qa = importlib.import_module("app.tools.qualitative_analysis")


@pytest.fixture
def small_index(monkeypatch, tmp_path):
    monkeypatch.setattr(index, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    monkeypatch.setattr(index.get_settings(), "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(qa, "_transcript_quarters", lambda: ["Q1 FY27", "Q2 FY27"])
    docs = [
        Document(page_content=f"{q} chunk {i}", metadata={"doc_id": q, "quarter": q, "page": i, "chunk": 0})
        for q in ("Q1 FY27", "Q2 FY27")
        for i in range(6)
    ]
    index.get_vector_store().add_documents(docs)
    index.reset_shared_vector_store()
    yield
    index.reset_shared_vector_store()


def test_parallel_retrieval_is_thread_safe(small_index):
    errors = []
    for _ in range(10):
        index.reset_shared_vector_store()

        def go():
            try:
                assert qa.retrieve(["demand"], "Q2 FY27")
            except Exception as exc:  # pragma: no cover - the bug being guarded against
                errors.append(exc)

        threads = [threading.Thread(target=go) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


def test_retrieval_covers_every_quarter_without_duplicates(small_index):
    docs = qa.retrieve(["demand", "margins"])
    quarters = {d.metadata["quarter"] for d in docs}
    keys = [(d.metadata["doc_id"], d.metadata["page"]) for d in docs]
    assert quarters == {"Q1 FY27", "Q2 FY27"}
    assert len(keys) == len(set(keys))
    assert [d.metadata["quarter"] for d in docs] == sorted(
        (d.metadata["quarter"] for d in docs), key=lambda q: q.split()[1] + q.split()[0]
    )


def test_tool_errors_are_returned_not_raised(monkeypatch):
    def boom(queries, quarter=None):
        raise RuntimeError("index missing")

    monkeypatch.setattr(qa, "analyze", boom)
    out = json.loads(qa.qualitative_analysis.invoke({"queries": ["risks"]}))
    assert "index missing" in out["error"]
