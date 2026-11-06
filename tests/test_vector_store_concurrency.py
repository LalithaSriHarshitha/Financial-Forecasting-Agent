"""Regression test: parallel tool calls must not create Chroma clients concurrently."""

import importlib
import json
import threading

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.ingestion import index

qa = importlib.import_module("app.tools.qualitative_analysis")


def test_parallel_retrieval_is_thread_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(index, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    monkeypatch.setattr(index.get_settings(), "chroma_dir", tmp_path / "chroma")
    index.get_vector_store().add_documents(
        [Document(page_content=f"chunk {i}", metadata={"quarter": "Q2 FY27", "page": 1}) for i in range(10)]
    )
    errors = []
    for _ in range(10):
        index.reset_shared_vector_store()

        def go():
            try:
                assert qa.retrieve("demand", "Q2 FY27")
            except Exception as exc:  # pragma: no cover - the bug being guarded against
                errors.append(exc)

        threads = [threading.Thread(target=go) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    index.reset_shared_vector_store()
    assert errors == []


def test_tool_errors_are_returned_not_raised(monkeypatch):
    def boom(query, quarter=None):
        raise RuntimeError("index missing")

    monkeypatch.setattr(qa, "analyze", boom)
    out = json.loads(qa.qualitative_analysis.invoke({"query": "risks"}))
    assert "index missing" in out["error"]
