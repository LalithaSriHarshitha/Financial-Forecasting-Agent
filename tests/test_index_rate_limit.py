from langchain_core.documents import Document

from app.ingestion import index


class FakeStore:
    def __init__(self, fail_times=0):
        self.added, self.fail_times = [], fail_times

    def add_documents(self, docs, ids):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 3.5s.")
        self.added.extend(ids)


def _chunks(n):
    return [Document(page_content="x", metadata={"doc_id": "d", "page": 1, "chunk": i}) for i in range(n)]


def test_build_index_paces_under_quota(monkeypatch, tmp_path):
    store, sleeps = FakeStore(), []
    monkeypatch.setattr(index, "build_transcript_chunks", lambda: _chunks(50))
    monkeypatch.setattr(index, "get_vector_store", lambda: store)
    monkeypatch.setattr(index.time, "sleep", sleeps.append)
    settings = index.get_settings()
    monkeypatch.setattr(settings, "embed_requests_per_minute", 20)
    monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")

    assert index.build_index() == 50
    assert len(store.added) == 50
    assert len(sleeps) == 2  # 50 chunks at 20/minute -> two pauses


def test_add_with_retry_waits_for_suggested_delay(monkeypatch):
    store, sleeps = FakeStore(fail_times=2), []
    monkeypatch.setattr(index.time, "sleep", sleeps.append)
    index._add_with_retry(store, _chunks(1), ["a"])
    assert store.added == ["a"]
    assert sleeps == [5.5, 5.5]
