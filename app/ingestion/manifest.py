"""The document manifest: which TCS documents we use and where they come from."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.config import get_settings

DocType = Literal["financial_report", "earnings_call_transcript"]


class DocumentEntry(BaseModel):
    id: str
    quarter: str
    period_end: str
    type: DocType
    title: str
    url: str
    filename: str

    @property
    def path(self) -> Path:
        return get_settings().raw_dir / self.filename

    @property
    def exists(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0


def load_manifest() -> list[DocumentEntry]:
    data = json.loads(get_settings().manifest_path.read_text(encoding="utf-8"))
    docs = [DocumentEntry(**d) for d in data["documents"]]
    return sorted(docs, key=lambda d: d.period_end)


def documents_of_type(doc_type: DocType) -> list[DocumentEntry]:
    return [d for d in load_manifest() if d.type == doc_type]
