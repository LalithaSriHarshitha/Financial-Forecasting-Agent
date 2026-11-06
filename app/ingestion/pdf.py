"""PDF text extraction."""

from functools import lru_cache
from pathlib import Path

import pdfplumber


@lru_cache(maxsize=32)
def _read_pages(path_str: str, mtime: float) -> tuple[str, ...]:
    with pdfplumber.open(path_str) as pdf:
        return tuple((page.extract_text() or "") for page in pdf.pages)


def read_pages(path: Path) -> list[str]:
    """Return the text of each page (index 0 = page 1). Cached per file version."""
    return list(_read_pages(str(path), path.stat().st_mtime))
