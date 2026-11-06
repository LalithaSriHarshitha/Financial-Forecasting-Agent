"""PDF text extraction (PyMuPDF).

Why PyMuPDF instead of pdfplumber: the TCS fact sheets are PowerPoint exports that
draw a phantom space glyph on top of the first digit of every number, so pdfplumber
reads "65,799" as "6 5,799"; one fact sheet also draws its income statement as
rotated text. PyMuPDF reads both correctly. Text is NFKC-normalised so ligatures
such as "ﬁ"/"ﬃ" become plain "fi"/"ffi".
"""

import unicodedata
from functools import lru_cache
from pathlib import Path

import pymupdf


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace(" ", " ").replace("’", "'").replace("‘", "'")


@lru_cache(maxsize=32)
def _read_pages(path_str: str, mtime: float, layout: bool) -> tuple[str, ...]:
    with pymupdf.open(path_str) as doc:
        return tuple(_normalize(page.get_text("text", sort=layout)) for page in doc)


def read_pages(path: Path, layout: bool = True) -> list[str]:
    """Text of each page (index 0 = page 1). `layout=True` keeps table rows on one line.
    Cached per file version."""
    return list(_read_pages(str(path), path.stat().st_mtime, layout))
