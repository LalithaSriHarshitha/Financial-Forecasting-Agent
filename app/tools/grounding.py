"""Grounding checks: is a quote really in the transcripts?

An LLM can paraphrase, merge two sentences, or invent a quote outright. Every quote the
system returns is checked against the transcript chunks: at least 80% of its content
words must appear in one chunk (from the same quarter when a quarter is given). Quotes
that fail are removed and the removal is reported.
"""

import re

from langchain_core.documents import Document

from app.ingestion.index import tokenize

MIN_COVERAGE = 0.8
_PREFIX = re.compile(r"\[[^\]]*\]:\s*")  # "[Speaker (role, section)]: " labels added at chunking


def _words(text: str) -> set[str]:
    return set(tokenize(_PREFIX.sub(" ", text)))


def best_match(quote: str, docs: list[Document], quarter: str | None = None) -> tuple[float, Document | None]:
    q = _words(quote)
    if not q:
        return 0.0, None
    best, best_doc = 0.0, None
    for d in docs:
        if quarter and d.metadata.get("quarter") != quarter:
            continue
        cov = len(q & _words(d.page_content)) / len(q)
        if cov > best:
            best, best_doc = cov, d
    return best, best_doc


def verify_quote(quote: str, docs: list[Document], quarter: str | None = None) -> tuple[bool, Document | None]:
    """True if the quote is supported by a chunk (from `quarter`, if given)."""
    cov, doc = best_match(quote, docs, quarter)
    if cov < MIN_COVERAGE and quarter:  # right words, wrong quarter label: look in every quarter
        cov, doc = best_match(quote, docs)
    return cov >= MIN_COVERAGE, doc if cov >= MIN_COVERAGE else None
