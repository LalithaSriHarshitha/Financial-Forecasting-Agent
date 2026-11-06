"""Helpers for TCS fiscal-quarter labels such as 'Q2 FY27'."""

import re

_PATTERN = re.compile(r"Q\s*([1-4])\s*FY\s*'?(\d{2,4})", re.IGNORECASE)


def parse_quarter(label: str) -> tuple[int, int] | None:
    """'Q2 FY27' -> (27, 2). Returns None if the label is not recognised."""
    m = _PATTERN.search(label or "")
    if not m:
        return None
    fy = int(m.group(2)) % 100
    return fy, int(m.group(1))


def normalize_quarter(label: str) -> str:
    parsed = parse_quarter(label)
    if not parsed:
        return label.strip()
    fy, q = parsed
    return f"Q{q} FY{fy:02d}"


def quarter_sort_key(label: str) -> tuple[int, int]:
    return parse_quarter(label) or (0, 0)


def previous_quarter(label: str, n: int = 1) -> str | None:
    parsed = parse_quarter(label)
    if not parsed:
        return None
    fy, q = parsed
    index = fy * 4 + (q - 1) - n
    return f"Q{index % 4 + 1} FY{index // 4:02d}"


def next_quarter(label: str) -> str | None:
    return previous_quarter(label, -1)
