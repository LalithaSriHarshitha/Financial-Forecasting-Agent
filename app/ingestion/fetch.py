"""Download the documents listed in data/manifest.json into data/raw/."""

import logging
from urllib.parse import urlparse

import requests

from app.ingestion.manifest import DocumentEntry, load_manifest

logger = logging.getLogger(__name__)

# BSE and tcs.com reject requests without browser-like headers.
BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _headers(url: str) -> dict[str, str]:
    """Browser-like headers with a Referer from the same site as the file."""
    host = urlparse(url).netloc
    referer = (
        "https://www.tcs.com/investor-relations/financial-statements"
        if "tcs.com" in host
        else f"https://{host}/"
    )
    return {**BASE_HEADERS, "Referer": referer}


def download(doc: DocumentEntry, force: bool = False, timeout: int = 60) -> bool:
    """Download one document. Returns True if the file is present afterwards."""
    if doc.exists and not force:
        logger.info("[ok] %s already present", doc.filename)
        return True
    doc.path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(doc.url, headers=_headers(doc.url), timeout=timeout)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise ValueError("response is not a PDF (the site may have blocked the request)")
        doc.path.write_bytes(resp.content)
        logger.info("[downloaded] %s (%.0f KB)", doc.filename, len(resp.content) / 1024)
        return True
    except Exception as exc:
        logger.warning(
            "[failed] could not download %s: %s\n    Download it manually from %s\n    and save it as %s",
            doc.filename, exc, doc.url, doc.path,
        )
        return False


def download_all(force: bool = False) -> dict[str, bool]:
    return {d.id: download(d, force=force) for d in load_manifest()}
