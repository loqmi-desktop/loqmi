"""core/ocr_support.py: locate Tesseract and report whether OCR can work."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pytesseract

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "Reading text from screenshots and scanned PDFs needs Tesseract, which isn't installed. "
    "Install it with `brew install tesseract`, then try again."
)

# GUI apps launched from Finder don't inherit your shell PATH, so look in the usual places.
_CANDIDATES = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/local/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

_available: bool | None = None


def configure() -> bool:
    global _available
    path = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not path:
        path = next((c for c in _CANDIDATES if Path(c).exists()), None)
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
    try:
        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract %s found at %s", version, path or "PATH")
        _available = True
    except Exception:
        logger.warning("Tesseract not found. %s", INSTALL_HINT)
        _available = False
    return _available


def available() -> bool:
    """Cheap after the first success; re-checks if it was missing, so installing it needs no restart."""
    return True if _available else configure()