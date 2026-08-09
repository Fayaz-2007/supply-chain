"""
Centralized logging setup.

Every module should call ``get_logger(__name__)`` instead of configuring its
own handlers, so log output stays consistent and is written both to the
console and to a shared log file under ``outputs/reports``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from utils.config import REPORTS_OUTPUT_DIR

_LOG_FILE: Path = REPORTS_OUTPUT_DIR / "app.log"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    """Attach console + file handlers to the root logger exactly once."""
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting/handlers.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _configure_root()
    return logging.getLogger(name)
