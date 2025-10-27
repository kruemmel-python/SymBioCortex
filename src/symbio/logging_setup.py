"""Zentrale Logging-Konfiguration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "symbio.log"


def configure_logging(level: int = logging.INFO) -> None:
    """Konfiguriere Logging-Ausgabe für Konsole und Datei."""

    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


__all__ = ["configure_logging", "LOG_FILE"]
