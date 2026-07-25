"""Centralized logging configuration."""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger with a consistent format.

    Idempotent: safe to call repeatedly for the same module without
    creating duplicate handlers (which would double-log messages).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
