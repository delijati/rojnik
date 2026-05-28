"""
Loguru logging configuration.

Two sinks:
  - Console: human-readable, coloured, INFO and above.
  - File:    newline-delimited JSON, DEBUG and above. Rotates at 50 MB.

Call setup_logging() once at process start (done automatically on first import
of rojnik).  All other modules just do:

    from loguru import logger
    logger.info("...", key=value)
"""


import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
    "{extra}"
)


def _format_extra(record: dict) -> str:
    """Append structured extra fields to the console line when present."""
    extra = record["extra"]
    if not extra:
        return ""
    pairs = " ".join(f"{k}={v!r}" for k, v in extra.items())
    return f"  [{pairs}]"


def setup_logging(log_level: str = "INFO", log_file: str = "agent.log") -> None:
    """Configure loguru sinks.  Safe to call multiple times (idempotent)."""
    logger.remove()  # remove default stderr sink

    # --- Console sink ---
    logger.add(
        sys.stderr,
        level=log_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # --- JSON file sink ---
    logger.add(
        log_file,
        level="DEBUG",
        serialize=True,          # writes newline-delimited JSON
        rotation="50 MB",
        retention="14 days",
        compression="gz",
        backtrace=True,
        diagnose=False,          # don't embed local vars in prod log file
    )

    logger.debug("Logging initialised", log_level=log_level, log_file=log_file)
