"""
Loguru logging configuration.

Sinks:
  - Console: human-readable, coloured, ``log_level`` and above (always).
  - File:    newline-delimited JSON, DEBUG and above, rotates at 50 MB.  Opt-in:
             only configured when ``log_file`` (``AGENT_LOG_FILE``) is non-empty,
             and opened lazily on the first record, never at import time.

Call setup_logging() once at process start (done automatically on first import
of rojnik).  All other modules just do:

    from loguru import logger
    logger.info("...", key=value)
"""

import sys

from loguru import logger

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


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru sinks.  Safe to call multiple times (idempotent).

    Without ``log_file`` only the console sink is configured, so importing rojnik
    never creates files (e.g. in a read-only working directory).
    """
    logger.remove()  # remove default stderr sink

    # --- Console sink ---
    logger.add(
        sys.stderr,
        level=log_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    logger.debug("Logging initialised", log_level=log_level, log_file=log_file or None)

    if not log_file:
        return

    # --- JSON file sink (opt-in, opened on the first record) ---
    logger.add(
        log_file,
        level="DEBUG",
        serialize=True,  # writes newline-delimited JSON
        rotation="50 MB",
        retention="14 days",
        compression="gz",
        delay=True,
        backtrace=True,
        diagnose=False,  # don't embed local vars in prod log file
    )
