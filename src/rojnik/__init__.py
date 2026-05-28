"""
rojnik — a minimal, from-scratch agentic harness.

Importing this package bootstraps logging automatically.
"""

from rojnik.logging_setup import setup_logging
from rojnik.config import settings

setup_logging(log_level=settings.log_level, log_file=settings.log_file)
