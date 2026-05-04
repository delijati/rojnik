"""
agent_harness — a minimal, from-scratch agentic harness.

Importing this package bootstraps logging automatically.
"""

from agent_harness.logging_setup import setup_logging
from agent_harness.config import settings

setup_logging(log_level=settings.log_level, log_file=settings.log_file)
