"""
rojnik — a minimal, from-scratch agentic harness.

Importing this package bootstraps logging automatically.
"""

from rojnik.agent import Agent, AgentRegistry, MaxIterationsError, RunState, run_loop
from rojnik.config import settings
from rojnik.logging_setup import setup_logging
from rojnik.skills import Skill, SkillError, load_skills
from rojnik.tools import ToolNotFoundError, ToolRegistry, tool

setup_logging(log_level=settings.log_level, log_file=settings.log_file)

__all__ = [
    "Agent",
    "AgentRegistry",
    "MaxIterationsError",
    "RunState",
    "Skill",
    "SkillError",
    "ToolNotFoundError",
    "ToolRegistry",
    "run_loop",
    "load_skills",
    "settings",
    "setup_logging",
    "tool",
]
