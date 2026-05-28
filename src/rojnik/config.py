"""
Central configuration for the rojnik.
All settings can be overridden via environment variables.

Provider support
----------------
Set AGENT_PROVIDER to one of:
  openai   — default, requires OPENAI_API_KEY
  deepseek — OpenAI-compatible, requires DEEPSEEK_API_KEY (or AGENT_API_KEY)
  local    — OpenAI-compatible local server (Ollama, LM Studio, vLLM, …)
             requires AGENT_BASE_URL; no API key needed unless the server
             enforces one (set AGENT_API_KEY in that case)

Quick start examples
--------------------
  # OpenAI (default)
  OPENAI_API_KEY=sk-... python examples/coding_agent.py

  # DeepSeek
  AGENT_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... python examples/coding_agent.py

  # Local Ollama
  AGENT_PROVIDER=local AGENT_BASE_URL=http://localhost:11434/v1 \
    AGENT_MODEL=llama3 python examples/coding_agent.py
"""


import os
from dataclasses import dataclass, field

# Recognised provider identifiers
PROVIDERS = ("openai", "deepseek", "local")

# Default model per provider (can always be overridden via AGENT_MODEL)
_DEFAULT_MODELS: dict[str, str] = {
    "openai":    "gpt-4o",
    "deepseek":  "deepseek-chat",
    "local":     "llama3",
}

# DeepSeek's public API endpoint
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class Settings:
    # ------------------------------------------------------------------ #
    # Provider selection                                                   #
    # ------------------------------------------------------------------ #
    provider: str = field(
        default_factory=lambda: os.environ.get("AGENT_PROVIDER", "openai").lower()
    )

    # ------------------------------------------------------------------ #
    # API keys — each provider checks its own env var first,              #
    # then falls back to the generic AGENT_API_KEY.                       #
    # ------------------------------------------------------------------ #
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    deepseek_api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "")
    )
    # Generic fallback / local server token
    api_key: str = field(
        default_factory=lambda: os.environ.get("AGENT_API_KEY", "")
    )

    # ------------------------------------------------------------------ #
    # Model & endpoint                                                     #
    # ------------------------------------------------------------------ #
    # AGENT_MODEL overrides the per-provider default set in __post_init__
    model: str = field(
        default_factory=lambda: os.environ.get("AGENT_MODEL", "")
    )
    # AGENT_BASE_URL overrides the per-provider default (required for local)
    base_url: str = field(
        default_factory=lambda: os.environ.get("AGENT_BASE_URL", "")
    )

    max_tokens: int = 4096

    # ------------------------------------------------------------------ #
    # Agent loop                                                           #
    # ------------------------------------------------------------------ #
    max_iterations: int = 20
    # Token budget for the context window sent to the LLM on each iteration.
    # Messages are trimmed oldest-first when this is exceeded.
    context_token_budget: int = 100_000

    # Retry behaviour for LLM calls
    max_retries: int = 3
    retry_wait_seconds: float = 2.0

    # ------------------------------------------------------------------ #
    # Storage                                                              #
    # ------------------------------------------------------------------ #
    db_url: str = field(
        default_factory=lambda: os.environ.get(
            "AGENT_DB_URL", "sqlite+aiosqlite:///./agent.db"
        )
    )

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    log_level: str = field(
        default_factory=lambda: os.environ.get("AGENT_LOG_LEVEL", "INFO")
    )
    log_file: str = field(
        default_factory=lambda: os.environ.get("AGENT_LOG_FILE", "agent.log")
    )

    # ------------------------------------------------------------------ #
    # Post-init: validate & fill in per-provider defaults                 #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown AGENT_PROVIDER={self.provider!r}. "
                f"Choose one of: {', '.join(PROVIDERS)}"
            )

        # Fill in model default when caller didn't specify one
        if not self.model:
            self.model = _DEFAULT_MODELS[self.provider]

        # Fill in base_url default for DeepSeek
        if not self.base_url and self.provider == "deepseek":
            self.base_url = DEEPSEEK_BASE_URL

        # Validate API key per provider
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running the rojnik, "
                "or set AGENT_PROVIDER=deepseek / AGENT_PROVIDER=local."
            )
        if self.provider == "deepseek" and not (self.deepseek_api_key or self.api_key):
            raise ValueError(
                "DEEPSEEK_API_KEY (or AGENT_API_KEY) environment variable is not set. "
                "Export it before running with AGENT_PROVIDER=deepseek."
            )
        # local: no key required — the server may not enforce auth

    # ------------------------------------------------------------------ #
    # Convenience helpers used by LLMClient                               #
    # ------------------------------------------------------------------ #
    def get_api_key(self) -> str:
        """Return the active API key for the configured provider."""
        if self.provider == "openai":
            return self.openai_api_key
        if self.provider == "deepseek":
            return self.deepseek_api_key or self.api_key
        # local — return whatever the caller set, defaulting to a dummy value
        # because AsyncOpenAI requires a non-empty string
        return self.api_key or "local"

    def get_base_url(self) -> str | None:
        """Return the base URL for the configured provider, or None for OpenAI default."""
        return self.base_url or None


# Module-level singleton — import this everywhere.
settings = Settings()
