#!/usr/bin/env bash
# run_chat_bwrap.sh — run chat.py inside a bubblewrap sandbox.
#
# Usage
# -----
#   # OpenAI (default)
#   OPENAI_API_KEY=sk-... ./examples/run_chat_bwrap.sh
#
#   # DeepSeek
#   ./examples/run_chat_bwrap.sh --provider deepseek --api-key sk-...
#
#   # Local llama.cpp / Ollama (must be running on the host — uses --share-net)
#   ./examples/run_chat_bwrap.sh --provider local \
#       --base-url http://localhost:8080/v1 --model mistral
#
#   # Custom session DB
#   OPENAI_API_KEY=sk-... ./examples/run_chat_bwrap.sh --db myproject.db
#
# Environment variables forwarded into the sandbox
# -------------------------------------------------
#   OPENAI_API_KEY, DEEPSEEK_API_KEY, AGENT_API_KEY
#   AGENT_PROVIDER, AGENT_MODEL, AGENT_BASE_URL
#   AGENT_LOG_LEVEL, AGENT_LOG_FILE, AGENT_DB_URL
#
# The working directory ($(pwd)) is bind-mounted read-write at /work so the
# agent can read and modify files in your project.  The Python venv inside
# the project (/work/venv) is used directly — no separate install needed.
#
# Note: --new-session is intentionally omitted.  chat.py is a Textual TUI
# that needs a controlling terminal (for raw-mode input, /dev/tty, etc.).
# Calling setsid() via --new-session would detach it from the terminal and
# break the UI.
#
# Prerequisites
# -------------
#   bwrap   (bubblewrap) — https://github.com/containers/bubblewrap
#   Python venv at /work/venv with rojnik + textual installed

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script / project paths before entering the sandbox
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
AGENT_SCRIPT="$PROJECT_DIR/examples/chat.py"

# ---------------------------------------------------------------------------
# Sandbox home directory (persists across runs for SQLite state, logs, etc.)
# ---------------------------------------------------------------------------
SANDBOX_HOME="$HOME/.cache/chat_bwrap/home"
mkdir -p "$SANDBOX_HOME"

# ---------------------------------------------------------------------------
# DNS — use a reliable public resolver inside the sandbox
# ---------------------------------------------------------------------------
RESOLV_CONF="$HOME/.cache/chat_bwrap/resolv.conf"
mkdir -p "$(dirname "$RESOLV_CONF")"
printf 'nameserver 9.9.9.9\nnameserver 1.1.1.1\n' > "$RESOLV_CONF"

echo "Project  : $PROJECT_DIR"
echo "Venv     : /work/venv/bin/python (inside sandbox)"
echo "Work dir : $(pwd)"
echo "Args     : $*"
echo ""

# ---------------------------------------------------------------------------
# Forward provider / key env vars into the sandbox.
# Each is only forwarded when set in the outer environment.
# ---------------------------------------------------------------------------
ENV_ARGS=()
for var in \
    OPENAI_API_KEY \
    DEEPSEEK_API_KEY \
    AGENT_API_KEY \
    AGENT_PROVIDER \
    AGENT_MODEL \
    AGENT_BASE_URL \
    AGENT_LOG_LEVEL \
    AGENT_LOG_FILE \
    AGENT_DB_URL; do
    if [[ -n "${!var:-}" ]]; then
        ENV_ARGS+=(--setenv "$var" "${!var}")
    fi
done

# ---------------------------------------------------------------------------
# bubblewrap invocation
#
# --new-session is intentionally absent — see note at the top of this file.
# ---------------------------------------------------------------------------
exec bwrap \
    --ro-bind /usr /usr \
    --dir /tmp \
    --dir /var \
    --symlink ../tmp var/tmp \
    --proc /proc \
    --dev /dev \
    --symlink usr/lib   /lib \
    --symlink usr/lib64 /lib64 \
    --symlink usr/bin   /bin \
    --symlink usr/sbin  /sbin \
    \
    --unshare-all \
    --share-net \
    --die-with-parent \
    \
    --dir /run/user/$(id -u) \
    --setenv XDG_RUNTIME_DIR "/run/user/$(id -u)" \
    --setenv PS1 "bwrap-chat$ " \
    \
    --ro-bind "$RESOLV_CONF"        /etc/resolv.conf \
    --ro-bind /etc/hosts            /etc/hosts \
    --ro-bind /etc/nsswitch.conf    /etc/nsswitch.conf \
    --ro-bind /etc/passwd           /etc/passwd \
    --ro-bind /etc/group            /etc/group \
    --ro-bind /etc/localtime        /etc/localtime \
    --ro-bind /etc/ssl              /etc/ssl \
    --ro-bind-try /etc/ca-certificates /etc/ca-certificates \
    \
    --bind    "$SANDBOX_HOME"       "$HOME" \
    --bind    "$(pwd)"              /work \
    --chdir   /work \
    \
    --setenv HOME "$HOME" \
    --setenv PATH "/work/venv/bin:/usr/local/bin:/usr/bin:/bin" \
    --setenv PYTHONPATH "/work/src" \
    \
    "${ENV_ARGS[@]}" \
    \
    /work/venv/bin/python /work/examples/chat.py "$@"
