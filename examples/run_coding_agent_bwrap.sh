#!/usr/bin/env bash
# run_coding_agent_bwrap.sh — run coding_agent.py inside a bubblewrap sandbox.
#
# Usage
# -----
#   # OpenAI (default)
#   OPENAI_API_KEY=sk-... ./examples/run_coding_agent_bwrap.sh
#
#   # DeepSeek
#   ./examples/run_coding_agent_bwrap.sh --provider deepseek --api-key sk-...
#
#   # Local Ollama (must be running on the host — uses --share-net)
#   ./examples/run_coding_agent_bwrap.sh --provider local \
#       --base-url http://localhost:11434/v1 --model llama3
#
#   # With a custom task
#   OPENAI_API_KEY=sk-... ./examples/run_coding_agent_bwrap.sh \
#       "List the files in /tmp and tell me how many there are"
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
# Prerequisites
# -------------
#   bwrap   (bubblewrap) — https://github.com/containers/bubblewrap
#   Python venv at /work/venv with agent_harness installed (pip install -e .)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script / project paths before entering the sandbox
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
AGENT_SCRIPT="$PROJECT_DIR/examples/coding_agent.py"

# ---------------------------------------------------------------------------
# Sandbox home directory (persists across runs for SQLite state, logs, etc.)
# ---------------------------------------------------------------------------
SANDBOX_HOME="$HOME/.cache/coding_agent_bwrap/home"
mkdir -p "$SANDBOX_HOME"

# ---------------------------------------------------------------------------
# DNS — use a reliable public resolver inside the sandbox
# ---------------------------------------------------------------------------
RESOLV_CONF="$HOME/.cache/coding_agent_bwrap/resolv.conf"
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
    --new-session \
    \
    --dir /run/user/$(id -u) \
    --setenv XDG_RUNTIME_DIR "/run/user/$(id -u)" \
    --setenv PS1 "bwrap-agent$ " \
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
    /work/venv/bin/python /work/examples/coding_agent.py "$@"
