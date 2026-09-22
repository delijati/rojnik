"""
Built-in shell execution tool.

Security note
-------------
shell_exec runs arbitrary commands in a subprocess.
Always audit the system prompt of any agent given this tool.
Consider restricting the working directory and using a timeout.
"""


import asyncio
import os

from loguru import logger

from rojnik.tools.base import tool


@tool(description=(
    "Execute a shell command in a subprocess and return its stdout + stderr. "
    "The command is run with /bin/sh -c in the given working directory "
    "(defaults to current directory). "
    "A timeout (in seconds, default 30) kills the process if exceeded. "
    "Returns combined stdout and stderr as a single string."
))
async def shell_exec(
    command: str,
    workdir: str = ".",
    timeout: int = 30,
) -> str:
    abs_workdir = os.path.abspath(workdir)
    if not os.path.isdir(abs_workdir):
        return f"ERROR: Working directory not found: {abs_workdir}"

    logger.debug("tool.shell_exec", command=command, workdir=abs_workdir, timeout=timeout)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=abs_workdir,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"ERROR: Command timed out after {timeout}s: {command}"

        output = stdout.decode(errors="replace")
        exit_code = proc.returncode

        if exit_code != 0:
            return f"[exit code {exit_code}]\n{output}"
        return output or "(no output)"

    except Exception as exc:
        return f"ERROR: Failed to run command: {exc}"
