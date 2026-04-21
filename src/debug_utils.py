"""
Утилиты для debug-логов в консоль и файл.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "design_input": "\033[36m",      # cyan
    "design_output": "\033[96m",     # light cyan
    "generate_input": "\033[34m",    # blue
    "generate_output": "\033[32m",   # green
    "validate_input": "\033[33m",    # yellow
    "validate_verdict": "\033[35m",  # magenta
    "format_info": "\033[90m",       # gray
    "rag_query": "\033[94m",         # light blue
    "rag_result": "\033[92m",        # light green
    "agent_start": "\033[36m",       # cyan
    "agent_result": "\033[32m",      # green
    "agent_error": "\033[31m",       # red
    "warning": "\033[31m",           # red
    "info": "\033[37m",              # white
}


def _single_line(text: str) -> str:
    return (text or "").strip().replace("\n", " ")


def _trim(text: str, limit: int | None) -> str:
    value = _single_line(text)
    if limit is None or len(value) <= limit:
        return value
    return value[:limit] + "..."


def debug_log(
    *,
    enabled: bool,
    stage: str,
    message: str,
    question_index: int | None = None,
    console_limit: int | None = 240,
    file_limit: int | None = 240,
    file_path: str | None = None,
    colorize: bool = True,
):
    """
    Вывести debug-лог в консоль и (опционально) в файл.
    """
    if not enabled:
        return

    qprefix = f"[Q{question_index}] " if question_index is not None else ""
    stage_label = stage.upper()

    console_text = _trim(message, console_limit)
    file_text = _trim(message, file_limit)
    base_console = f"[DEBUG] {qprefix}[{stage_label}] {console_text}"
    base_file = f"{datetime.now().isoformat()} [DEBUG] {qprefix}[{stage_label}] {file_text}"

    if colorize:
        color = ANSI_COLORS.get(stage, "")
        if color:
            print(f"{color}{base_console}{ANSI_RESET}")
        else:
            print(base_console)
    else:
        print(base_console)

    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(base_file + "\n")

