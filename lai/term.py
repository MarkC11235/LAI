"""Terminal output: colour (only when stdout is a TTY) and message helpers."""

import sys

_TTY = sys.stdout.isatty()


def _sgr(code: str) -> str:
    return f"\033[{code}m" if _TTY else ""


BOLD, DIM, RED, YELLOW, GREEN, CYAN, RESET = (
    _sgr("1"), _sgr("2"), _sgr("31"), _sgr("33"), _sgr("32"), _sgr("36"), _sgr("0")
)


def info(message: str) -> None:
    print(f"{CYAN}::{RESET} {message}")


def warn(message: str) -> None:
    print(f"{YELLOW}warn:{RESET} {message}", file=sys.stderr)


def error(message: str) -> None:
    print(f"{RED}error:{RESET} {message}", file=sys.stderr)
