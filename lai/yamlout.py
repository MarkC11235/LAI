"""A deliberately small YAML emitter for the one YAML document lai writes.

The runtime stays stdlib-only, and the llama-swap config needs only a small
subset of YAML: nested mappings, block sequences, scalars, and literal blocks
for multi-line commands. Strings are always double-quoted, so no value can be
misread as a number, boolean, or null. tests/test_yamlout.py round-trips the
output through PyYAML to prove it parses back to the same data.
"""

from __future__ import annotations

import re

_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]*")
# Plain keys that YAML 1.1 parsers would still read as booleans or null.
_AMBIGUOUS_KEYS = {"y", "n", "yes", "no", "on", "off", "true", "false", "null"}


def dump(document: dict, header: str = "") -> str:
    """Render a mapping as YAML, preceded by `header` as comment lines."""
    lines = [f"# {line}".rstrip() for line in header.splitlines()]
    if lines:
        lines.append("")
    lines.extend(_mapping(document, indent=0))
    return "\n".join(lines) + "\n"


def _mapping(mapping: dict, indent: int) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    for raw_key, value in mapping.items():
        key = _key(raw_key)
        if isinstance(value, dict) and value:
            lines.append(f"{pad}{key}:")
            lines.extend(_mapping(value, indent + 2))
        elif isinstance(value, list) and value:
            lines.append(f"{pad}{key}:")
            lines.extend(_sequence(value, indent + 2))
        elif _is_literal_block(value):
            lines.append(f"{pad}{key}: |")
            lines.extend(f"{pad}  {line}" if line else "" for line in value[:-1].split("\n"))
        else:
            lines.append(f"{pad}{key}: {_scalar(value)}")
    return lines


def _sequence(items: list, indent: int) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict) and item:
            nested = _mapping(item, indent + 2)
            nested[0] = f"{pad}- {nested[0].lstrip()}"
            lines.extend(nested)
        elif isinstance(item, list) and item:
            lines.append(f"{pad}-")
            lines.extend(_sequence(item, indent + 2))
        else:
            lines.append(f"{pad}- {_scalar(item)}")
    return lines


def _is_literal_block(value: object) -> bool:
    """Multi-line strings ending in exactly one newline, with no line that would
    confuse block indentation, are emitted as `|` blocks for readability."""
    if not isinstance(value, str) or not value.endswith("\n") or value.endswith("\n\n"):
        return False
    body = value[:-1]
    return "\n" in body and not any(line[:1].isspace() for line in body.split("\n"))


def _key(key: object) -> str:
    text = str(key)
    if _PLAIN_KEY.fullmatch(text) and text.lower() not in _AMBIGUOUS_KEYS:
        return text
    return _quote(text)


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        # YAML 1.1 parsers need a dot and a signed exponent: 1e-05 -> 1.0e-05
        return text.replace("e", ".0e") if "e" in text and "." not in text else text
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, list):
        return "[]"
    return _quote(str(value))


def _quote(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'
