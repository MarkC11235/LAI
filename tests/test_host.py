"""Parsing what the machine reports, without running anything."""

from __future__ import annotations

import pytest

from lai.host import build_number


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        # b10729 on the desktop (the old format)
        ("version: 10729 (1e5ad35d5)\nbuilt with IntelLLVM 2026.0.0 for Linux x86_64\n", 10729),
        # b11259 on the laptop: the leading "0" of 0.5.0 is not the build
        ("version: 0.5.0-dev (build 11259, commit d280808f5)\nbuilt with GNU 15.2.0\n", 11259),
        ("libsvml.so: cannot open shared object file", None),
        ("", None),
    ],
)
def test_build_number(output, expected):
    assert build_number(output) == expected
