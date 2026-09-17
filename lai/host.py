"""Everything lai reads from the machine, behind one seam.

Checks and rendering take a `Host` instead of touching the filesystem directly,
so tests substitute a fake and the rules stay pure functions of the registry.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path

from lai.schema import Settings

_BUILD_NUMBER = re.compile(r"version:\s*(\d+)")


class Host:
    def __init__(self, settings: Settings, env_wrapper: Path) -> None:
        self.settings = settings
        self.env_wrapper = env_wrapper
        self._build: int | None = None
        self._build_probed = False

    def find(self, relative_pattern: str) -> str | None:
        """First match (sorted) of a glob under the model directory, or None."""
        matches = sorted(glob.glob(str(self.settings.model_dir / relative_pattern)))
        return matches[0] if matches else None

    def exists(self, path: Path) -> bool:
        return path.exists()

    def read_text(self, path: Path) -> str | None:
        try:
            return path.read_text()
        except OSError:
            return None

    def llama_build(self) -> int | None:
        """The llama.cpp build number (e.g. 10729), or None if it can't be read.

        Tried through the generated wrapper first: a bare llama-server built with
        icpx dies without oneAPI on the library path. Probed once, then cached.
        """
        if not self._build_probed:
            self._build = self._probe_build()
            self._build_probed = True
        return self._build

    def _probe_build(self) -> int | None:
        candidates = []
        if self.env_wrapper.exists() and os.access(self.env_wrapper, os.X_OK):
            candidates.append([str(self.env_wrapper), "--version"])
        candidates.append([str(self.settings.llama_server), "--version"])
        for argv in candidates:
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            match = _BUILD_NUMBER.search(result.stdout + result.stderr)
            if match:
                return int(match.group(1))
        return None
