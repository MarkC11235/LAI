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

# Newer builds print "version: 0.5.0-dev (build 11259, commit ...)"; older ones
# printed "version: 10729 (1e5ad35d5)". The trailing \s stops the old form
# matching the 0 in 0.5.0.
_BUILD_NUMBER = re.compile(r"\(build (\d+)|version:\s*(\d+)\s")


def build_number(version_output: str) -> int | None:
    """The build number in `llama-server --version` output, or None."""
    match = _BUILD_NUMBER.search(version_output)
    return int(match.group(1) or match.group(2)) if match else None


class Host:
    def __init__(self, settings: Settings, env_wrapper: Path) -> None:
        self.settings = settings
        self.env_wrapper = env_wrapper        
        self._builds: dict[Path, int | None] = {}

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

    def llama_build(self, server: Path | None = None, wrapper: Path | None = None) -> int | None:
        """The build number (e.g. 10729) of a llama-server binary, or None if it
        can't be read. Defaults to Settings.llama_server and the shared wrapper.

        Tried through the generated wrapper first: a bare llama-server built with
        icpx (SYCL) dies without oneAPI on the library path. Probed once per
        binary, then cached.
        """
        server = server or self.settings.llama_server
        if server not in self._builds:
            self._builds[server] = self._probe_build(server, wrapper or self.env_wrapper)
        return self._builds[server]

    def _probe_build(self, server: Path, wrapper: Path) -> int | None:
        candidates = []
        if wrapper.exists() and os.access(wrapper, os.X_OK):
            candidates.append([str(wrapper), "--version"])
        candidates.append([str(server), "--version"])      
        for argv in candidates:
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            build = build_number(result.stdout + result.stderr)
            if build is not None:
                return build
        return None
