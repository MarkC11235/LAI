"""Starting llama-swap, with a stand-in binary instead of the real one."""

from __future__ import annotations

from pathlib import Path

import pytest

from lai import LaiError
from lai.proxy import ProxyService
from lai.schema import Settings


class DownProxy:
    """A proxy that never answers /health."""

    base_url = "http://127.0.0.1:9"

    def healthy(self) -> bool:
        return False


def service_with(tmp_path: Path, binary_body: str) -> ProxyService:
    binary = tmp_path / "llama-swap"
    binary.write_text(f"#!/bin/sh\n{binary_body}\n")
    binary.chmod(0o755)
    config = tmp_path / "gen/llama-swap.yaml"
    config.parent.mkdir()
    config.write_text("models: {}\n")
    settings = Settings(activity_db=tmp_path / "state/llama-swap/activity.db")
    service = ProxyService(settings, DownProxy(), config=config, pidfile=tmp_path / "gen/pid",
                           logfile=tmp_path / "gen/llama-swap.log", unit=tmp_path / "no-such.service")
    service.binary = lambda: str(binary)  # instance attribute shadows the staticmethod
    return service


def test_start_creates_the_activity_db_directory(tmp_path):
    service = service_with(tmp_path, "exit 1")
    with pytest.raises(LaiError):
        service.start()
    assert (tmp_path / "state/llama-swap").is_dir()


def test_start_reports_an_early_exit_with_its_log(tmp_path):
    service = service_with(tmp_path, 'echo "ERROR failed to load config: bad key"; exit 3')
    (tmp_path / "gen/llama-swap.log").write_text("an older run that should not be shown\n")
    with pytest.raises(LaiError, match="status 3") as raised:
        service.start()
    assert "bad key" in str(raised.value)
    assert "older run" not in str(raised.value)
    assert not (tmp_path / "gen/pid").exists()
