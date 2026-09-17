"""Shared fixtures: an in-memory registry and a fake host, so no test needs
GPUs, model files, oneAPI, or a running proxy."""

from __future__ import annotations

from pathlib import Path

import pytest

from lai.registry import Registry
from lai.schema import Model, Settings


class FakeHost:
    """Stands in for lai.host.Host. Every path exists unless listed in `missing`."""

    def __init__(self, settings: Settings, *, build: int | None = 10729, missing: set[str] = frozenset(),
                 files: dict[Path, str] | None = None) -> None:
        self.settings = settings
        self.build = build
        self.missing = set(missing)
        self.files = files or {}

    def find(self, relative_pattern: str) -> str | None:
        if relative_pattern in self.missing:
            return None
        return str(self.settings.model_dir / relative_pattern)

    def exists(self, path: Path) -> bool:
        return str(path) not in self.missing

    def read_text(self, path: Path) -> str | None:
        return self.files.get(path)

    def llama_build(self) -> int | None:
        return self.build


def make_model(**overrides) -> Model:
    fields = {"id": "m", "name": "M", "weights": "m/m.gguf", "context": 65536, "vram_gb": 20.0}
    return Model(**{**fields, **overrides})


def make_registry(*models: Model, **settings_overrides) -> Registry:
    models = models or (make_model(),)
    settings = {
        "model_dir": Path("/models"),
        "llama_server": Path("/bin/llama-server"),
        "oneapi_setvars": Path("/opt/intel/oneapi/setvars.sh"),
        "default_model": models[0].id,
        "small_model": models[0].id,
        "activity_db": Path("/state/activity.db"),
        "opencode_config": Path("/config/opencode.json"),
        **settings_overrides,
    }
    return Registry(settings=Settings(**settings), models=tuple(models))


@pytest.fixture
def host_for():
    def build(registry: Registry, **kwargs) -> FakeHost:
        return FakeHost(registry.settings, **kwargs)

    return build
