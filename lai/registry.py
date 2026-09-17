"""Load `models.py` and answer questions about it.

The registry is loaded from a file path rather than imported by name, so tests
can build a `Registry` in memory and nothing depends on `sys.path` tricks.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from lai import LaiError
from lai.paths import REGISTRY_FILE
from lai.schema import Model, Settings


@dataclass(frozen=True)
class Registry:
    settings: Settings
    models: tuple[Model, ...]

    def active(self) -> list[Model]:
        """Enabled models, in registry order. Only these are generated."""
        return [m for m in self.models if m.enabled]

    def get(self, model_id: str) -> Model:
        for model in self.models:
            if model.id == model_id:
                return model
        known = ", ".join(m.id for m in self.active())
        raise LaiError(f"unknown model {model_id!r}. Known: {known}")

    def pinned_to(self, card: str) -> list[Model]:
        """Enabled models pinned to one card: the candidates for co-residency."""
        return [m for m in self.active() if m.device == card]


def load(path: Path = REGISTRY_FILE) -> Registry:
    """Import a registry file and return its SETTINGS and MODELS."""
    spec = importlib.util.spec_from_file_location("lai_registry", path)
    if spec is None or spec.loader is None:
        raise LaiError(f"cannot load registry from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    settings = getattr(module, "SETTINGS", None)
    models = getattr(module, "MODELS", None)
    if not isinstance(settings, Settings):
        raise LaiError(f"{path} must define SETTINGS = Settings(...)")
    if not isinstance(models, list) or not all(isinstance(m, Model) for m in models):
        raise LaiError(f"{path} must define MODELS as a list of Model(...)")
    return Registry(settings=settings, models=tuple(models))
