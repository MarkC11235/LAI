"""The real models.py: it loads, passes every check, and keeps its interfaces."""

from __future__ import annotations

import pytest
from conftest import FakeHost, make_model, make_registry

from lai import LaiError, checks, paths, registry

# Clients (opencode sessions, ldr.env, scripts) refer to these keys. Changing
# this set should be a deliberate edit to this test, never a side effect.
ROUTING_KEYS = {
    "qwen38-vllm",
    "qwen38-flash-next-medium",
    "qwen38-flash-next-low",
    "qwen38-27b-c0-mtp-xhigh",
    "qwen38-27b-c1-mtp-xhigh",
    "qwen38-27b-c0-mtp",
    "qwen38-27b-c1-mtp",
    "qwen36-moe-c1-mtp",
}


@pytest.fixture(scope="module")
def repo_registry():
    return registry.load(paths.REGISTRY_FILE)


def test_routing_keys_are_stable(repo_registry):
    assert {m.id for m in repo_registry.active()} == ROUTING_KEYS


def test_repo_registry_passes_every_check(repo_registry):
    found = checks.run(repo_registry, FakeHost(repo_registry.settings, build=10729))
    assert [str(f) for f in found if f.severity is checks.Severity.ERROR] == []


def test_default_models_are_single_card(repo_registry):
    """Defaults load on first opencode start; a both-cards default would evict everything."""
    for model_id in (repo_registry.settings.default_model, repo_registry.settings.small_model):
        assert repo_registry.get(model_id).device is not None


def test_same_weights_variants_differ_only_where_intended(repo_registry):
    c0 = repo_registry.get("qwen38-27b-c0-mtp")
    c1 = repo_registry.get("qwen38-27b-c1-mtp")
    xhigh = repo_registry.get("qwen38-27b-c1-mtp-xhigh")
    assert (c0.device, c1.device) == ("SYCL0", "SYCL1")
    assert xhigh.request_params["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    ignore = {"id", "name", "device", "request_params"}
    assert {k: v for k, v in vars(c1).items() if k not in ignore} == \
           {k: v for k, v in vars(xhigh).items() if k not in ignore}


def test_mcp_search_is_generated(repo_registry):
    extra = repo_registry.settings.opencode_extra
    assert "searxng" in extra["mcp"]
    assert extra["tools"]["webfetch"] is False


def test_get_unknown_model_lists_known_ids():
    reg = make_registry(make_model(id="known"))
    with pytest.raises(LaiError, match="known"):
        reg.get("unknown")


def test_pinned_to_ignores_disabled_models():
    reg = make_registry(make_model(id="a", device="SYCL1"), make_model(id="b", device="SYCL1", enabled=False))
    assert [m.id for m in reg.pinned_to("SYCL1")] == ["a"]


def test_load_rejects_a_file_without_the_contract(tmp_path):
    bad = tmp_path / "models.py"
    bad.write_text("MODELS = []\n")
    with pytest.raises(LaiError, match="SETTINGS"):
        registry.load(bad)
