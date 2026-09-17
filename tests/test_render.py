"""Generation is pure: registry in, file contents out."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from conftest import make_model, make_registry

from lai import render

GEN = Path("/repo/gen")


def args_of(model, **settings):
    registry = make_registry(model, **settings)
    groups = render.llama_server_args(model, registry.settings, "/models/m.gguf", None)
    return [token for group in groups for token in group]


def test_server_args_map_fields_to_flags():
    model = make_model(device="SYCL1", cpu_moe_layers=12, load_mode="none", kv_type="q8_0",
                       parallel=2, ubatch=4096, batch=8192, flash_attn="off")
    tokens = args_of(model)
    joined = " ".join(tokens)
    for expected in ["--port ${PORT}", "-a m", "-m /models/m.gguf", "--device SYCL1", "--n-cpu-moe 12",
                     "--load-mode none", "-c 65536", "-ub 4096 -b 8192", "--parallel 2", "-fa off",
                     "-ctk q8_0 -ctv q8_0", "--jinja"]:
        assert expected in joined


def test_optional_flags_are_omitted_at_defaults():
    joined = " ".join(args_of(make_model()))
    for absent in ["--device", "--n-cpu-moe", "--load-mode", "-ctk", "--mmproj"]:
        assert absent not in joined


def test_sampler_keys_normalise_to_dashes():
    joined = " ".join(args_of(make_model(sampler={"top_p": 0.95, "min-p": 0.0})))
    assert "--top-p 0.95" in joined and "--min-p 0.0" in joined


def test_group_flags_keeps_values_with_their_flag():
    assert render.group_flags(["-ot", "x=CPU", "--jinja", "--rope-scale", "-1.5"]) == [
        ["-ot", "x=CPU"], ["--jinja"], ["--rope-scale", "-1.5"],
    ]


@pytest.mark.parametrize("tokens", [
    ["-ot", r"per_layer_token_embd\.weight=CPU"],
    ["--chat-template-kwargs", '{"reasoning_effort": "low"}'],
    ["--port", "${PORT}"],
    ["--name", "it's"],
])
def test_shell_join_round_trips_through_a_shell_split(tokens):
    line = render.shell_join(tokens)
    assert shlex.split(line) == tokens
    if "${PORT}" in tokens:
        assert "'${PORT}'" not in line  # llama-swap substitutes the bare macro


def test_backend_command_marks_missing_weights():
    registry = make_registry(make_model())

    class NothingHost:
        def find(self, _):
            return None

    command = render.backend_command(registry.models[0], registry.settings, NothingHost(), GEN)
    assert "MISSING:/models/m/m.gguf" in command
    assert command.splitlines()[0] == "/repo/gen/llama-env.sh"


def test_vllm_backend_command_is_the_launcher(host_for):
    model = make_model(id="v", engine="vllm", launcher="true")
    registry = make_registry(model)
    command = render.backend_command(model, registry.settings, host_for(registry), GEN)
    assert command == "/repo/gen/vllm-v.sh ${PORT}"


def test_swap_config_parses_and_carries_capabilities(host_for):
    registry = make_registry(
        make_model(id="a", vision=True, mmproj="a/p.gguf", parallel=2, ttl=60,
                   request_params={"chat_template_kwargs": {"reasoning_effort": "low"}}),
        make_model(id="b", enabled=False),
    )
    document = yaml.safe_load(render.llama_swap_config(registry, host_for(registry), GEN))
    assert list(document["models"]) == ["a"]
    entry = document["models"]["a"]
    assert entry["capabilities"] == {
        "tools": True, "context": 32768, "in": ["text", "image"], "out": ["text"],
    }
    assert entry["ttl"] == 60
    assert entry["setParams"] == {"chat_template_kwargs": {"reasoning_effort": "low"}}
    assert "--mmproj /models/a/p.gguf" in entry["cmd"]


def test_coresidency_matrix_pairs_one_model_per_card():
    registry = make_registry(
        make_model(id="zero-a", device="SYCL0", vram_gb=21.7),
        make_model(id="both"),
        make_model(id="one-a", device="SYCL1"),
        make_model(id="one-b", device="SYCL1"),
    )
    router = render.coresidency_router(registry)
    matrix = router["settings"]["matrix"]
    assert router["use"] == "matrix"
    assert matrix["vars"] == {"a0": "zero-a", "b0": "one-a", "b1": "one-b"}
    assert matrix["evict_costs"]["a0"] == 21
    assert matrix["sets"] == {"coresident": "(a0) & (b0 | b1)"}
    assert "both" not in matrix["vars"].values()  # unpinned: runs alone


def test_single_populated_card_falls_back_to_group_router():
    registry = make_registry(make_model(id="x", device="SYCL1"), make_model(id="y"))
    assert render.coresidency_router(registry) == {"use": "group"}


def test_opencode_limits_and_capabilities():
    registry = make_registry(
        make_model(id="a", parallel=4, max_output=1234, vision=True, mmproj="p", reasoning_field="reasoning"),
        make_model(id="b"),
    )
    config = render.opencode_config(registry)
    provider = config["provider"][render.OPENCODE_PROVIDER]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:9090/v1"
    a, b = provider["models"]["a"], provider["models"]["b"]
    assert a["limit"] == {"context": 16384, "output": 1234}
    assert a["modalities"] == {"input": ["text", "image"], "output": ["text"]} and a["attachment"] is True
    assert a["interleaved"] == {"field": "reasoning"}
    assert "modalities" not in b and "interleaved" not in b
    assert config["model"] == "llamacpp/a"


def test_opencode_extra_deep_merges():
    registry = make_registry(opencode_extra={"mcp": {"s": {"type": "local"}},
                                             "provider": {"llamacpp": {"name": "renamed"}}})
    config = render.opencode_config(registry)
    assert config["mcp"] == {"s": {"type": "local"}}
    assert config["provider"]["llamacpp"]["name"] == "renamed"
    assert "models" in config["provider"]["llamacpp"]  # merged, not replaced


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    render.deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


def test_templates_fill_completely():
    registry = make_registry(make_model(engine="vllm", launcher="echo \"$MODEL_ID\"\n"))
    settings, model = registry.settings, registry.models[0]
    wrapper = render.env_wrapper(settings)
    launcher = render.vllm_launcher_script(model, settings, repo_root=Path("/repo"))
    unit = render.systemd_unit(settings, "/bin/llama-swap", GEN / "llama-swap.yaml")
    for text in (wrapper, launcher, unit):
        assert not re.search(r"@[A-Z_]+@", text)
    assert "LLAMA_SERVER=/bin/llama-server" in wrapper
    assert "CONTEXT=65536" in launcher and "REPO_ROOT=/repo" in launcher
    assert launcher.endswith('echo "$MODEL_ID"\n')
    assert "--listen 127.0.0.1:9090" in unit


def test_fill_template_rejects_unfilled_placeholders():
    with pytest.raises(ValueError, match="@LLAMA_SERVER@"):
        render.fill_template("llama-env.sh", ONEAPI_SETVARS="/x")


def test_artifacts_cover_every_generated_file(host_for):
    registry = make_registry(make_model(id="a"), make_model(id="v", engine="vllm", launcher="true"))
    files = render.artifacts(registry, host_for(registry), GEN)
    assert [f.path.name for f in files] == ["llama-swap.yaml", "llama-env.sh", "vllm-v.sh", "opencode.json"]
    assert [f.executable for f in files] == [False, True, True, False]
    json.loads(files[-1].content)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_generated_scripts_pass_shellcheck(tmp_path):
    """Covers the templates and the real launcher bodies in models.py together."""
    from conftest import FakeHost

    from lai import paths, registry

    repo = registry.load(paths.REGISTRY_FILE)
    scripts = [a for a in render.artifacts(repo, FakeHost(repo.settings), tmp_path) if a.executable]
    assert scripts
    for script in scripts:
        script.path.write_text(script.content)
    checked = [str(s.path) for s in scripts] + [str(paths.REPO_ROOT / "bin/ldr")]
    result = subprocess.run(["shellcheck", "-x", "-S", "warning", *checked], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
