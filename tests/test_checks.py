"""Each rule fires on the configuration it exists for, and stays quiet otherwise."""

from __future__ import annotations

import pytest
from conftest import make_model, make_registry

from lai import checks
from lai.checks import Severity
from lai.paths import LDR_ENV_FILE


def findings(registry, host):
    return checks.run(registry, host)


def messages(found, severity=None):
    return [str(f) for f in found if severity is None or f.severity is severity]


def assert_error(found, fragment):
    errors = messages(found, Severity.ERROR)
    assert any(fragment in e for e in errors), f"no ERROR containing {fragment!r} in {errors}"


def assert_warn(found, fragment):
    warns = messages(found, Severity.WARN)
    assert any(fragment in w for w in warns), f"no WARN containing {fragment!r} in {warns}"


def test_clean_registry_has_no_findings(host_for):
    registry = make_registry(make_model(vision=True, mmproj="m/mmproj.gguf"))
    assert findings(registry, host_for(registry)) == []


def test_errors_sort_before_warnings(host_for):
    registry = make_registry(make_model(split_mode="row", parallel=2))
    found = findings(registry, host_for(registry))
    assert [f.severity for f in found] == sorted((f.severity for f in found),
                                                 key=lambda s: s is not Severity.ERROR)
    assert found[0].severity is Severity.ERROR


# --- registry rules ----------------------------------------------------------------


def test_missing_toolchain(host_for):
    registry = make_registry()
    host = host_for(registry, missing={"/bin/llama-server", "/opt/intel/oneapi/setvars.sh"})
    found = findings(registry, host)
    assert_error(found, "llama-server not found")
    assert_error(found, "setvars.sh not found")


def test_toolchain_not_required_for_vllm_only_registry(host_for):
    registry = make_registry(make_model(engine="vllm", launcher="--max-model-len $CONTEXT "
                                        "--served-model-name $MODEL_ID"))
    host = host_for(registry, missing={"/bin/llama-server"})
    assert messages(findings(registry, host), Severity.ERROR) == []


def test_duplicate_routing_key(host_for):
    registry = make_registry(make_model(id="a"), make_model(id="a", enabled=False))
    assert_error(findings(registry, host_for(registry)), "duplicate routing key 'a'")


def test_client_default_must_be_enabled(host_for):
    registry = make_registry(make_model(id="a"), small_model="gone")
    assert_error(findings(registry, host_for(registry)), "small_model='gone'")


@pytest.mark.parametrize(("build", "expect_error"), [(10663, True), (10664, False), (10729, False)])
def test_architecture_minimum_build(host_for, build, expect_error):
    registry = make_registry(make_model(arch="qwen4exp"))
    found = findings(registry, host_for(registry, build=build))
    assert any("b10664" in m for m in messages(found, Severity.ERROR)) is expect_error


def test_unreadable_build_number_is_a_warning(host_for):
    registry = make_registry(make_model(arch="qwen4exp"))
    assert_warn(findings(registry, host_for(registry, build=None)), "could not read the llama.cpp build")


def test_ldr_env_model_must_be_enabled(host_for):
    registry = make_registry(make_model(id="a"))
    host = host_for(registry, files={LDR_ENV_FILE: "LDR_LLM_MODEL=renamed-model\n"})
    assert_warn(findings(registry, host), "LDR_LLM_MODEL=renamed-model")


def test_opencode_extra_must_not_replace_generated_keys(host_for):
    registry = make_registry(opencode_extra={"provider": {}})
    assert_warn(findings(registry, host_for(registry)), "opencode_extra sets 'provider'")


# --- common model rules ------------------------------------------------------------


def test_unknown_values(host_for):
    registry = make_registry(make_model(device="SYCL7", flash_attn="yes"))
    found = findings(registry, host_for(registry))
    assert_error(found, "device='SYCL7'")
    assert_error(found, "flash_attn='yes'")


def test_model_too_big_for_pinned_card(host_for):
    registry = make_registry(make_model(device="SYCL0", vram_gb=31.5))
    assert_error(findings(registry, host_for(registry)), "pinned to SYCL0")


def test_small_context_warns_for_tool_models(host_for):
    registry = make_registry(make_model(context=16384))
    assert_warn(findings(registry, host_for(registry)), "tight for opencode")


def test_missing_weights_error_for_llama_cpp_warn_for_vllm(host_for):
    gguf = make_registry(make_model(weights="x.gguf"))
    assert_error(findings(gguf, host_for(gguf, missing={"x.gguf"})), "no file matches")
    vllm = make_registry(make_model(weights="x/config.json", engine="vllm",
                                    launcher="--max-model-len $CONTEXT --served-model-name $MODEL_ID"))
    found = findings(vllm, host_for(vllm, missing={"x/config.json"}))
    assert_warn(found, "no file matches")
    assert messages(found, Severity.ERROR) == []


# --- llama.cpp rules ------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--mmproj", "-c", "--ctx-size=8192", "-ngl", "--n-cpu-moe"])
def test_extra_args_cannot_shadow_a_field(host_for, flag):
    registry = make_registry(make_model(extra_args=[flag, "1"]))
    assert_error(findings(registry, host_for(registry)), "already controls")


def test_extra_args_cannot_shadow_a_sampler_key(host_for):
    registry = make_registry(make_model(sampler={"top_p": 0.9}, extra_args=["--top-p", "0.8"]))
    assert_error(findings(registry, host_for(registry)), "`sampler` already controls")


def test_vision_requires_projector_and_projector_implies_vision(host_for):
    no_projector = make_registry(make_model(vision=True))
    assert_error(findings(no_projector, host_for(no_projector)), "vision=True with no mmproj")
    unused = make_registry(make_model(mmproj="p.gguf"))
    assert_warn(findings(unused, host_for(unused)), "vision=False")
    missing = make_registry(make_model(vision=True, mmproj="p.gguf"))
    assert_error(findings(missing, host_for(missing, missing={"p.gguf"})), "mmproj missing")


@pytest.mark.parametrize(("mode", "fragment"), [("tensor", "56.7 t/s"), ("row", "segfaults")])
def test_known_bad_split_modes(host_for, mode, fragment):
    registry = make_registry(make_model(split_mode=mode))
    assert_error(findings(registry, host_for(registry)), fragment)


@pytest.mark.parametrize(
    ("kv_type", "blocked"), [("q4_0", True), ("q4_1", True), ("q8_0", False), (None, False)]
)
def test_four_bit_kv_blocked(host_for, kv_type, blocked):
    registry = make_registry(make_model(kv_type=kv_type))
    found = findings(registry, host_for(registry))
    assert any("tool-call argument JSON" in m for m in messages(found, Severity.ERROR)) is blocked


def test_parallel_slots_warn_with_per_request_context(host_for):
    registry = make_registry(make_model(context=131072, parallel=2))
    assert_warn(findings(registry, host_for(registry)), "each request gets 65536")


def test_cpu_offload_should_not_be_mapped(host_for):
    registry = make_registry(make_model(cpu_moe_layers=20))
    assert_warn(findings(registry, host_for(registry)), "0.1 t/s")
    loaded = make_registry(make_model(cpu_moe_layers=20, load_mode="none"))
    assert not any("0.1 t/s" in m for m in messages(findings(loaded, host_for(loaded))))


def test_qwen4exp_constraints(host_for):
    registry = make_registry(make_model(arch="qwen4exp", ubatch=4096, kv_type="q8_0", load_mode="none",
                                        cpu_moe_layers=10, extra_args=["-ot", "per_layer_token_embd=CPU"]))
    found = findings(registry, host_for(registry))
    assert_error(found, "ubatch=4096")
    assert_error(found, "needs f16 KV")
    assert_error(found, "load_mode='none'")
    assert_warn(found, "wastes ~11 GB")


def test_qwen4exp_rules_do_not_apply_to_other_architectures(host_for):
    registry = make_registry(make_model(ubatch=4096))
    assert findings(registry, host_for(registry)) == []


# --- vLLM rules -----------------------------------------------------------------------


def vllm(**overrides):
    fields = {"engine": "vllm", "context": 32768,
              "launcher": 'serve --max-model-len "$CONTEXT" --served-model-name "$MODEL_ID"'}
    return make_model(**{**fields, **overrides})


def test_vllm_launcher_using_variables_is_clean(host_for):
    registry = make_registry(vllm())
    assert findings(registry, host_for(registry)) == []


def test_vllm_launcher_literal_values_must_match(host_for):
    matching = make_registry(vllm(launcher="--max-model-len 32768 --served-model-name m"))
    assert findings(matching, host_for(matching)) == []
    drifted = make_registry(vllm(launcher="--max-model-len=16384 --served-model-name other"))
    found = findings(drifted, host_for(drifted))
    assert_error(found, "--max-model-len 16384, registry says 32768")
    assert_error(found, "--served-model-name other, registry says m")


def test_vllm_requires_launcher_and_single_slot(host_for):
    no_launcher = make_registry(vllm(launcher=None))
    assert_error(findings(no_launcher, host_for(no_launcher)), "needs a launcher")
    parallel = make_registry(vllm(parallel=2))
    assert_error(findings(parallel, host_for(parallel)), "no --parallel")


def test_vllm_launcher_without_flag_warns(host_for):
    registry = make_registry(vllm(launcher="vllm serve /model"))
    assert_warn(findings(registry, host_for(registry)), "has no --max-model-len")


def test_llama_cpp_rules_skip_vllm(host_for):
    registry = make_registry(vllm(split_mode="tensor", kv_type="q4_0"))
    assert findings(registry, host_for(registry)) == []


def test_every_rule_is_registered():
    """A rule function that is not in a table never runs; catch that here."""
    registered = set(checks.REGISTRY_RULES) | set(checks.COMMON_RULES)
    for rules in checks.ENGINE_RULES.values():
        registered |= set(rules)
    public = {
        obj for name, obj in vars(checks).items()
        if callable(obj) and getattr(obj, "__module__", "") == "lai.checks"
        and not name.startswith("_") and name not in {"run", "has_errors", "error", "warn"}
        and not isinstance(obj, type)
    }
    assert public <= registered, f"unregistered rules: {sorted(f.__name__ for f in public - registered)}"

