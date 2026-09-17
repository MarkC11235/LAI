"""`lai check`: reject registry configurations that are known to fail.

Every ERROR here is a failure that already happened on this hardware, and its
message says what happened. That is the bar for adding one: a rule without a
recorded failure behind it is an opinion, and opinions are WARNs.

Rules are plain generator functions that yield `Finding`s:

    registry rules   rule(registry, host)          whole-registry invariants
    model rules      rule(model, registry, host)   run once per enabled model

To add a rule: write the function, append it to the matching tuple at the
bottom of this file, add a test that triggers it (tests/test_checks.py), and
record the failure behind it in docs/findings.md.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from lai.host import Host
from lai.paths import LDR_ENV_FILE
from lai.registry import Registry
from lai.schema import ENGINES, FLASH_ATTN_MODES, LOAD_MODES, SPLIT_MODES, Model


class Severity(enum.Enum):
    ERROR = "FAIL"  # blocks `lai gen` unless --force
    WARN = "NOTE"  # printed; does not block


@dataclass(frozen=True)
class Finding:
    severity: Severity
    message: str
    model_id: str | None = None

    def __str__(self) -> str:
        return f"{self.model_id}: {self.message}" if self.model_id else self.message


def error(message: str) -> Finding:
    return Finding(Severity.ERROR, message)


def warn(message: str) -> Finding:
    return Finding(Severity.WARN, message)


def run(registry: Registry, host: Host) -> list[Finding]:
    """Every finding for the registry, errors first."""
    findings: list[Finding] = []
    for registry_rule in REGISTRY_RULES:
        findings.extend(registry_rule(registry, host))
    for model in registry.active():
        rules = COMMON_RULES + ENGINE_RULES.get(model.engine, ())
        for model_rule in rules:
            findings.extend(replace(f, model_id=model.id) for f in model_rule(model, registry, host))
    return sorted(findings, key=lambda f: f.severity is not Severity.ERROR)


def has_errors(findings: Iterable[Finding]) -> bool:
    return any(f.severity is Severity.ERROR for f in findings)


# ===========================================================================
# Registry rules
# ===========================================================================

# Architectures newer than some builds: the first llama.cpp build that has them.
ARCH_MIN_BUILD = {"qwen4exp": 10664}


def toolchain_present(registry: Registry, host: Host) -> Iterable[Finding]:
    """The native binaries every llama.cpp entry starts through."""
    if not any(m.engine == "llama.cpp" for m in registry.active()):
        return
    settings = registry.settings
    if not host.exists(settings.llama_server):
        yield error(f"llama-server not found at {settings.llama_server}")
    if not host.exists(settings.oneapi_setvars):
        yield error(f"oneAPI setvars.sh not found at {settings.oneapi_setvars}")


def routing_keys_unique(registry: Registry, host: Host) -> Iterable[Finding]:
    """Two entries claiming one key would silently shadow each other in llama-swap."""
    seen: set[str] = set()
    for model in registry.models:
        if model.id in seen:
            yield error(f"duplicate routing key {model.id!r}")
        seen.add(model.id)


def client_defaults_enabled(registry: Registry, host: Host) -> Iterable[Finding]:
    enabled = {m.id for m in registry.active()}
    settings = registry.settings
    for setting, model_id in (
        ("default_model", settings.default_model),
        ("small_model", settings.small_model),
    ):
        if model_id not in enabled:
            yield error(f"Settings.{setting}={model_id!r} is not an enabled model")


def build_supports_architectures(registry: Registry, host: Host) -> Iterable[Finding]:
    needed = {m.id: ARCH_MIN_BUILD[m.arch] for m in registry.active() if m.arch in ARCH_MIN_BUILD}
    if not needed:
        return
    build = host.llama_build()
    if build is None:
        yield warn(
            "could not read the llama.cpp build number, so the minimum-build check "
            f"was skipped for: {', '.join(needed)}"
        )
        return
    for model_id, minimum in needed.items():
        if build < minimum:
            yield error(
                f"{model_id} needs llama.cpp b{minimum} or newer; "
                f"{registry.settings.llama_server} is b{build}"
            )


_LDR_MODEL = re.compile(r"""^\s*LDR_LLM_MODEL\s*=\s*["']?([^"'\s#]+)""", re.MULTILINE)


def ldr_model_enabled(registry: Registry, host: Host) -> Iterable[Finding]:
    """ldr.env names a model by routing key; renaming the model breaks research runs."""
    text = host.read_text(LDR_ENV_FILE)
    match = _LDR_MODEL.search(text) if text else None
    if match and match.group(1) not in {m.id for m in registry.active()}:
        yield warn(
            f"ldr.env sets LDR_LLM_MODEL={match.group(1)}, which is not an enabled "
            "model; every Local Deep Research request would fail"
        )


# Keys of opencode.json that the generator derives from the registry.
GENERATED_OPENCODE_KEYS = ("provider", "model", "small_model")


def opencode_extra_does_not_replace_generated(registry: Registry, host: Host) -> Iterable[Finding]:
    """opencode_extra is deep-merged last, so it can silently override the routing keys."""
    for key in GENERATED_OPENCODE_KEYS:
        if key in registry.settings.opencode_extra:
            yield warn(
                f"Settings.opencode_extra sets {key!r}, which is generated from the registry; "
                "the override can reintroduce the routing-key drift generation exists to prevent"
            )


# ===========================================================================
# Model rules: every engine
# ===========================================================================


def known_values(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    """Literal types are not enforced at runtime; a typo would otherwise pass through."""
    if model.engine not in ENGINES:
        yield error(f"engine={model.engine!r}; expected one of {ENGINES}")
    cards = registry.settings.cards
    if model.device is not None and model.device not in cards:
        yield error(
            f"device={model.device!r} is not in Settings.cards ({', '.join(cards)}), "
            "so co-residency would be generated wrong"
        )
    if model.engine == "llama.cpp":
        if model.split_mode not in SPLIT_MODES:
            yield error(f"split_mode={model.split_mode!r}; expected one of {SPLIT_MODES}")
        if model.flash_attn not in FLASH_ATTN_MODES:
            yield error(f"flash_attn={model.flash_attn!r}; expected one of {FLASH_ATTN_MODES}")
        if model.load_mode is not None and model.load_mode not in LOAD_MODES:
            yield error(f"load_mode={model.load_mode!r}; expected None or one of {LOAD_MODES}")


def fits_pinned_card(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    capacity = registry.settings.cards.get(model.device) if model.device else None
    if capacity is not None and model.vram_gb > capacity:
        yield error(
            f"{model.vram_gb:.0f} GB pinned to {model.device}, which has ~{capacity:.0f} GB usable"
        )


def context_fits_agent_prompts(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if model.tools and model.slot_context < 32768:
        yield warn(
            f"{model.slot_context} tokens per request is tight for opencode: its system "
            "prompt and tool schemas use 10-15k before the first message"
        )


def weights_present(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if host.find(model.weights) is not None:
        return
    path = registry.settings.model_dir / model.weights
    if model.engine == "vllm":
        # The launcher mounts the directory itself; this path only confirms the download.
        yield warn(f"no file matches {path}; `lai ls` will show it as missing")
    else:
        yield error(f"no file matches {path}")


# ===========================================================================
# Model rules: llama.cpp
# ===========================================================================

# Flags the generator already emits from a field. llama.cpp keeps the last
# occurrence of a repeated flag, so any of these in extra_args silently wins
# over the field. That is how a stale --mmproj overrode the mmproj field.
FIELD_OWNED_FLAGS = {
    "-m": "weights", "--model": "weights",
    "--mmproj": "mmproj",
    "--port": "the proxy", "--host": "Settings.listen_host",
    "-a": "id", "--alias": "id",
    "-dev": "device", "--device": "device",
    "-ngl": "gpu_layers", "--gpu-layers": "gpu_layers", "--n-gpu-layers": "gpu_layers",
    "-sm": "split_mode", "--split-mode": "split_mode",
    "-ncmoe": "cpu_moe_layers", "--n-cpu-moe": "cpu_moe_layers",
    "--load-mode": "load_mode",
    "-c": "context", "--ctx-size": "context",
    "-ub": "ubatch", "--ubatch-size": "ubatch",
    "-b": "batch", "--batch-size": "batch",
    "-np": "parallel", "--parallel": "parallel",
    "-fa": "flash_attn", "--flash-attn": "flash_attn",
    "-ctk": "kv_type", "--cache-type-k": "kv_type",
    "-ctv": "kv_type", "--cache-type-v": "kv_type",
}  # fmt: skip


def extra_args_do_not_shadow_fields(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    sampler_flags = {f"--{key.replace('_', '-')}" for key in model.sampler}
    for token in model.extra_args:
        flag = token.split("=", 1)[0]
        owner = FIELD_OWNED_FLAGS.get(flag) or ("sampler" if flag in sampler_flags else None)
        if owner:
            yield error(
                f"extra_args sets {flag}, which `{owner}` already controls; llama.cpp "
                "keeps the last occurrence, so the field would be silently ignored"
            )


def vision_has_projector(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    model_dir = registry.settings.model_dir
    if model.mmproj and host.find(model.mmproj) is None:
        yield error(f"mmproj missing: {model_dir / model.mmproj}")
    if model.vision and not model.mmproj:
        yield error(
            "vision=True with no mmproj: llama-swap and opencode would advertise image "
            "input that llama-server has no projector for"
        )
    if model.mmproj and not model.vision:
        yield warn(
            "mmproj is loaded but vision=False, so opencode strips images before they "
            "are sent and the projector occupies VRAM for nothing"
        )


def split_mode_known_bad(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if model.split_mode == "tensor":
        yield error(
            "split_mode='tensor' is blocked: with no GPU interconnect it measured 56.7 t/s "
            "decode vs 84.5 on one card, older builds hung on long prefill, and it "
            "disables memory auto-fit, so a model larger than VRAM hangs at load. Use 'layer'."
        )
    elif model.split_mode == "row":
        yield error("split_mode='row' segfaults at model load on dual-B70 SYCL")


def kv_type_known_bad(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if model.kv_type in ("q4_0", "q4_1"):
        yield error(
            f"kv_type={model.kv_type!r}: 4-bit KV corrupted tool-call argument JSON. "
            "Use 'q8_0', or None (f16, also the fastest when it fits)."
        )


def parallel_divides_context(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if model.parallel > 1:
        yield warn(
            f"parallel={model.parallel}: each request gets {model.slot_context} tokens, "
            f"not {model.context}; opencode is told {model.slot_context}"
        )


def offload_is_loaded_not_mapped(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if model.cpu_moe_layers and model.load_mode != "none":
        yield warn(
            f"{model.cpu_moe_layers} expert layers on CPU without load_mode='none': "
            "mmap pages hot weights from disk every token (measured 0.1 t/s vs 10 t/s)"
        )


# qwen4exp (Qwen3.8 Flash-Next): three constraints found the hard way.
QWEN4EXP_MAX_UBATCH = 2048  # its Qwen Sparse Attention budget


def qwen4exp_constraints(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if model.arch != "qwen4exp":
        return
    if model.ubatch > QWEN4EXP_MAX_UBATCH:
        yield error(
            f"ubatch={model.ubatch}: qwen4exp aborts with UR_RESULT_ERROR_DEVICE_LOST above "
            f"{QWEN4EXP_MAX_UBATCH} (its sparse-attention budget); 2048 runs, 2049 does not"
        )
    if model.kv_type is not None:
        yield error("qwen4exp needs f16 KV (kv_type=None): quantized KV breaks its sparse-attention path")
    if model.load_mode == "none":
        yield error(
            "load_mode='none' pins qwen4exp's ~29 GB n-gram table into host RAM and "
            "exceeds 64 GB; leave it memory-mapped (None)"
        )
    if model.cpu_moe_layers and any("per_layer_token_embd" in arg for arg in model.extra_args):
        yield warn(
            "cpu_moe_layers already places per_layer_token_embd on CPU; overriding it "
            "in extra_args as well wastes ~11 GB of VRAM"
        )


# ===========================================================================
# Model rules: vLLM
# ===========================================================================


def vllm_launcher_consistent(model: Model, registry: Registry, host: Host) -> Iterable[Finding]:
    if not model.launcher:
        yield error("engine='vllm' needs a launcher script")
        return
    if model.parallel != 1:
        yield error(
            "vLLM has no --parallel: slot_context would disagree with --max-model-len "
            "and opencode would compact at the wrong point"
        )
    yield from _launcher_flag_matches(
        model, "--max-model-len", "CONTEXT", str(model.context),
        consequence="opencode is told the registry value and would compact at the wrong point",
    )
    yield from _launcher_flag_matches(
        model, "--served-model-name", "MODEL_ID", model.id,
        consequence="llama-swap forwards the routing key unchanged and vLLM 404s any other name",
    )


def _launcher_flag_matches(
    model: Model, flag: str, variable: str, expected: str, *, consequence: str
) -> Iterable[Finding]:
    match = re.search(rf"{re.escape(flag)}(?:=|\s+)(\S+)", model.launcher or "")
    if not match:
        yield warn(f"launcher has no {flag}; cannot confirm it matches the registry")
        return
    value = match.group(1).strip("'\"\\")
    if value in (f"${variable}", f"${{{variable}}}", expected):
        return
    yield error(f"launcher passes {flag} {value}, registry says {expected}: {consequence}. Use ${variable}.")


# ===========================================================================
# Rule tables
# ===========================================================================

RegistryRule = Callable[[Registry, Host], Iterable[Finding]]
ModelRule = Callable[[Model, Registry, Host], Iterable[Finding]]

REGISTRY_RULES: tuple[RegistryRule, ...] = (
    toolchain_present,
    routing_keys_unique,
    client_defaults_enabled,
    build_supports_architectures,
    ldr_model_enabled,
    opencode_extra_does_not_replace_generated,
)

COMMON_RULES: tuple[ModelRule, ...] = (
    known_values,
    fits_pinned_card,
    context_fits_agent_prompts,
    weights_present,
)

ENGINE_RULES: dict[str, tuple[ModelRule, ...]] = {
    "llama.cpp": (
        extra_args_do_not_shadow_fields,
        vision_has_projector,
        split_mode_known_bad,
        kv_type_known_bad,
        parallel_divides_context,
        offload_is_loaded_not_mapped,
        qwen4exp_constraints,
    ),
    "vllm": (vllm_launcher_consistent,),
}
