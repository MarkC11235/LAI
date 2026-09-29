"""Registry for mark-Victus: HP Victus laptop, RTX 4050 Laptop GPU (6 GB), CUDA build.

Selected automatically on this machine by hostname (see lai/paths.py). Same
workflow as models.py:  lai check && lai gen && lai restart && lai smoke <id>

Starts with one tiny model to prove the plumbing (proxy, wrapper, CUDA build,
opencode config) before anything that needs tuning for 6 GB of VRAM.
"""

from pathlib import Path

from lai.schema import Model, Settings

HOME = Path.home()

SETTINGS = Settings(
    # CUDA build (-DGGML_CUDA=ON, sm_89). Needs no oneAPI or other environment.
    llama_server=HOME / "llama.cpp/build-cuda/bin/llama-server",
    oneapi_setvars=None,
    # 6141 MiB total. The display runs on the AMD iGPU (PRIME on-demand), so
    # only the CUDA context comes off the top.
    cards={"CUDA0": 5.5},
    default_model="qwen3-0.6b",
    small_model="qwen3-0.6b",
    # No searxng MCP here yet: it needs Docker and node on this machine, so
    # opencode's built-in webfetch stays enabled.
)

# Plumbing test: small enough to load in seconds, still a thinking model with
# tool calls, so every smoke rung exercises something real.
QWEN3_TINY = Model(
    id="qwen3-0.6b",
    name="Qwen3-0.6B Q8_0 · CUDA0 · plumbing test",
    weights="Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf",
    # 32768 is the smallest context the check accepts without a warning for
    # opencode, and it holds the smoke test's ~18k-token prefill.
    context=32768,
    # ~0.6 GiB weights + ~3.5 GiB f16 KV at 32k (28 layers x 8 KV heads x 128 dims)
    # + compute buffers. An estimate: confirm with nvidia-smi while it is loaded.
    vram_gb=4.5,
    device="CUDA0",
    reasoning_field="reasoning_content",
    max_output=8192,
)

MODELS: list[Model] = [
    QWEN3_TINY,
]
