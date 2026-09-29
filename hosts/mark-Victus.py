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
    # 6141 MiB total; the display runs on the AMD iGPU (PRIME on-demand).
    # Largest load measured: 5724 MiB (Bonsai PTQ1_0, -ngl 52). One more layer
    # needed ~230 MiB more and failed, so usable is between the two.
    cards={"CUDA0": 5.6},
    default_model="qwen35-9b",
    small_model="qwen35-9b",
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

# Qwen3.5-9B (hybrid Gated-DeltaNet, dense), IQ4_XS, everything on the GPU.
# Measured 2026-09-29 on this card:
#   f16 KV, -ub 2048 or 512: out of memory before the compute buffer.
#   auto-fit (no -ngl): 4.4 GiB, some layers on CPU, 20.5 t/s decode.
#   q8_0 KV, -ub 512, -ngl 99: 5.3 GiB, 30.8 t/s decode. <- this entry
QWEN35_9B = Model(
    id="qwen35-9b",
    name="Qwen3.5-9B IQ4_XS · CUDA0",
    weights="Qwen3.5-9B/Qwen3.5-9B-IQ4_XS.gguf",
    context=32768,
    max_output=8192,
    vram_gb=5.2,  # nvidia-smi under LAI (--parallel 1): 5296 MiB
    device="CUDA0",
    kv_type="q8_0",  # halves the KV cache; f16 does not fit alongside the weights
    ubatch=512,      # the compute buffer scales with this; 2048 does not fit
    batch=2048,
    reasoning_field="reasoning_content",
)

# Measured 2026-09-29, q8_0 KV, -ub 512, 32k context, each config run through a
# full-context prefill (loading alone proves nothing: CUDA grows a scratch pool for
# long prompts after load, and -ngl 44 loaded fine, then died on the first long one).
#   PTQ1_0 (5.9 GB): short-prompt tests only; 7.2 t/s decode at -ngl 52.
#   PQ2_0  (7.2 GB): -ngl 43 peaks at 5666 MiB with 31.5k tokens in context;
#                    prefill 242 t/s (130 s for 31.5k); decode 9.0 t/s at short
#                    context, 5.0 at 20k, 4.0 at 31.5k.                  <- this entry
#   PQ2_0 keeps more layers on the CPU than PTQ1_0 and is still faster: the
#   fork's PTQ1_0 CPU kernels are slow (a listed known issue), PQ2_0's are not.
#   -t 6 / 8 / 12: 9.0 / 8.6 / 7.8 t/s. One thread per physical core.
BONSAI_27B = Model(
    id="bonsai-27b",
    name="Ternary Bonsai 2 27B PQ2_0 · 44/64 layers on CUDA0 · fork",
    weights="Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf",
    llama_server=HOME / "llama.cpp-prism/build-cuda/bin/llama-server",
    context=32768,
    max_output=8192,
    vram_gb=5.53,    # 5666 MiB peak, with a full 32k context
    device="CUDA0",
    gpu_layers=43,   # 45 needs ~230 MiB more than the card has
    kv_type="q8_0",
    ubatch=512,
    batch=2048,
    reasoning_field="reasoning_content",
    # From the model card: thinking-mode sampling. The GGUF lacks min_p.
    sampler={"temp": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05},
    # Default effort is xhigh, which often spends the whole budget reasoning and
    # returns an empty answer; "high" is rejected with HTTP 500. medium is the
    # model card's workaround, and was tested as a top-level request field.
    request_params={"reasoning_effort": "medium"},
    extra_args=["-t", "6"],  # measured best; the CPU kernels are compute-bound
    notes=(
        "20 of 64 layers run on the CPU: ~9 t/s decode. Stronger than qwen35-9b, "
        "3x slower. Fork build (llama.cpp-prism). The server rejects requests with "
        "more than one system message or a non-leading one (HTTP 500)."
    ),
)

MODELS: list[Model] = [
    QWEN35_9B,
    BONSAI_27B,
    QWEN3_TINY,
]
