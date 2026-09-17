"""The model registry: the only file edited to add, remove, or retune a model.

`lai gen` turns this into the llama-swap config, the backend launch scripts,
and opencode's config (see lai/render.py). Every field is documented in
lai/schema.py, and every non-default value below has a reason, recorded
either in a comment here or with its measurement in docs/findings.md.

After editing:  lai check && lai gen && lai restart && lai smoke <id>

Routing keys (`id`) are an interface: opencode sessions, ldr.env, and scripts
refer to them. Rename one only on purpose.
"""

from pathlib import Path

from lai.schema import Model, Settings

HOME = Path.home()

# ===========================================================================
# Machine and client settings
# ===========================================================================

SETTINGS = Settings(
    # b10729. qwen4exp (Flash-Next) needs b10664+; `lai check` verifies it.
    llama_server=HOME / "llama.cpp/build-new/bin/llama-server",
    default_model="qwen38-27b-c1-mtp",
    small_model="qwen38-27b-c1-mtp",  # session titles; local so nothing leaves the box
    opencode_extra={
        # Web search through the SearXNG container on :8888 (docs/deep-research.md).
        # Direct node invocation with absolute paths: opencode spawns this as a
        # subprocess, so it must not depend on PATH, conda, or the npm shebang.
        # The entry point is dist/cli.js, not index.js, as of mcp-searxng 1.16.
        "mcp": {
            "searxng": {
                "type": "local",
                "command": [
                    "/usr/bin/node",
                    str(HOME / ".npm-global/lib/node_modules/mcp-searxng/dist/cli.js"),
                ],
                "enabled": True,
                "environment": {"SEARXNG_URL": "http://127.0.0.1:8888"},
            },
        },
        # Two URL readers make a local model pick the wrong one. The MCP
        # `web_url_read` wins: it has length and section controls, webfetch has none.
        "tools": {"webfetch": False},
    },
)

# ===========================================================================
# Shared values
# ===========================================================================

# Qwen's recommended thinking-mode sampling. The two penalties are llama.cpp's
# defaults, stated so a future default change can't silently alter behaviour.
QWEN_THINKING_SAMPLER = {
    "temp": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}

# Image token budget for the vision projector. The floor keeps small images
# legible; the ceiling bounds prefill cost per image.
IMAGE_TOKEN_ARGS = ["--image-min-tokens", "1024", "--image-max-tokens", "2048"]

# Speculative decoding from the GGUF's own MTP head. Draft length 2: longer
# drafts from a single MTP layer lose acceptance faster than they gain tokens.
MTP_ARGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]


def effort(level: str) -> dict:
    """Per-request reasoning effort, injected by llama-swap into the chat template.

    It must be a request parameter, not `--chat-template-kwargs` in extra_args:
    that is how entries sharing weights differ, and the JSON quoting it needs on
    a command line was mangled once already.
    """
    return {"chat_template_kwargs": {"reasoning_effort": level}}


# ===========================================================================
# Qwen3.8-27B, vLLM XPU (GPTQ-Int4 with MTP speculative decoding)
# ===========================================================================

QWEN38_VLLM = Model(
    id="qwen38-vllm",
    name="Qwen3.8-27B GPTQ-Int4 · vLLM XPU · MTP2",
    engine="vllm",
    # Safetensors, not GGUF: llama.cpp's converter drops the mtp.* tensors
    # this checkpoint exists for. SergiioB's GPTQ-Int4, mtp.* at BF16, rev 9d189a60.
    weights="Qwen3.8-27B-GPTQ-Int4-MTP/model.safetensors.index.json",
    context=32768,
    max_output=8192,
    # What vLLM claims, not the 16.6 GB of weights: it takes
    # gpu-memory-utilization of the card at load and holds it.
    vram_gb=28,
    # Card 0 on purpose: vLLM's all-or-nothing claim would make it exclusive
    # with the SYCL1 models. The launcher's ZE_AFFINITY_MASK=0 must agree.
    device="SYCL0",
    ttl=3600,  # cold start is ~93 s without the compile cache, 12.7 s with it
    reasoning_field="reasoning",  # vLLM's field name, not "reasoning_content"
    # No llama-server CLI to map a sampler onto, so it rides in every request.
    request_params={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    notes=(
        "vLLM in Docker, not llama.cpp. Confirm 'XPUwNa16LinearKernel for "
        "AutoGPTQLinearMethod' in the startup log: without it the XMX integer path "
        "is off and the server runs at half speed with no error. MTP needs the two "
        "patches in vendor/vllm-xpu-mtp. GPTQ-Int4 quality vs the GGUF is unverified."
    ),
    # Plain bash. The generated header defines MODEL_ID, MODEL_DIR, CONTEXT and
    # REPO_ROOT; the image is pinned by digest because the MTP patches target it.
    launcher=r"""
PORT="${1:?usage: $0 PORT}"
IMAGE=vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
CONTAINER="vllm-$MODEL_ID"
PATCHES="$REPO_ROOT/vendor/vllm-xpu-mtp"

# MTP on B70 needs two monkeypatches to the nightly image (docs/findings.md).
for patch in patch_mtp_nightly.py patch_mtp_boundary.py; do
  if [[ ! -r "$PATCHES/$patch" ]]; then
    echo "FATAL: $PATCHES/$patch missing; see vendor/vllm-xpu-mtp/README.md" >&2
    exit 1
  fi
done

# A SIGKILLed run leaves the container name taken, and the next start fails
# with a name conflict that looks nothing like the real problem.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

serve_args=(
  /model
  --served-model-name "$MODEL_ID"
  --max-model-len "$CONTEXT"
  --port 8000
  --quantization gptq --dtype float16
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.88     # 0.90 without MTP; the draft buffers need the rest
  --max-num-seqs 8 --max-num-batched-tokens 8192
  --language-model-only
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder    # Qwen3.8 emits XML tool calls; "hermes" leaves them in content
  --reasoning-parser qwen3
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
)

# /dev/dri is both a device and a bind mount: oneCCL opens it by path and dies
# in ze_fd_manager without the mount, even at tensor-parallel size 1.
exec docker run --rm --name "$CONTAINER" \
  --device /dev/dri -v /dev/dri:/dev/dri:ro \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v "$MODEL_DIR/Qwen3.8-27B-GPTQ-Int4-MTP:/model:ro" \
  -v "$HOME/.cache/vllm-xpu:/cache" \
  -v "$PATCHES:/patches:ro" \
  -p "127.0.0.1:$PORT:8000" \
  -e VLLM_CACHE_ROOT=/cache \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" \
  -lc 'set -e
       python /patches/patch_mtp_nightly.py
       python /patches/patch_mtp_boundary.py
       exec vllm serve "$@"' vllm-serve "${serve_args[@]}"
""",
)

# ===========================================================================
# Qwen3.8 Flash-Next 125B-A6B (arch qwen4exp), UD-IQ3_XXS, both cards
# ===========================================================================


def flash_next(level: str) -> Model:
    """One entry per reasoning effort. They share weights, so switching reloads."""
    return Model(
        id=f"qwen38-flash-next-{level}",
        name=f"Qwen3.8 Flash-Next 125B-A6B IQ3_XXS · {level}",
        arch="qwen4exp",
        weights="Qwen3.8-Flash-Next/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
        mmproj="Qwen3.8-Flash-Next/mmproj-F16.gguf",
        vision=True,
        context=65536,
        max_output=32768,
        # Weights on the GPUs only; the ~29 GB n-gram (PLE) table stays in host RAM.
        vram_gb=52,
        device=None,  # needs both cards
        # The architecture's hard limits, all enforced by `lai check`:
        ubatch=2048,  # its sparse-attention budget; 2049 aborts with DEVICE_LOST
        kv_type=None,  # f16 only
        load_mode=None,  # mmap only; pinning the PLE table exceeds 64 GB of RAM
        ttl=3600,  # an 82 GB cold load; the default idle eviction makes it feel broken
        reasoning_field="reasoning_content",
        sampler={k: v for k, v in QWEN_THINKING_SAMPLER.items() if "penalty" not in k},
        request_params=effort(level),
        extra_args=[
            # Keep the 51B n-gram embedding table in host RAM; -ngl 99 would try to
            # put all of it on the cards. Never combine with cpu_moe_layers, which
            # already places this tensor: doing both wastes ~11 GB of VRAM.
            "-ot", r"per_layer_token_embd\.weight=CPU",
            *IMAGE_TOKEN_ARGS,
        ],
        notes=(
            "qwen4exp: 3 of 4 layers gated delta-net, the 4th Qwen Sparse Attention with a "
            "2048-token budget. Short prompts never reach the sparse path, so only the "
            "long-prefill smoke rung tests it. Do not use --slot-save-path: restores are "
            "accepted and never reused, so every request re-prefills."
        ),
    )


# ===========================================================================
# Qwen3.8-27B (dense), UD-Q5_K_XL, one card, MTP speculative decoding
# ===========================================================================


def qwen38_27b(card: str, level: str) -> Model:
    """The same 21 GB model pinned to either card, so two can run at once
    (e.g. an orchestrator and a subagent). Medium effort carries no suffix."""
    suffix = "" if level == "medium" else f"-{level}"
    return Model(
        id=f"qwen38-27b-c{card[-1]}-mtp{suffix}",
        name=f"Qwen3.8-27B Q5_K_XL · {card} · MTP · {level}",
        weights="Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf",
        mmproj="Qwen3.8-27B/mmproj-F16.gguf",
        vision=True,
        context=65536,
        max_output=32768,
        vram_gb=21,
        device=card,
        reasoning_field="reasoning_content",
        sampler=QWEN_THINKING_SAMPLER,
        request_params=effort(level),
        extra_args=[*MTP_ARGS, *IMAGE_TOKEN_ARGS],
        notes=(
            "MTP drafting combined with image input is untested; if vision output looks "
            "wrong, compare against a run without MTP_ARGS at temp 0."
        ),
    )


# ===========================================================================
# Qwen3.6-35B-A3B (MoE), UD-Q4_K_XL, one card, MTP
# ===========================================================================

QWEN36_MOE = Model(
    id="qwen36-moe-c1-mtp",
    name="Qwen3.6-35B-A3B Q4_K_XL · SYCL1 · MTP",
    weights="Qwen3.6-35B-A3B-MTP/UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    context=32768,
    max_output=16384,
    vram_gb=21,
    device="SYCL1",
    reasoning_field="reasoning_content",
    sampler=QWEN_THINKING_SAMPLER,
    request_params=effort("medium"),
    extra_args=MTP_ARGS,
)

# ===========================================================================
# The registry. Order is display order in `lai ls` and the opencode picker.
# ===========================================================================

MODELS: list[Model] = [
    QWEN38_VLLM,
    flash_next("medium"),
    flash_next("low"),
    qwen38_27b("SYCL0", "xhigh"),
    qwen38_27b("SYCL1", "xhigh"),
    qwen38_27b("SYCL0", "medium"),
    qwen38_27b("SYCL1", "medium"),
    QWEN36_MOE,
]
