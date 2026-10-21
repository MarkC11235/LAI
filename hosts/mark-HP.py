from pathlib import Path

from lai.schema import Model, Settings

HOME = Path.home()

SETTINGS = Settings(
    llama_server=HOME / "llama.cpp/build-cuda/bin/llama-server",
    oneapi_setvars=None,
    cards={"CUDA0": 7.5},
    default_model="qwen35-9b",
    small_model="qwen35-9b",
    # ── MCP setup on a new host (one-time) ──────────────────────────────────
    # This block must live in THIS host's registry (hosts/<hostname>.py); on the
    # multi-host branch models.py is not read on other machines. If `lai gen`
    # says opencode.json "unchanged" after adding it, it's in the wrong file.
    #
    # 1. Node from apt (paths below assume /usr/bin/node, not nvm/conda):
    #      sudo apt install -y nodejs npm
    #      mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global
    # 2. MCP servers:
    #      npm install -g mcp-searxng @playwright/mcp
    # 3. Chromium matching the bundled playwright version:
    #      node ~/.npm-global/lib/node_modules/@playwright/mcp/node_modules/playwright/cli.js \
    #        install --with-deps chromium
    #    (if --with-deps refuses the distro: drop it, then `ldd` the chrome
    #     binary under ~/.cache/ms-playwright and apt-install what's missing)
    # 4. SearXNG on :8888:
    #      docker run -d --name searxng --restart unless-stopped \
    #        -p 127.0.0.1:8888:8080 -v ~/searxng:/etc/searxng searxng/searxng
    #    The image writes a minimal settings.yml with NO json format → 403.
    #    Overwrite it (file is owned by the container's uid, so sudo):
    #      sudo tee ~/searxng/settings.yml >/dev/null <<EOF
    #      use_default_settings: true
    #      server:
    #        secret_key: "$(openssl rand -hex 32)"
    #        limiter: false        # limiter needs Valkey; pointless on localhost
    #      search:
    #        formats: [html, json]
    #      EOF
    #      docker restart searxng
    # 5. Verify:
    #      curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 200   # JSON, not 403
    #      lai check && lai gen          # restart opencode after (not lai restart)
    #      opencode mcp list             # searxng + playwright connected
    # ────────────────────────────────────────────────────────────────────────
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
            # Headless Chromium so the agent can load pages it builds and read
            # console errors. Same rule as searxng: absolute node + entry point,
            # no npx/PATH. WebGL headless runs on SwiftShader (slow, but enough
            # to catch shader/context errors).
            # RUN:
            #   npm install -g @playwright/mcp
            #   node ~/.npm-global/lib/node_modules/@playwright/mcp/node_modules/playwright/cli.js install chromium
            "playwright": {
                "type": "local",
                "command": [
                    "/usr/bin/node",
                    str(HOME / ".npm-global/lib/node_modules/@playwright/mcp/cli.js"),
                    "--browser", "chromium",
                    "--headless",
                    "--isolated",
                    ],
                "enabled": True,
                },
        },            
        # Two URL readers make a local model pick the wrong one. The MCP
        # `web_url_read` wins: it has length and section controls, webfetch has none.
        "tools": {"webfetch": False},
    },
)

# Qwen3.5/3.6 thinking mode, "precise coding" preset from the model cards
# (verified 2026-10-09 against Qwen3.5-9B and Qwen3.6-35B-A3B; identical).
# The general preset is temp 1.0 + presence_penalty 1.5; the penalty fights code
# that legitimately repeats identifiers, so the coding preset suits opencode.
# repeat_penalty is stated so a llama.cpp default change can't alter behaviour.
QWEN_THINKING_SAMPLER = {
    "temp": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}

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
    sampler=QWEN_THINKING_SAMPLER,
)

# ===========================================================================
# LFM2.5-24B-A2B (MoE, arch lfm2moe), QAD Q4_0, RTX 2060 Super + CPU experts
# ===========================================================================

# Liquid's recommended sampler (the base model's generation_config.json). The
# GGUF also carries these as general.sampling.*; stated here so a llama.cpp
# default change can't alter behaviour, same as QWEN_THINKING_SAMPLER.
LFM_SAMPLER = {
    "temp": 0.1,
    "top_k": 50,
    "repeat_penalty": 1.05,
}

LFM25_MOE = Model(
    id="lfm25-24b",
    name="LFM2.5-24B-A2B QAD-Q4_0 · CUDA0 · CPU experts",
    arch="lfm2moe",
    # Quantization-aware-distilled Q4_0: same 13.6 GB as Qwen3.6 IQ3_XXS,
    # but a real 4-bit trained to recover the quantization loss.
    weights="LFM2.5-24B-A2B/LFM2.5-24B-A2B-QAD-Q4_0.gguf",
    context=32768,
    max_output=16384,
    # 8 GB card. Most experts live in host RAM; this is what stays on the GPU.
    # Re-measure with nvidia-smi once cpu_moe_layers is tuned.
    vram_gb=6.5,
    device="CUDA0",
    # Experts of N layers on the CPU. Start high, lower until the card is
    # ~7.5 GB used; each layer moved back to the GPU is a speed gain.
    cpu_moe_layers=24,
    # mmap with CPU-offloaded weights measured 0.1 vs 10 t/s (docs/findings.md).
    # The CPU share (~10 GB) fits in 16 GB RAM, so read it in fully.
    load_mode="none",
    # Smaller than the B70 entries: the compute buffer comes out of 8 GB.
    ubatch=1024,
    batch=2048,
    sampler=LFM_SAMPLER,
    reasoning_field=None,  # instruct model, no separate reasoning stream
    ttl=3600,  # cold load reads ~10 GB into RAM; don't evict on short idle
    notes=(
        "LFM tool-call format is its own; confirm opencode gets structured "
        "tool_calls (not text) before relying on it. Fallback: the Qwen3.5-9B entry."
    ),
)

MODELS: list[Model] = [
    QWEN35_9B,
    LFM25_MOE,
]

