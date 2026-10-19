from pathlib import Path

from lai.schema import Model, Settings

HOME = Path.home()

SETTINGS = Settings(
    llama_server=HOME / "llama.cpp/build-cuda/bin/llama-server",
    oneapi_setvars=None,
    cards={"CUDA0": 8},
    default_model="qwen35-9b",
    small_model="qwen35-9b",
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


MODELS: list[Model] = [
    QWEN35_9B,
]
