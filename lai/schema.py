"""The registry schema: what `models.py` is allowed to say.

`models.py` is data. This module is the contract that data must satisfy, and
the one place a reader learns what a field does: every field names the
llama-server flag, llama-swap key, or opencode setting it becomes.

Nothing here touches the filesystem or the network, so the schema can be
imported anywhere (tests, CI) without the GPUs or the toolchain present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

HOME = Path.home()

Engine = Literal["llama.cpp", "vllm"]
ENGINES: tuple[str, ...] = ("llama.cpp", "vllm")

# "row" and "tensor" are valid llama.cpp values; they are listed so that
# `lai check` can reject them with the reason, rather than as unknown values.
SplitMode = Literal["layer", "none", "row", "tensor"]
SPLIT_MODES: tuple[str, ...] = ("layer", "none", "row", "tensor")

FlashAttn = Literal["on", "off", "auto"]
FLASH_ATTN_MODES: tuple[str, ...] = ("on", "off", "auto")

LoadMode = Literal["none", "mmap", "mlock", "mmap+mlock", "dio"]
LOAD_MODES: tuple[str, ...] = ("none", "mmap", "mlock", "mmap+mlock", "dio")


@dataclass(frozen=True)
class Settings:
    """Machine-level configuration shared by every model."""

    # --- toolchain ------------------------------------------------------------
    llama_server: Path = HOME / "llama.cpp/build/bin/llama-server"
    oneapi_setvars: Path = Path("/opt/intel/oneapi/setvars.sh")
    model_dir: Path = HOME / "models"
    """Root that `Model.weights` and `Model.mmproj` are relative to."""

    # --- proxy (llama-swap) ---------------------------------------------------
    listen_host: str = "127.0.0.1"
    """Deliberately loopback: llama-swap has no authentication."""
    listen_port: int = 9090
    """The only inference URL any client uses."""
    backend_start_port: int = 10001
    """llama-swap hands each backend a ${PORT} counting up from here."""
    health_timeout_s: int = 900
    """Cold loads of 80+ GB off a cold page cache need minutes, not the default 120 s."""
    idle_ttl_s: int = 1800
    """Unload a model after this many idle seconds unless the model sets `ttl`."""
    activity_db: Path = HOME / ".local/state/llama-swap/activity.db"

    # --- hardware ---------------------------------------------------------------
    cards: dict[str, float] = field(default_factory=lambda: {"SYCL0": 31.0, "SYCL1": 32.0})
    """Usable GiB per card, keyed by llama.cpp device name, in device order.
    SYCL0 drives the display and loses ~1 GiB. Drives `device` validation, the
    per-card size check, and the co-residency matrix."""

    # --- opencode -----------------------------------------------------------------
    opencode_config: Path = HOME / ".config/opencode/opencode.json"
    default_model: str = ""
    """opencode's startup model. Must be an enabled model id."""
    small_model: str = ""
    """Used for session titles. Keep it local so nothing leaves the machine."""
    opencode_extra: dict = field(default_factory=dict)
    """Deep-merged into the generated opencode.json: `mcp`, `permission`, ...
    Anything added by hand to the installed file is lost on the next `lai gen`,
    so it belongs here instead."""

    @property
    def base_url(self) -> str:
        return f"http://{self.listen_host}:{self.listen_port}"


@dataclass(frozen=True)
class Model:
    """One servable model. `id` is its routing key everywhere."""

    # --- identity -------------------------------------------------------------------
    id: str
    """Routing key: llama-swap model key, llama-server alias (-a), opencode model
    id, vLLM --served-model-name. Renaming it breaks every client that uses it."""
    name: str
    """Display name in the llama-swap UI and opencode's model picker."""
    weights: str
    """Path relative to `Settings.model_dir`; globs allowed. For split GGUFs,
    point at shard 1 (llama.cpp finds the rest). For vLLM, any file inside the
    checkpoint directory, used only to confirm the download exists."""
    context: int
    """Total context (-c). See `slot_context` for what one request gets."""
    vram_gb: float
    """Approximate GiB the model occupies on its card(s). For vLLM this is what it
    *claims* (gpu-memory-utilization x card), not the weight size."""

    engine: Engine = "llama.cpp"
    enabled: bool = True
    notes: str = ""
    """Shown in the llama-swap UI. What an operator needs at a glance; the
    evidence belongs in docs/findings.md."""

    # --- placement ------------------------------------------------------------------
    device: str | None = None
    """A key of `Settings.cards` (--device) to pin the model to one card; pinned
    models on different cards may run concurrently. None = needs every card and
    runs alone."""
    arch: str | None = None
    """GGUF `general.architecture`. Enables architecture-specific rules in
    `lai check` (see lai/checks.py)."""

    # --- llama-server (engine="llama.cpp") --------------------------------------------
    gpu_layers: int = 99  # -ngl
    split_mode: SplitMode = "layer"  # -sm
    flash_attn: FlashAttn = "on"  # -fa
    cpu_moe_layers: int = 0  # --n-cpu-moe; 0 keeps every expert on the GPU
    load_mode: LoadMode | None = None  # --load-mode; None = llama.cpp default (mmap)
    kv_type: str | None = None  # -ctk / -ctv, e.g. "q8_0"; None = f16
    ubatch: int = 2048  # -ub
    batch: int = 4096  # -b
    parallel: int = 1  # --parallel; slots share `context` equally
    mmproj: str | None = None  # --mmproj, relative to Settings.model_dir
    sampler: dict[str, float | int | str] = field(default_factory=dict)
    """Server-side sampling defaults, emitted as `--<key> <value>`. Keys may use
    `_` or `-` (`top_p` becomes --top-p)."""
    extra_args: list[str] = field(default_factory=list)
    """Appended to the llama-server command, one argv token per item, exactly as
    subprocess would take them: ["-ot", r"blk\\.0\\.=CPU"]. Flags that a field
    above already controls are rejected by `lai check`."""

    # --- vLLM (engine="vllm") -----------------------------------------------------------
    launcher: str | None = None
    """Bash body of the generated gen/vllm-<id>.sh. llama-swap runs it with the
    backend port as $1. The generated header defines $MODEL_ID, $MODEL_DIR,
    $CONTEXT and $REPO_ROOT from this registry; use them instead of repeating
    the values."""

    # --- client-facing ----------------------------------------------------------------
    max_output: int = 8192
    """opencode's output-token budget (limit.output)."""
    tools: bool = True
    """Advertised to clients, and gates the tool-call smoke rung."""
    vision: bool = False
    """Advertise image input. llama.cpp models also need `mmproj`."""
    reasoning_field: str | None = None
    """Response field carrying reasoning text: "reasoning_content" (llama.cpp)
    or "reasoning" (vLLM). Without it opencode shows reasoning-only turns as blank."""
    request_params: dict = field(default_factory=dict)
    """Merged into every request body by llama-swap (setParams). Lets entries
    that share weights differ per request, e.g. reasoning effort."""
    ttl: int | None = None
    """Idle seconds before unload. None = the proxy-wide `Settings.idle_ttl_s`."""

    @property
    def slot_context(self) -> int:
        """Context available to a single request: llama-server divides `context`
        evenly across `parallel` slots. This, not `context`, is opencode's limit."""
        return self.context // max(1, self.parallel)
