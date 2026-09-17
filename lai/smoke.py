"""`lai smoke`: the acceptance ladder a model passes before any client uses it.

Rungs run in order and stop at the first failure, because each one assumes the
previous ones hold. The order is cheapest-first, and the last rung is the one
that matters: an ~18k-token prefill. Every earlier rung has passed on configs
that then hung, crashed, or silently truncated on a real agent prompt.

Each rung is a function `(SmokeContext) -> str` that returns a pass message or
raises `RungFailed`. Rungs never print, so tests drive them with a fake proxy.
To add a rung: write the function, add it to `ladder()`, and note in
docs/operations.md what its failure means.
"""

from __future__ import annotations

import base64
import json
import struct
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from lai import term
from lai.proxy import Proxy, Response
from lai.schema import Model, Settings


class RungFailed(Exception):
    """A rung's failure. The message is shown as-is; `hint` is shown indented below it."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class RungSkipped(Exception):
    """The rung does not apply to this model (e.g. tools=False)."""


@dataclass
class SmokeContext:
    model: Model
    settings: Settings
    proxy: Proxy
    prefill_timeout_s: int


@dataclass(frozen=True)
class Rung:
    title: str
    run: Callable[[SmokeContext], str]


# ===========================================================================
# Rungs
# ===========================================================================


def registered(ctx: SmokeContext) -> str:
    response = ctx.proxy.list_models()
    if not response.ok:
        raise RungFailed(f"/v1/models returned {response.status}: {response.text[:200]}")
    ids = {entry.get("id") for entry in _json(response).get("data", [])}
    if ctx.model.id not in ids:
        known = sorted(i for i in ids if i)
        raise RungFailed(
            f"{ctx.model.id} is not in /v1/models. Known: {known}",
            hint="Regenerate and restart so the proxy sees the registry: lai gen && lai restart",
        )
    return f"routing key {ctx.model.id!r} resolves"


def loads_and_completes(ctx: SmokeContext) -> str:
    started = time.monotonic()
    response = ctx.proxy.chat(
        ctx.model.id,
        [{"role": "user", "content": "Say hello in five words."}],
        max_tokens=50,
        timeout=ctx.settings.health_timeout_s,
    )
    if not response.ok:
        hint = ""
        if "prematurely" in response.text or "exited" in response.text:
            hint = (
                "The backend died before serving anything. Bisect it:\n"
                "  lai env                 # does the environment wrapper work at all?\n"
                f"  lai run {ctx.model.id:<15} # this model in the foreground, errors visible"
            )
        raise RungFailed(f"HTTP {response.status}: {response.text[:400]}", hint)
    content = (_message(response).get("content") or "").strip()
    return f"{time.monotonic() - started:.0f}s including cold load: {content[:60]!r}"


def context_matches_client(ctx: SmokeContext) -> str:
    """opencode compacts at `slot_context`; the server truncates at its own n_ctx."""
    if ctx.model.engine != "llama.cpp":
        raise RungSkipped("no /props on this engine; `lai check` compares --max-model-len instead")
    response = ctx.proxy.props(ctx.model.id)
    if not response.ok:
        raise RungSkipped(f"/props returned {response.status}")
    props = _json(response)
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
    if n_ctx is None:
        raise RungSkipped("/props has no n_ctx")
    if int(n_ctx) != ctx.model.slot_context:
        raise RungFailed(
            f"server reports n_ctx={n_ctx}, opencode is told {ctx.model.slot_context}",
            hint="opencode would compact at the wrong point while the server silently "
            "truncates underneath it. Fix context/parallel in models.py, then lai gen.",
        )
    return f"n_ctx={n_ctx} == limit.context"


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def tool_call_parses(ctx: SmokeContext) -> str:
    if not ctx.model.tools:
        raise RungSkipped("tools=False")
    response = ctx.proxy.chat(
        ctx.model.id,
        [{"role": "user", "content": "List the files in the current directory."}],
        tools=[BASH_TOOL],
        max_tokens=1024,  # reasoning models think before they call
        timeout=300,
    )
    if not response.ok:
        raise RungFailed(f"HTTP {response.status}: {response.text[:400]}")
    message = _message(response)
    calls = message.get("tool_calls") or []
    if not calls:
        reasoning = _reasoning(message, ctx.model)[:200]
        raise RungFailed(
            "no tool_calls in the response",
            hint="If the call is sitting as text in the content or reasoning, the chat "
            "template or tool parser is not matching the model's format (llama.cpp: "
            f"--jinja and build age; vLLM: --tool-call-parser). reasoning={reasoning!r}",
        )
    try:
        json.loads(calls[0]["function"]["arguments"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RungFailed(
            "tool_calls present but the arguments are not valid JSON",
            hint="This is what an over-aggressive KV quantization looks like (kv_type q4_*).",
        ) from exc
    return f"{len(calls)} call(s), arguments parse as JSON"


def sees_images(ctx: SmokeContext) -> str:
    """Every text rung passes on a model whose projector never loaded: the model
    just says it cannot see images. Two colours, because one could be a guess."""
    if not ctx.model.vision:
        raise RungSkipped("vision=False")
    colours = {"red": (220, 20, 20), "blue": (20, 40, 210)}
    for name, rgb in colours.items():
        image = base64.b64encode(solid_png(rgb)).decode()
        response = ctx.proxy.chat(
            ctx.model.id,
            [{"role": "user", "content": [
                {"type": "text", "text": "What single colour fills this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ]}],
            # A reasoning model spends hundreds of tokens before any content, and a
            # truncated answer is an empty string that looks like a vision failure.
            max_tokens=1024,
            timeout=ctx.settings.health_timeout_s,
        )  # fmt: skip
        if not response.ok:
            hint = ""
            if "image" in response.text.lower() or "multimodal" in response.text.lower():
                hint = (
                    "That reads like the server has no projector. Confirm --mmproj is in "
                    f"the cmd block: grep -A25 '{ctx.model.id}:' gen/llama-swap.yaml"
                )
            raise RungFailed(f"HTTP {response.status}: {response.text[:400]}", hint)
        message = _message(response)
        # Read both: a thinking model often names the colour in its reasoning and
        # is cut off before writing content.
        said = f"{message.get('content') or ''} {_reasoning(message, ctx.model)}".lower()
        if name not in said:
            raise RungFailed(
                f"solid {name} image not identified: {said.strip()[:200]!r}",
                hint="Coherent but colour-blind: the vision encoder runs but is wrong. "
                "\"Cannot see images\": the image never reached the server.",
            )
        other = next(c for c in colours if c != name)
        if other in said:
            raise RungFailed(f"the answer for the {name} image also mentions {other}: "
                             "the model is not looking at this request's image")
    return "solid red and solid blue both identified"


PREFILL_PROMPT = "The quick brown fox jumps over the lazy dog. " * 2000  # ~18-20k tokens


def long_prefill(ctx: SmokeContext) -> str:
    """The rung that matters. It crosses every batch and attention-budget boundary
    a short prompt stays under (qwen4exp's sparse attention is dense below 2048
    tokens, so short prompts pass whether or not the sparse path works)."""
    started = time.monotonic()
    response = ctx.proxy.chat(
        ctx.model.id,
        [{"role": "user", "content": PREFILL_PROMPT}],
        max_tokens=20,
        timeout=ctx.prefill_timeout_s,
    )
    if response.status == 0:
        raise RungFailed(
            f"no response after {ctx.prefill_timeout_s}s: every other rung can pass while "
            "this one hangs, which is the failure this ladder exists to catch",
            hint=hang_evidence(ctx.model),
        )
    if not response.ok:
        raise RungFailed(f"HTTP {response.status}: {response.text[:400]}")
    elapsed = max(1e-3, time.monotonic() - started)
    prompt_tokens = _json(response).get("usage", {}).get("prompt_tokens", 0)
    return f"{elapsed:.0f}s, prompt_tokens={prompt_tokens}, ~{prompt_tokens / elapsed:.0f} tok/s prefill"


def hang_evidence(model: Model) -> str:
    lines = [
        "Collect evidence NOW, before killing anything:",
        "  sudo xpu-smi dump -d 0,1 -m 0,5 -n 5     # GPU utilisation and memory",
        "  free -h",
    ]
    if model.engine == "llama.cpp":
        lines.append(
            "  sudo gdb -p $(pgrep -f 'llama-server.*-a " + model.id + "') -batch "
            "-ex 'thread apply all bt' > /tmp/hang-bt.txt"
        )
    else:
        lines.append("  lai logs | tail -100")
    lines.append("GPU pegged: a stuck compute kernel. GPU idle: blocked on a lock or socket;")
    lines.append("read the backtrace.")
    return "\n".join(lines)


def ladder(model: Model) -> list[Rung]:
    """The rungs for one model. The vision rung only exists for vision models, so
    the rung count (and the numbering operators quote) matches the original tool."""
    rungs = [
        Rung("registered with the proxy", registered),
        Rung("loads and completes a short prompt", loads_and_completes),
        Rung("server context matches what opencode is told", context_matches_client),
        Rung("tool call with valid JSON arguments", tool_call_parses),
    ]
    if model.vision:
        rungs.append(Rung("image input reaches the model", sees_images))
    rungs.append(Rung("~18k-token prefill (the one that matters)", long_prefill))
    return rungs


def run(ctx: SmokeContext) -> bool:
    """Run the ladder, printing progress. True when every applicable rung passed."""
    rungs = ladder(ctx.model)
    for number, rung in enumerate(rungs, start=1):
        print(f"[{number}/{len(rungs)}] {rung.title}")
        try:
            print(f"  {term.GREEN}pass{term.RESET} {rung.run(ctx)}")
        except RungSkipped as skipped:
            print(f"  {term.DIM}skip {skipped}{term.RESET}")
        except RungFailed as failure:
            print(f"  {term.RED}FAIL{term.RESET} {failure}")
            for line in failure.hint.splitlines():
                print(f"       {line}")
            print(f"\n{term.RED}stopped at rung {number}{term.RESET}: "
                  f"do not point a client at {ctx.model.id} yet")
            return False
    print(f"\n{term.GREEN}all rungs passed{term.RESET}: {ctx.model.id} is safe to use")
    return True


# ===========================================================================
# Helpers
# ===========================================================================


def solid_png(rgb: tuple[int, int, int], size: int = 96) -> bytes:
    """A solid-colour PNG built by hand, so the smoke test stays stdlib-only.
    8-bit truecolour (colour type 2), filter byte 0 on every scanline."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    scanlines = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _json(response: Response) -> dict:
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RungFailed(f"response is not JSON: {response.text[:200]!r}") from exc
    return data if isinstance(data, dict) else {}


def _message(response: Response) -> dict:
    try:
        return _json(response)["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RungFailed(f"no choices[0].message in response: {response.text[:200]!r}") from exc


def _reasoning(message: dict, model: Model) -> str:
    field = model.reasoning_field or "reasoning_content"
    return message.get(field) or message.get("reasoning_content") or message.get("reasoning") or ""
