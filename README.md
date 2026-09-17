# LAI

Local LLM inference on a dual Intel Arc Pro B70 workstation, run from a single
model registry. One Python file describes every model; `lai` validates it
against failures already measured on this hardware, generates the proxy,
launcher and client configuration from it, and acceptance-tests each model
before an agent is pointed at it.

Two inference engines, llama.cpp (SYCL) and vLLM (XPU, in Docker), sit behind
one OpenAI-compatible endpoint. Models load on first request and swap on demand,
so a client such as [opencode](https://opencode.ai) changes models mid-session
without restarting anything.

## Why it is built this way

**llama-server serves one model per process.** Switching models used to mean
killing the server, which killed the coding agent's session. [llama-swap](https://github.com/mostlygeek/llama-swap)
fixes that by owning process supervision: it listens on one port, reads the
`model` field of each request, and starts or swaps the backend behind it.

**The expensive failures have no error message.** If the client asks for model
A while the server has model B loaded, llama-server ignores the `model` field
and answers anyway: fluent output from the wrong model, nothing in any log. The
same class of silent failure hides in context limits (the client compacts at
the wrong point while the server truncates underneath it), image input (the
client strips images it thinks the model can't see), and quantized KV caches
(tool-call arguments stop being valid JSON). LAI generates the proxy config and
the client config from the same registry so they cannot disagree, and
`lai check` rejects configurations that reproduce a recorded failure.

**Short tests pass on broken configs.** Several configurations here served short
prompts perfectly and hung, crashed, or silently fell back to host memory on a
real 18k-token agent prompt. `lai smoke` runs a fixed acceptance ladder, cheapest
rung first, ending in the long prefill that caught those failures.

## Architecture

```
opencode ─────────┐                     ┌─► llama-server  (SYCL, card 0 / 1 / both)
                  │                     │     via gen/llama-env.sh
Local Deep ───────┼─► llama-swap :9090 ─┤
Research (:5000)  │   one endpoint,     └─► vLLM XPU container
      │           │   swap on demand          via gen/vllm-<id>.sh
      └─► SearXNG :8888 (Docker) ──► the web
                                        ▲
             models.py ─► lai gen ──────┘  also writes ~/.config/opencode/opencode.json
```

Everything binds to `127.0.0.1`. llama-swap has no authentication.

Models pinned to one card can run concurrently (one per card); models that need
both cards run alone. That constraint is generated into llama-swap's matrix
router from each model's `device` field.

## Hardware

| Component | Value |
|---|---|
| GPUs | 2 × Intel Arc Pro B70 (Battlemage, 32 GB each), x8/x8 PCIe lanes, no GPU interconnect |
| CPU / board | AMD Ryzen 9 7900X, MSI MPG X670E Carbon WiFi |
| Memory | 64 GB DDR5-6000 CL30 |
| OS | Ubuntu 26.04, Linux 7.0, in-tree `xe` driver, oneAPI 2026.1 |

## Selected results

All measured on this machine; methodology and caveats are in
[docs/findings.md](docs/findings.md).

| Result | Measurement |
|---|---|
| Qwen3.8-27B GPTQ-Int4 on vLLM XPU, decode at 18k context: no speculative decoding → MTP with 2 draft tokens | 31.1 → **56.1 t/s** (1.8×), 84.7% draft acceptance |
| Qwen3.6-35B-A3B prefill, `-ub 1024` → `-ub 4096 -b 8192` | 1797 → **2546 t/s** (+42%), decode unchanged |
| Tensor split across both cards vs one card (same MoE model) | decode **56.7 vs 84.5 t/s**: all-reduce over PCIe with no interconnect costs a third of decode |
| f16 KV cache vs `q8_0` / `q4_1` | f16 faster in every cell (~3% prefill); 4-bit KV corrupted tool-call JSON |
| mmap with CPU-offloaded weights vs `--load-mode none` | **0.1 vs 10 t/s** |
| vLLM warm start with a persistent `torch.compile` cache | 93 s → **12.7 s** |

## Quick start

On a machine already set up per [docs/setup.md](docs/setup.md):

```bash
git clone <this repo> ~/projects/lai
ln -s ~/projects/lai/bin/lai ~/bin/lai

lai check                  # validate models.py against the host
lai gen                    # write gen/* and install opencode.json
lai service                # llama-swap under systemd --user
sudo loginctl enable-linger "$USER"
lai smoke qwen36-moe-c1-mtp
```

Adding or retuning a model is always the same loop:

```bash
$EDITOR models.py
lai check && lai gen && lai restart
lai smoke <id>             # before any client uses it
```

## Everyday commands

| Command | Does |
|---|---|
| `lai ls` | Models, per-request context, card, whether the weights exist, what is loaded |
| `lai status` / `lai ps` | Proxy health and resident models |
| `lai load <id>` / `lai unload [id]` | Load now / free VRAM now |
| `lai logs -f` | Proxy and backend logs |
| `lai check` | Validate the registry |
| `lai gen [--dry-run]` | Generate configs (`--dry-run` shows a diff and writes nothing) |
| `lai smoke <id>` | Acceptance ladder |
| `lai env` / `lai run <id>` | Bisect a backend that dies at startup |
| `lai up` / `down` / `restart` / `service` / `ui` | Proxy lifecycle |

The full runbook, including troubleshooting, is [docs/operations.md](docs/operations.md).

## Repository layout

```
models.py              the registry: the only file edited to change models
ldr.env                Local Deep Research configuration
bin/lai                CLI launcher (stdlib-only; no install step)
bin/ldr                starts Local Deep Research after checking its dependencies
lai/                   the package; lai/__init__.py has the module map
  schema.py            what a registry entry may say, field by field
  checks.py            validation rules, each tied to a recorded failure
  render.py            registry → generated files (pure functions)
  smoke.py             the acceptance ladder
  templates/           environment wrapper, vLLM launcher, systemd unit
vendor/vllm-xpu-mtp/   patches vLLM needs for MTP on B70 (see its README)
tests/                 runs without GPUs, models, or a proxy
docs/
  setup.md             from BIOS to a working stack
  operations.md        daily use, model workflow, benchmarking, troubleshooting
  findings.md          every measurement and failure mode, with context
  deep-research.md     SearXNG + Local Deep Research layer
  development.md       code structure and how to extend it
gen/                   generated by `lai gen`; never edited, not committed
```

## Development

The runtime uses only the Python standard library. Tests and linting need the
dev extras:

```bash
pip install -e '.[dev]'
pytest && ruff check .
```

Every check rule, the renderer, the YAML emitter, the smoke ladder and the CLI
are covered without hardware: the machine is reached only through `lai/host.py`
and `lai/proxy.py`, which the tests replace. See
[docs/development.md](docs/development.md) before adding a field, rule, or
command.

## Status and limits

This is one workstation's configuration, not a general-purpose server. The
registry, the card sizes, and the check rules encode this hardware; porting it
means editing `models.py` and `Settings`, and treating the rules as a record of
what failed here rather than universal truths. Open questions (a newer
llama.cpp build, larger ubatches on more models, GPTQ-Int4 output quality versus
the GGUF) are listed at the end of [docs/findings.md](docs/findings.md).
