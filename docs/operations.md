# Operations

Daily use, changing models, benchmarking, and troubleshooting. Setup from
scratch is in [setup.md](setup.md); the evidence behind the rules here is in
[findings.md](findings.md).

## Rules that keep this working

1. **`models.py` is the only place a model is defined.** The proxy and client
   configs are generated from it, so a routing key cannot drift between them.
2. **Never edit `gen/` or the installed `opencode.json`.** The next `lai gen`
   overwrites both. Client settings belong in `Settings.opencode_extra`.
3. **`lai smoke` before a client uses a new or changed model.** Every rung
   except the last has passed on configurations that then failed in real use.
4. **A `lai check` failure is a prior injury.** Each rule is something that
   already cost time on this machine. Override with `lai gen --force` only after
   re-measuring.
5. **Nothing hardcodes a backend port.** llama-swap assigns them.

## Ports

| Port | Service | Notes |
|---|---|---|
| **9090** | llama-swap | The only inference URL any client uses |
| 10001+ | backends | Assigned per start by llama-swap (`${PORT}`) |
| **8888** | SearXNG (Docker) | 8888 rather than 8080, which stays free |
| **5000** | Local Deep Research UI | |
| 8080 | *kept free* | Default port for `lai run` |

All loopback-only. llama-swap has no authentication; do not bind it to
`0.0.0.0` or expose it with a tunnel (see [remote access](#remote-access)).

## After a boot

llama-swap runs under `systemd --user` with lingering, so it is already up.
Models load on first request, not at boot.

```bash
lai status
docker start searxng                               # if it didn't autostart
curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 80
bin/ldr                                            # only when doing research
```

## Commands

| Command | Does | Notes |
|---|---|---|
| `lai ls` | Every registered model: per-request context, VRAM, card, weights present, state | Also prints which models can co-reside |
| `lai status` | Proxy health, how it was started, what is loaded | Exit status 1 when down |
| `lai ps` | Loaded models only | |
| `lai check` | Validate the registry and the host | `FAIL` blocks `gen`; `NOTE` is advisory |
| `lai gen` | Write `gen/*`, install `opencode.json` | Only writes files whose content changed; backs up a differing `opencode.json` |
| `lai gen --dry-run` | Unified diff of what `gen` would change | Review before a restart |
| `lai gen --no-opencode` | Generate without touching the opencode config | |
| `lai up` / `down` / `restart` | Proxy lifecycle | Uses the systemd unit when installed; `down` warns about orphaned `llama-server` processes |
| `lai service` | Install and enable the systemd `--user` unit | Then `sudo loginctl enable-linger $USER` |
| `lai load <id>` | Load a model now | Waits up to `health_timeout_s` |
| `lai unload [id]` | Free VRAM now, instead of after the idle TTL | Always before `llama-bench` |
| `lai logs [-f]` | Proxy and backend output | |
| `lai smoke <id> [--timeout S]` | Acceptance ladder | Raise `--timeout` for CPU-offloaded models |
| `lai env` | Run the environment wrapper with `--version` | Fails → no llama.cpp model can start |
| `lai run <id> [--port P]` | Run one model's generated command in the foreground | Bypasses the proxy; errors reach the terminal |
| `lai ui` | Print the llama-swap web UI URL | |

In opencode, `/models` switches models; llama-swap follows. Restart opencode
only after `lai gen` adds or removes a model.

## Changing models

### The loop

```bash
$EDITOR models.py
lai check && lai gen --dry-run     # read the diff
lai gen && lai restart
lai smoke <id>
```

### Anatomy of an entry

Fields are documented where they are defined, in `lai/schema.py`. A typical
single-card llama.cpp entry:

```python
Model(
    id="qwen36-moe-c1-mtp",          # routing key everywhere; clients store it
    name="Qwen3.6-35B-A3B Q4_K_XL · SYCL1 · MTP",
    weights="Qwen3.6-35B-A3B-MTP/UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    context=32768,                    # -c; opencode is told context // parallel
    max_output=16384,
    vram_gb=21,                       # checked against the pinned card's size
    device="SYCL1",                   # pinned: may co-reside with a SYCL0 model
    reasoning_field="reasoning_content",
    sampler=QWEN_THINKING_SAMPLER,    # server-side defaults, --temp etc.
    request_params=effort("medium"),  # merged into every request by llama-swap
    extra_args=MTP_ARGS,              # one argv token per list item
)
```

Things that are easy to get wrong:

- **`device`** decides co-residency. Pin a model only if it fits on one card
  (SYCL0 has ~1 GiB less, because it drives the display). `None` means the
  model needs both cards and evicts everything else when it loads.
- **Reasoning effort variants are separate entries** that share weights, so
  switching between them reloads the model. One process with several aliases
  (llama-swap `setParamsByID`) failed under the matrix router with
  `no router found for model`.
- **`extra_args` cannot repeat a flag a field controls** (`--mmproj`, `-c`,
  sampler keys, ...). llama.cpp keeps the last occurrence, so the field would be
  silently ignored; `lai check` rejects it.
- **Chat template arguments go in `request_params`**, not
  `--chat-template-kwargs` on the command line.
- **vLLM entries** put the whole container invocation in `launcher` and must use
  `$MODEL_ID` and `$CONTEXT` for `--served-model-name` and `--max-model-len`, so
  the registry and the server cannot disagree.

### Bringing up a model the stack has never run

`lai check` only knows failures that already happened. For a new architecture or
a model that needs CPU offload, find a working configuration by hand first:

1. **Prefer maturity.** An architecture merged into llama.cpp months ago is a
   safer bet than one merged last week, whatever the benchmarks say.
2. **Count active parameters.** MoE models with few active parameters tolerate
   offloading experts to host RAM; dense models pay for every offloaded layer on
   every token.
3. **Size it:** weights + KV cache + compute buffers against ~31 GB per card or
   ~63 GB for both.
4. **First run small**, with `lai run <id>` (or the command it prints): small
   `context`, `load_mode="none"` if anything lives on the CPU.
5. **Read the load log** for `resolve_fused_ops`. If ops are disabled on SYCL,
   stop: the fallback graph may be untested and crash on first decode.
6. **Tune `cpu_moe_layers` down** until it fails to load, then back off by 2.
7. **Walk `context` up**, trading offloaded layers for context.
8. **Distrust a successful load.** SYCL can satisfy an allocation from host
   memory instead of failing, so a config that loads may be running attention
   over PCIe. Compare throughput against smaller contexts.
9. **Measure on the real workload shape** (long prompts), then `lai smoke`.

## The smoke ladder

Rungs run in order and stop at the first failure.

| Rung | Checks | A failure means |
|---|---|---|
| Registered | The id is in `/v1/models` | Registry not regenerated, or the proxy wasn't restarted |
| Loads and completes | Cold load plus a 50-token answer | Backend died: `lai env`, then `lai run <id>` |
| Context matches | Server `n_ctx` per slot equals opencode's `limit.context` (llama.cpp only) | opencode would compact at the wrong point while the server truncates |
| Tool call | `tool_calls` present with arguments that parse as JSON | Missing: template or tool parser mismatch. Unparseable: KV quantization too aggressive |
| Image input (vision models) | Solid red and solid blue images are each named, and not the other colour | Projector missing, or the client stripped the image |
| ~18k-token prefill | Completes within `--timeout`, prints prefill rate | Hang: collect the printed evidence before killing anything |

The last rung is the reason the ladder exists: it crosses every ubatch boundary
and attention budget that short prompts stay under. On a hang, collect
`xpu-smi dump`, `free -h` and a `gdb` backtrace first. A pegged GPU means a stuck
compute kernel; an idle GPU means the host is blocked on a lock or socket.

## Which model

| Model | Use | Notes |
|---|---|---|
| `qwen38-27b-c1-mtp` | Default for opencode and research | Dense 27B, vision, 65k context, one card |
| `qwen38-27b-c*-mtp-xhigh` | Hard problems | Same weights, maximum reasoning effort; slow and verbose |
| `qwen38-27b-c0-*` + `c1-*` | Two agents at once | One per card |
| `qwen38-vllm` | Fastest decode (56 t/s) | 32k context, no vision, holds all of card 0 while loaded |
| `qwen36-moe-c1-mtp` | Fast iteration, plumbing tests | 3B active parameters; loads in ~17 s |
| `qwen38-flash-next-*` | Largest model | 125B-A6B across both cards plus host RAM; evicts everything; 82 GB cold load |

## Benchmarking

Free the cards first, then run `llama-bench` through oneAPI:

```bash
lai unload
source /opt/intel/oneapi/setvars.sh
ZES_ENABLE_SYSMAN=1 ~/llama.cpp/build-new/bin/llama-bench \
  -m ~/models/<path>.gguf -ngl 99 -sm layer -dev SYCL1 \
  -ub 4096 -b 8192 -p 4096 -n 128 -r 3 -o md
```

- **Measure the prompt length you will serve.** `pp512` is one ubatch and says
  nothing about an 18k-token agent prompt; it once made `-sm tensor` look 53%
  faster on a configuration that then hung on every long prompt.
- `-dev SYCL0/SYCL1` (slash) combines both cards in one run; a comma sweeps them
  as separate runs. `-mg` conflicts with `-dev` naming a single device.
- `llama-cli` on current builds rejects `-no-cnv`; use `llama-completion` for
  one-shot runs.
- For vLLM, time a fixed prompt against the backend with a warm prefix, and
  confirm the reported prompt token count before trusting a number: an empty
  prompt variable and cold prefill leaking into the timed request both produced
  bad measurements. llama-swap's UI shows no speed for vLLM backends (it reads
  llama.cpp's `timings` field); use the container's `/metrics`.

Record results with build number and flags in [findings.md](findings.md).

## Rebuilding llama.cpp

Build into a new directory so the working build remains a rollback, and recover
the existing flags rather than guessing them:

```bash
cd ~/llama.cpp
grep -E 'GGML_SYCL|CMAKE_BUILD_TYPE|CMAKE_C(XX)?_COMPILER' build-new/CMakeCache.txt
git pull
conda deactivate 2>/dev/null || true
source /opt/intel/oneapi/setvars.sh
cmake -B build-next <flags from the cache>
cmake --build build-next --config Release -j "$(nproc)" \
  --target llama-server llama-bench llama-cli llama-completion llama-mtmd-cli
```

Then point `Settings.llama_server` at `build-next`, and regression-test:

```bash
lai check                # also re-reads the build number for architecture minimums
lai gen && lai env && lai restart
lai smoke qwen36-moe-c1-mtp
lai smoke <each model you use>
```

Rollback is one line in `models.py`.

## Updating the vLLM image

The launcher pins the image by digest because the MTP patches target that
build. To move to a newer image: pull it, change the digest in `models.py`,
re-check that the patches still apply (the startup log shows a populated
`speculative_config=`), and re-measure acceptance rate and decode speed. The
first start after any engine-config change pays the full `torch.compile`
(~52 s) once.

## Remote access

Keep llama-swap on loopback. To use the stack from a laptop, reach the desktop
over Tailscale (or plain SSH) and run opencode on the desktop inside tmux or
mosh. That keeps the single generated `opencode.json` valid; a second copy on
the laptop would need a different base URL and an MCP block pointing at binaries
that don't exist there. If a client must run on the laptop, forward ports 9090
and 8888 over SSH rather than exposing them. Suspend on the desktop ends
everything.

## Troubleshooting

### Backends

| Symptom | Cause | Fix |
|---|---|---|
| `upstream command exited prematurely` | Backend died before logging anything | `lai env`, then `lai run <id>` |
| `lai env` prints nothing or exits 127 | Wrapper broken: oneAPI path, or `setvars.sh` failing | Fix `Settings.oneapi_setvars`; run the wrapper by hand |
| `libsvml.so: cannot open shared object file` | oneAPI not sourced | Start through `gen/llama-env.sh`; by hand, `source /opt/intel/oneapi/setvars.sh` |
| Works in one terminal, fails in a fresh one | A launcher relied on the parent shell's environment | Always retest launchers from a new terminal |
| Answers fluently but it's the wrong model | Routing key mismatch | `curl -s localhost:9090/v1/models`; keys come from `models.py` only |
| Loads, then segfaults on first decode | Ops disabled on SYCL for this architecture | `grep -i resolve_fused_ops` in the load log; newer build or different model |
| Segfault at load | `-sm row` | `split_mode="layer"` (`lai check` blocks row) |
| Model larger than VRAM hangs at load | `-sm tensor` disables memory auto-fit | `split_mode="layer"` |
| `UR_RESULT_ERROR_OUT_OF_RESOURCES` at warmup | Out of VRAM in the first matrix multiply | Raise `cpu_moe_layers`, or lower `ubatch`/`context` |
| `UR_RESULT_ERROR_DEVICE_LOST` on qwen4exp | `ubatch` above 2048 | `ubatch=2048` (`lai check` enforces it) |
| Throughput ~0.1 t/s with offloaded weights | mmap paging weights from disk | `load_mode="none"` (not for qwen4exp; see findings) |
| Loads at a context that shouldn't fit, then crawls | SYCL satisfied the allocation from host memory | Treat as not fitting; lower `context` |
| Host OOM on load | Weights in host RAM exceed 64 GB | Lower `context`, then `kv_type="q8_0"`, then offload less |
| VRAM still held after `lai down` | Orphaned `llama-server` | `pkill -x llama-server` |
| `lai unload`/`ps` return 404 | llama-swap endpoint moved | Check the installed release's API |

### vLLM backend

| Symptom | Cause | Fix |
|---|---|---|
| Dies in `ze_fd_manager` | `/dev/dri` passed as a device but not bind-mounted | Launcher needs both `--device /dev/dri` and `-v /dev/dri:/dev/dri:ro` |
| Name conflict on start | Previous container survived a SIGKILL | Launcher removes it; by hand, `docker rm -f vllm-qwen38-vllm` |
| Works at half the expected speed | GPTQ integer kernel not selected | Startup log must show `XPUwNa16LinearKernel for AutoGPTQLinearMethod` |
| Tool calls left in `content` | Wrong tool parser | `--tool-call-parser qwen3_coder` (Qwen3.8 emits XML calls) |
| Reasoning-only turns render blank in opencode | Wrong reasoning field | `reasoning_field="reasoning"` for vLLM |
| `speculative_config=None` in the log | MTP patches not applied | Check `vendor/vllm-xpu-mtp/` and `B70_MTP_BF16_DRAFT=1` |
| Every start takes ~90 s | No compile cache mount | `~/.cache/vllm-xpu` mounted at `/cache` with `VLLM_CACHE_ROOT=/cache` |

### Clients

| Symptom | Cause | Fix |
|---|---|---|
| opencode compacts too early, or the server truncates | `limit.context` ≠ server `n_ctx` per slot | Generated as `context // parallel`; `lai smoke` rung 3 verifies |
| Tool calls appear as text in `reasoning_content` | Chat template not parsing | `--jinja` (always generated); check build age |
| `tool_calls` present, arguments not JSON | 4-bit KV cache | `kv_type="q8_0"` or f16 (`lai check` blocks q4) |
| Blank responses, server looks idle | Output only in the reasoning field | Set `reasoning_field`; check raw JSON with curl |
| Model says it cannot see images | opencode stripped the image | Registry `vision=True` generates the undocumented `modalities` block |
| Requests go to `undefined/chat/completions` | `baseURL` outside `options` | Generated correctly; don't hand-edit |
| Everything queues, web UI unresponsive | One wedged request holds the only slot (`parallel=1`) | Unload the model; find the stuck client |
| Session titles slow the first reply | `small_model` shares the one slot with the real request | Point `small_model` at a model on the other card |
| MCP tools missing | `mcp-searxng` not at the path in `models.py`, or opencode not restarted | `opencode mcp list`; `opencode mcp debug searxng` |

Search and research failures are covered in
[deep-research.md](deep-research.md#failure-modes).

### Environment

| Symptom | Cause | Fix |
|---|---|---|
| Build fails inexplicably | conda shadowing system libraries | Build from a shell with `conda deactivate` |
| `clinfo` empty, `sudo clinfo` works | Render group not in the session | Log out and back in |
| Nothing printed after `newgrp` | The subshell swallowed the pasted line | Run commands separately |
| `check_tensor_dims: tensor 'token_embd.weight' not found` | A vocab test fixture, not a model | Download real weights |
| Log shows no progress | `| tee` block-buffers | Use llama-server's `--log-file` |
| Memory slower than rated | XMP reset to 4800 MT/s after a CMOS clear | Re-enable XMP; check `dmidecode -t 17` |

## Where things live

```
~/projects/lai/                      this repository
  gen/llama-swap.yaml                proxy config (generated)
  gen/llama-env.sh                   environment wrapper for every llama-server (generated)
  gen/vllm-<id>.sh                   container launchers (generated)
  gen/opencode.json                  copy of what was installed (generated)
  gen/llama-swap.{pid,log}           only when started without systemd
  ldr-data/  .venv-ldr/  searxng/    research layer state (not committed)
~/.config/opencode/opencode.json     installed by `lai gen`
~/.config/opencode/AGENTS.md         personal model instructions (not managed)
~/.config/systemd/user/llama-swap.service   installed by `lai service`
~/.local/state/llama-swap/activity.db       llama-swap activity store
~/.cache/vllm-xpu/                   vLLM compile cache
~/models/                            weights
~/llama.cpp/build-new/bin/           llama-server, llama-bench, llama-cli
~/bin/llama-swap, ~/bin/lai          binaries on PATH
```
