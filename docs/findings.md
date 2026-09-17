# Findings

Measurements and failure modes from running this stack, with enough context to
judge whether each still applies. This is the evidence behind the rules in
`lai check`, the defaults in `models.py`, and the troubleshooting tables in
[operations.md](operations.md).

**How to read it.** Each finding gives the date and llama.cpp build where
known. Results on SYCL change quickly between builds, and several findings here
were reversed by later ones; superseded conclusions are kept, marked, because
the reason they were wrong is usually the useful part. Hardware for everything
below: 2 × Arc Pro B70 at PCIe x8/x8 with no GPU interconnect, Ryzen 9 7900X,
64 GB DDR5, Ubuntu 26.04, oneAPI 2026.1.

**Adding a finding.** Record what was run (model, quantization, build, flags,
prompt length), the numbers, and what it changed. If it becomes a `lai check`
rule, say so here and quote the number in the rule's message.

---

## Split mode

**Current rule: `split_mode="layer"`; pin to one card when the model fits.**
`lai check` blocks `tensor` and `row`.

The conclusion changed twice.

| When | Model and setup | Result | Conclusion at the time |
|---|---|---|---|
| Aug 2026 | gpt-oss-120b, `-ncmoe 6`, `-ub 512 -b 2048`, `llama-bench` pp512 / tg128 | layer 201.9 ± 54.5 / 31.5; **tensor 309.0 ± 7.6 / 32.8** | "Use `-sm tensor`, always": +53% prefill, and layer split was erratic (27% std dev) |
| 2026-08-11 | Same model in the server, 18,000-token prompt | **Tensor hangs indefinitely** (>5 min; once >4 hours). 30-token prompts with tools finished in 2.6 s | Tensor for short prompts only; layer for agents |
| Sept 2026, b10729 | Qwen3.6-35B-A3B Q4_K_XL, `-b 8192 -ub 4096` | one card 2545 / **84.5**; layer across both 2494 / 84.1; tensor 1900 / **56.7** (pp4096 / tg, t/s) | Tensor no longer hangs, but costs a third of decode. Layer across both cards is ~2% slower than one card |

Why the first benchmark misled: a 512-token prompt at `-ub 512` is **one
ubatch**, a single pass with no repeated cross-GPU synchronisation. An
18,000-token prompt is 36 ubatches, each with an all-reduce between the cards.
The benchmark measured a code path agent workloads never use.

Evidence collected during the August hang: host memory fine (45 GiB available,
no swap); backtrace top frame `__GI_sched_yield()` (host busy-polling a sync
primitive); server log showed `launch_slot_` but never `prompt eval time`. A
livelock in tensor-split synchronisation, not a crash or OOM. llama.cpp's own
multi-GPU documentation calls tensor split experimental and CUDA-oriented.

Why tensor is slow here even when it works: it all-reduces activations every
layer, and the only path between the cards is PCIe through the CPU, with no
NCCL equivalent on SYCL. Decode is bandwidth-bound on exactly that path.

Tensor split also **disables memory auto-fit** (every launch logs
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`), so there is no
guard against overcommitting VRAM; a model larger than VRAM (Flash-Next) hangs at
load under tensor split.

**Row split** (`-sm row`) segfaults at model load on dual-B70 SYCL.

Not measured: tensor split for a model that needs both cards but fits entirely
in VRAM, and any of this on a platform with x16 links.

---

## Batch size

**Measured on Qwen3.6-35B-A3B Q4_K_XL, one card, b10729 (Sept 2026).**

| Flags | pp4096 | Decode |
|---|---|---|
| `-ub 1024` | 1797 t/s | ~84 t/s |
| `-ub 4096 -b 8192` | **2546 t/s** (+42%) | ~84 t/s |

Larger ubatches help prefill substantially and do nothing for decode, at the
cost of larger compute buffers. The registry still uses the schema default
(`-ub 2048 -b 4096`) for every llama.cpp model, and qwen4exp cannot exceed 2048
at all ([Flash-Next](#qwen38-flash-next-125b-a6b)); applying 4096 to single-card
entries is an open item.

---

## KV cache type

**f16 (the default, `kv_type=None`) is fastest when it fits.** On the same
Qwen3.6 MoE at b10729, f16 beat `q8_0` and `q4_1` in every cell, by ~3% on
prefill and ~2% on decode. Quantized KV is only worth it for context headroom.

**4-bit KV corrupts tool calls.** With `q4_0`, the model still produced
`tool_calls`, but the argument strings were not valid JSON, which defeats an
agent model entirely. `lai check` blocks `q4_0` and `q4_1`; `q8_0` is the safe
quantization. qwen4exp needs f16 regardless.

---

## Loading and host memory

### mmap with offloaded weights

DeepSeek V4-Flash 2-bit (91 GB, weights larger than RAM), same flags, only the
load mode changed: `--load-mode mmap` **0.1 t/s**, `--load-mode none` **10 t/s**.
With CPU-resident weights mapped from disk, every token pages them back in. The
loader warns (`tensor overrides to CPU are used with mmap enabled`); take it
seriously. `lai check` warns when `cpu_moe_layers` is set without
`load_mode="none"`.

The exception is a tensor that is too large to pin: Flash-Next's 29 GB n-gram
table must stay memory-mapped, because `load_mode="none"` makes it non-pageable
and exceeds 64 GB of host RAM.

`-ngl 0` combined with `--load-mode none` fails with `insufficient memory`:
that combination needs every weight in RAM.

### A successful load does not prove a fit

gpt-oss-120b's offload requirement measured linear with context: 32k needed
`-ncmoe 6`, 64k needed 12. Running 131,072 at `-ncmoe 6` did **not** fail: SYCL
satisfied the allocation from host memory, the server loaded, reported the full
context, accepted work, and ran attention over PCIe. On SYCL, an over-committed
config can look like a working one. Compare throughput against a smaller context
before trusting a new maximum.

### Out-of-memory in disguise

`UR_RESULT_ERROR_OUT_OF_RESOURCES` during the warmup matrix multiply is VRAM
exhaustion (seen with gpt-oss at `-ncmoe 0`).

### No cgroup memory cap on the proxy

A `MemoryMax=40G` / `MemorySwapMax=0` drop-in was added to the llama-swap unit
after a runaway backend OOM-killed the desktop session. It was removed: the cap
starved models that legitimately keep weights in host RAM. The generated unit
documents the decision.

---

## SYCL backend behaviour

**Check for disabled fused ops first.** New architectures land CUDA-first, and
SYCL coverage can lag enough that ops are *disabled*, not just slow, leaving
fallback graphs nobody has run on SYCL. DeepSeek V4 loaded and served, then
segfaulted on first decode with four ops disabled. `grep -i resolve_fused_ops`
on the load log before anything else. Corollary: an architecture merged months
ago is safer than one merged last week.

**Gated-DeltaNet on older builds.** Qwen3.5 (GDN layers) hung at slot init
unless flash attention was off, and went unstable after a few responses even
then (llama.cpp #20423). A Qwen3.8-27B canary (`qwen3_5` architecture) on a fresh
build in mid-August ran cleanly with no CPU fallback, so the current Qwen3.8
entries run with flash attention on.

**The display costs ~1 GiB on SYCL0.** `--list-devices` shows 31,521 vs
32,602 MiB free. `Settings.cards` records 31 and 32 GiB usable.

**MoE tolerates offload; dense does not.** Active parameters drive the offload
penalty: gpt-oss-120b (5.1B active) lost little from six expert layers in host
RAM, while DeepSeek V4 (13B active) paid for every offloaded layer. This is
about models that don't fit in VRAM; a dense model that fits one card (the 27B)
is a different trade-off.

**Quantization sensitivity is model-specific.** gpt-oss-120b's expert tensors
are natively MXFP4 at every quantization rung (the whole ladder spans
62.6–64.4 GB), so a lower rung saves almost nothing. DeepSeek V4 at 2-bit
measured ~78% top-token agreement with the reference, too low for agentic work.

**Renamed flags.**

| Old | Current |
|---|---|
| `--no-mmap` / `--mmap` | `--load-mode none` / `--load-mode mmap` (also `mlock`, `mmap+mlock`, `dio`) |
| `--reasoning none` | `--reasoning off` |
| `llama-cli -no-cnv` for one-shot runs | `llama-completion` |

**Tool quirks.** In `llama-bench`, `-dev SYCL0/SYCL1` combines devices in one
run and a comma sweeps them separately; `-mg` conflicts with `-dev` naming one
device. The `models/ggml-vocab-*.gguf` files in the llama.cpp repository are
tokenizer fixtures (`tensor 'token_embd.weight' not found`).

**MKL warning.** oneAPI 2026.1 prints `Incompatible OpenCL driver version. GPU
performance may be reduced.` against the PPA compute runtime. Untested whether a
different runtime changes throughput; oneMKL GEMM is on the prefill path, so
that is where it would show.

---

## Environment and process pitfalls

**conda** auto-activates its base environment and shadows the system
`libstdc++`/`libsycl`. It has broken builds and backend starts. llama-swap
inherits the environment of whatever started it, so `gen/llama-env.sh` strips
conda paths unconditionally and deliberately avoids a login shell (which would
re-activate conda).

**The first wrapper failed silently.** It ran `setvars.sh` under `set -u` with
stderr discarded. Intel's script references unset variables, so the shell
exited 127 before llama-server started, and llama-swap could only report
`upstream command exited prematurely`. The template now omits `set -u` around
`setvars.sh` and keeps stderr.

**`source` glued to a comment.** An old launcher contained
`# ...tools call.source /opt/intel/oneapi/setvars.sh`. `source` never ran; the
launcher worked only because the parent shell was already sourced. Retest any
launcher change from a fresh terminal.

**`| tee` block-buffers** llama-server's output (4–8 KB), so progress lines sit
in the buffer and a quiet log is not evidence of a hang. Use `--log-file`.

**One slot, one wedged request.** With `--parallel 1`, a stuck request holds
the only slot and every later request queues behind it, including the built-in
web UI and opencode's session-title request. "The UI stopped working" is a
symptom of the wedged slot, not a second failure.

**Memory speed.** The DDR5 kit ran at 4800 MT/s for a long time because it has only
an XMP profile, which the AMD board doesn't apply by default.

---

## Models

### Qwen3.8-27B on vLLM XPU (`qwen38-vllm`)

**2026-09-13.** SergiioB's GPTQ-Int4 checkpoint with the `mtp.*` head kept at
BF16 (rev `9d189a60`), image `vllm/vllm-openai-xpu@sha256:f01e24f6…`, single card
via `ZE_AFFINITY_MASK=0`, `--kv-cache-dtype fp8`. A GGUF can't be used for MTP:
llama.cpp's converter strips the head.

Speculative decoding sweep, 18k-token context, 512 generated tokens:

| Draft tokens | Decode | Acceptance |
|---|---|---|
| none | 31.1 t/s | – |
| 1 | 47.3 t/s | 89.7% |
| **2** | **56.1 t/s** | **84.7%** |
| 4 | ~51 t/s, unstable | 54.4% |

MTP2 was chosen. A single MTP layer run repeatedly loses accuracy with depth,
as vLLM's startup warning says, and at 4 drafts nearly half the work is thrown
away. The cookbook this setup follows reports 93–96% acceptance at 4 drafts;
the gap is unexplained (candidates: its sampling presets versus temperature 0
here, and its 230 W power cap).

Other measurements: prefill ~1,627 tok/s; cold time-to-first-token ~11 s at 18k;
no-speculation decode 33.0 t/s at a 560-token context. Start time ~93 s without
a persistent compile cache, of which ~52 s is `torch.compile`; **12.7 s** with
`VLLM_CACHE_ROOT` mounted. The cache keys on engine configuration, so changing
context or batch settings costs one slow start. For comparison, the same model
as a GGUF on llama.cpp decoded ~21.8 t/s in an August canary (different
quantization and build, no speculative decoding; not a controlled comparison).

Configuration facts, each found by failing without it:

- **MTP needs two patches** to the nightly image plus `B70_MTP_BF16_DRAFT=1`,
  and `--gpu-memory-utilization 0.88` instead of 0.90 to fit the draft buffers.
  The patches are vendored in `vendor/vllm-xpu-mtp/`. Confirm
  `speculative_config=` is populated in the startup log; if it says `None`, any
  measured speedup is something else.
- **`--device /dev/dri` alone is not enough.** oneCCL opens the device by path
  and dies in `ze_fd_manager` without `-v /dev/dri:/dev/dri:ro`, even at tensor
  parallel size 1.
- **Confirm `XPUwNa16LinearKernel for AutoGPTQLinearMethod`** in the startup
  log. Without the integer XMX path the server works at half speed with no error.
- **Tool calls are XML** (`<function=…>`), so the parser is `qwen3_coder`; the
  `hermes` parser leaves calls in `content` with `tool_calls` absent.
- **Reasoning parser `qwen3`**, and vLLM names the response field `reasoning`,
  not `reasoning_content`.
- **vLLM claims `gpu-memory-utilization` of the card at load and holds it**, so
  its `vram_gb` in the registry is the claim, not the 16.6 GB of weights. It is
  pinned to SYCL0 so the SYCL1 models stay co-residable.
- **Sampling rides in the request** (`request_params`), since there is no
  server flag to map it to.

Not attempted: tensor parallelism across both cards (an open vLLM issue covers
dual B70). **Unverified: GPTQ-Int4 output quality versus the GGUF.**

### Qwen3.8-27B GGUF (`qwen38-27b-*`)

UD-Q5_K_XL, one card, 65k context, vision. Use Unsloth `UD-*` quantizations:
the plain ones shipped a chat template that throws `System message must be at
the beginning`. The GGUF keeps the nextn layers, so llama.cpp's
`--spec-type draft-mtp` works; draft length 2 (reference tests found longer
drafts slower). Decode with MTP on llama.cpp has not been recorded here, and
MTP combined with image input is untested.

Registry bugs found in the rewrite: the entries set `mmproj` to a BF16 file that
wasn't downloaded while `extra` passed `--mmproj` for the F16 file, and llama.cpp
silently used the last one. `lai check` now rejects any `extra_args` flag that a
field controls. Sampler keys also mixed `top-p` and `top_p` spellings; the
generator now normalises them.

### Qwen3.8 Flash-Next 125B-A6B

Architecture `qwen4exp`, **requires b10664+**. UD-IQ3_XXS in three shards,
76.3 GiB (shard 1 is ~11 MB; that is normal for this split). About 52 GB of
weights on the two cards and a 29 GB per-layer n-gram embedding table in host
RAM via `-ot per_layer_token_embd\.weight=CPU`. Combining that override with
`--n-cpu-moe` wastes ~11 GB of VRAM, because `--n-cpu-moe` already places the
tensor. Vision works on b10729 with Unsloth's `mmproj-F16.gguf`.

Structure: three of every four layers are gated delta-net; the fourth is Qwen
Sparse Attention with a 2048-token budget. Below that budget sparse attention is
dense by construction, so **short prompts pass whether or not the sparse path
works**; only the long-prefill smoke rung exercises it.

Hard constraints (all enforced by `lai check`):

| Constraint | Evidence |
|---|---|
| `ubatch ≤ 2048` | 2048 runs; 2049 aborts with `UR_RESULT_ERROR_DEVICE_LOST`. `-fa 0` only moves the abort from `FLASH_ATTN_EXT` to `MUL_MAT` |
| f16 KV only | Quantized KV corrupts or crashes the sparse-attention path (separate from the q4 tool-call bug) |
| stay memory-mapped | `load_mode="none"` pins the n-gram table and exceeds 64 GB of RAM |
| layer split | tensor split disables auto-fit; the model exceeds VRAM and hangs at load |

Also: `--slot-save-path` is accepted and never reused, so every request silently
re-prefills. Its default reasoning effort ran away on the first test: 32,389
tokens over 28 minutes, reaching the context limit with no visible output. Every
entry now sets effort per request. MTP is not available: the converter drops the
MTP tensors for this architecture (llama.cpp PR #27836 was still a draft).

Performance: decode ~25 t/s at low depth and ~19.5 t/s at 32k; 139 t/s prefill
on a 379-token prompt early on. A later `llama-bench` on b10729 at
`-sm layer -b 8192 -ub 4096` recorded pp4096 ~431 t/s and tg64 ~25.8 t/s, which
conflicts with the server aborting above `-ub 2048`. **Not reconciled**;
re-measure pp4096 at `-ub 2048` before relying on either prefill figure.

Context headroom: only 12 of 48 layers carry a KV cache (the rest hold
fixed-size recurrent state), at ~24 KB per token in f16, so 131,072 tokens need
~3.1 GB of KV and should fit; compute buffers at `-ub 2048` are the real limit.
262,144 is architecturally blocked (a grid-dimension overflow in `rms_norm` at
exactly that size; the practical ceiling is 261,888). The registry uses 65,536;
131,072 is untested.

### Qwen3.6-35B-A3B (`qwen36-moe-c1-mtp`)

The batch-size, KV-type and split-mode measurements above. Cold load ~16.6 s,
which makes it the fastest model for plumbing tests.

### Retired or parked configurations

**gpt-oss-120b** (117B, 5.1B active, UD-Q8_K_XL, 60 GiB). 31–33 t/s decode at
32k with `-ncmoe 6`, 28 t/s at 64k with `-ncmoe 12`; needs `--jinja` and
`--temp 1.0 --top-p 1.0 --top-k 0`. It puts every word in `reasoning_content`
with `content` empty, so a turn without a tool call renders blank in a client
that reads only `content`. Unusable with opencode under tensor split (the
long-prefill hang); whether layer split fixes it was never tested. No longer
used.

**DeepSeek V4-Flash** (284B, 13B active, 2-bit, 91 GB). ~10 t/s. Needed PRs
#26515 and #26568 at the time (segfault on first decode otherwise); later
mainline builds (b10448) load it unpatched. `--load-mode none` and `-ncmoe 30`
were hard floors; the 3-bit build (103 GB) doesn't fit. Quality too low for
agentic work.

**Qwen3-Coder-Next 80B-A3B Q4** was the first working opencode model (August),
before the Qwen3.8 models. **Qwen3.5-122B** was parked over the Gated-DeltaNet
gap, which may no longer apply.

---

## opencode

- **`baseURL` must be inside `options`.** At the provider root it is silently
  ignored and requests go to `undefined/chat/completions`. `apiKey` is nominally
  optional, but the AI SDK can throw without one.
- **`limit.context` must equal what one request gets** (`n_ctx` per slot, which
  is `-c` divided by `--parallel`), because opencode compacts based on it.
- **The model key must equal the server's alias.** Without `-a`, llama.cpp
  reports the GGUF path as the model id.
- **Image input needs `modalities`.** opencode 1.18.27 silently strips image
  parts for custom OpenAI-compatible providers unless the model declares
  `modalities: {input: [text, image], output: [text]}`. Undocumented;
  `attachment: true` alone is not enough.
- **Reasoning models need `reasoning: true` and `interleaved.field`**, or
  reasoning-only turns render blank.
- **`small_model`** defaults to a hosted model for session titles; pointing it at
  a local model keeps everything on the machine.
- **MCP:** opencode names tools `{server}_{tool}`; match them with a wildcard
  such as `searxng*` rather than exact names. The `mcp-searxng` entry point is
  `dist/cli.js` (not `index.js`) as of 1.16, and runs via absolute paths so it
  doesn't depend on PATH or conda. Two URL readers (built-in `webfetch` and
  `web_url_read`) made a local model pick the wrong one; `webfetch` is disabled
  because `web_url_read` has length and section controls. Local models don't
  search unprompted, so `~/.config/opencode/AGENTS.md` tells them when to.
- **Permissions:** a catch-all `"*": "allow"` overrides more specific `edit` deny
  rules, so keep explicit rules alongside it if the write boundary matters. The
  clarifying-question tool is a separate permission (`"question": "deny"`).
  Neither is currently in `Settings.opencode_extra`.

---

## Methodology lessons

- **A benchmark measures the configuration you benchmarked.** Reproduce the
  workload's shape, especially prompt length, before trusting a tuned config.
- **A successful load does not prove a fit** on SYCL.
- **Short tests pass on broken long paths** (tensor-split synchronisation,
  sparse attention). Every acceptance test ends with a long prefill.
- **Absence of log output is not evidence of a hang** when output is piped.
- **Check the sources before blaming the model** in research workloads
  ([deep-research.md](deep-research.md)).

---

## Open questions

| Question | Why it matters |
|---|---|
| Newer llama.cpp (around b10819: sparse flash attention, peer-to-peer copy, oneMKL flash-attention refactor) | Rewrites the code paths behind several findings; re-measure before tuning environment variables |
| `-ub 4096 -b 8192` on the other single-card entries | +42% prefill on the MoE; costs compute-buffer VRAM |
| Flash-Next prefill at `-ub 2048`, and 131,072 context | Resolves the conflicting figures; doubles usable context |
| `GGML_SYCL_USM_SYSTEM=1` for Flash-Next's host spill | Needs `CONFIG_DRM_XE_GPUSVM` in the kernel |
| MTP decode on llama.cpp for the 27B GGUF, and MTP with images | Neither recorded |
| GPTQ-Int4 output quality versus the GGUF | Unverified |
| vLLM acceptance at 4 drafts versus the cookbook's 93–96% | Unexplained gap |
| Tensor split for a model that needs both cards but fits in VRAM | Never measured |
| The oneAPI/OpenCL version warning | Possible prefill cost |
