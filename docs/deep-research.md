# Deep research

Autonomous multi-source research on the same machine: a question goes in; the
model plans, searches, reads, reflects, searches again, and writes a cited
report. Nothing leaves the machine except the search queries themselves.

This layer is optional. The inference stack works without it, and it changes
nothing in the stack except one generated MCP entry for opencode.

## Architecture

```
Local Deep Research (agent loop)  ───►  SearXNG :8888 (Docker)  ───►  the web
  .venv-ldr, UI on :5000
        │
        └──────────────────────────►  llama-swap :9090  ───►  any registry model
```

| Layer | Component | Why this one |
|---|---|---|
| Search | SearXNG, self-hosted | No quota; aggregates several engines |
| Agent | [Local Deep Research](https://github.com/LearningCircuit/local-deep-research) (LDR) | The only candidate publishing benchmarks on local models |
| Inference | llama-swap | Already there; nothing else had to change |

The layers are wired up and verified separately because **a failed research
run has three possible culprits**. `bin/ldr` checks that llama-swap answers and
SearXNG returns JSON before it starts LDR, so a failure after that points at
LDR or the model.

Three speeds of research are in use: opencode with `searxng_web_search` for
lookups, LDR's quick mode for short questions, and LDR's `langgraph-agent`
strategy for full reports. The dominant latency is usually llama-swap
cold-loading a large model, not search or the agent loop.

## Design choices

**Not LDR's bundled Docker Compose.** It ships Ollama with SearXNG. That would
install a second inference server beside llama-swap, and Ollama has no useful
SYCL path on Arc, so models would silently run on the CPU. Only the search and
agent layers were taken.

**SearXNG rather than Brave, Tavily or Exa.** One `langgraph-agent` run issues
roughly 20–60 searches. Brave's free tier (~2,000 queries a month) is 30–60
reports, and evaluation burns far more queries than research: comparing two
models or a tuning change means rerunning the same question. A quota that runs
out mid-comparison is worse than a slower engine. The trade-offs are real:

- SearXNG is a scraper and engines push back. The failure is gradual and
  silent: results thin out over weeks as an engine starts serving captchas.
  **If reports degrade, test SearXNG directly before blaming the model.**
- It returns raw result snippets, while Tavily and Exa return cleaned, ranked
  content. **If reports are shallow while the model is fine, add a second search
  engine; don't reach for a bigger model.**

**LDR rather than gpt-researcher, local-deep-researcher, or writing one.** LDR
publishes benchmarks with local models on consumer hardware and a community
leaderboard by model, engine and strategy; the alternatives benchmark with
frontier hosted models, which says nothing about whether a small local model
holds a multi-hop chain together. It has a generic `openai_endpoint` provider
(no shim in front of llama-swap), an agentic strategy where the model chooses
its own engines, and an MCP server.

Against it: it is heavy (LangChain, pandas, scikit-learn, SQLCipher, Flask) and
treats an encrypted database as authoritative, the opposite of LAI's generated
configuration. That is why `ldr.env` exists. If the friction becomes
intolerable, LangChain's `local-deep-researcher` is a few hundred readable lines
with no UI or database.

**Don't write one first.** The loop is easy. The hard parts only show up
watching real runs fail: deduplicating sources across queries, capping page size
before it blows the context, stripping navigation boilerplate, detecting a thin
result set and re-querying instead of synthesising from nothing, backing off
when an engine rate-limits. Run LDR long enough to learn which of those matter,
then write a thin replacement against real requirements.

---

## Setup

### SearXNG

Write the configuration rather than patching whatever the image generates on
first run. `use_default_settings` keeps it an override layer, so it stays short
and survives image upgrades.

```bash
mkdir -p ~/projects/lai/searxng && cd ~/projects/lai/searxng
cat > settings.yml <<'EOF'
use_default_settings: true

server:
  secret_key: "PLACEHOLDER"
  # The limiter is bot detection, not rate limiting: it returns 403 to every API
  # client, including LDR. Safe to disable only because the port is loopback.
  limiter: false
  public_instance: false

search:
  # The shipped default is [html]. Declaring the list REPLACES it rather than
  # appending, so html must be repeated or the web UI 404s.
  formats:
    - html
    - json
EOF
sed -i "s/PLACEHOLDER/$(openssl rand -hex 32)/" settings.yml

docker run -d --name searxng --restart unless-stopped \
  -p 127.0.0.1:8888:8080 \
  -v ~/projects/lai/searxng:/etc/searxng \
  searxng/searxng
```

The container writes as its own uid; if files in `searxng/` end up unreadable,
`sudo chown -R "$USER:$USER" ~/projects/lai/searxng`. Verify:

```bash
curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 200   # starts with {"query":
docker exec searxng cat /etc/searxng/settings.yml                           # what it actually loaded
```

The `searxng/` directory contains a secret and is not committed.

### Local Deep Research

In its own virtual environment, deliberately separate from conda and from LAI:

```bash
cd ~/projects/lai
python3 -m venv .venv-ldr
.venv-ldr/bin/pip install local-deep-research
bin/ldr                       # checks llama-swap and SearXNG, then starts the UI on :5000
```

Create an account on first start. **There is no password reset**: the database
is encrypted with a key derived from the password, and backups use the same key.
Keep it in a password manager.

Then set, in the web UI, the settings that environment variables can't control
(they degrade quietly when wrong):

| Setting | Value | Why |
|---|---|---|
| Search strategy | `langgraph-agent` for full research | The agentic loop; other strategies are closer to RAG |
| Search snippets only | **on** | Required with SearXNG |
| Search engine | `searxng` | The old `auto` value was removed upstream |
| Rate-limit profile | `balanced`; `conservative` if captchas appear | |

### Web search inside opencode

Separate from LDR: the `mcp-searxng` server gives opencode a search tool
directly. It is generated from `Settings.opencode_extra` in `models.py`; install
steps are in [setup.md](setup.md#9-opencode).

---

## Configuration

### `ldr.env`

The versioned source of truth for LDR, loaded by `bin/ldr`:

| Variable | Value | Note |
|---|---|---|
| `LDR_LLM_PROVIDER` | `openai_endpoint` | |
| `LDR_LLM_OPENAI_ENDPOINT_URL` | `http://127.0.0.1:9090/v1` | llama-swap |
| `LDR_LLM_OPENAI_ENDPOINT_API_KEY` | `dummy` | Required by the client, ignored by the server |
| `LDR_LLM_MODEL` | a routing key | Exactly as `lai ls` prints it |
| `LDR_SEARCH_TOOL` | `searxng` | |
| `LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL` | `http://127.0.0.1:8888` | |
| `LDR_DATA_DIR` | `<repo>/ldr-data` | Set by `bin/ldr` |

`LDR_LLM_MODEL` lands in each request's `model` field, which is how llama-swap
picks the backend. Renaming a model in `models.py` breaks research runs;
`lai check` warns when `ldr.env` names a model that isn't enabled. Removing the
line lets each run choose a model in the UI instead, at the cost of
reproducibility.

Derive variable names from the key LDR's settings page shows next to each field,
not from its documentation, which has churned: `SEARXNG_INSTANCE` is documented
and ignored by current builds, while
`search.engine.web.searxng.default_params.instance_url` becomes
`LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL`.

### Environment versus database

LDR keeps one SQLCipher database per user under `LDR_DATA_DIR`: settings, API
keys, research history, downloaded documents and their embeddings, metrics.
`LDR_*` variables override it on every settings read, with two consequences:

- **An environment variable silently wins over the UI.** Change a pinned field
  in Settings and the UI accepts it, appears to save, and the variable still
  applies on the next request. Diagnose with `env | grep -i <setting>` in the
  shell that started LDR.
- **An empty value means "not set", not "clear".** A stored value still
  applies; to block something, set a non-empty invalid value.

The working rule: pin in `ldr.env` what must be reproducible; leave out what you
tune live (strategy, iteration depth, engine mix). The database is scratch space.

---

## Choosing a model

LDR's published local results (`langgraph-agent` strategy), as recorded in
August 2026:

| Model | SimpleQA | xbench-DeepSearch |
|---|---|---|
| Qwen3.6-27B (dense) | 95.7% | 77.0% |
| Qwen3.5-9B | 91.2% | 59.0% |
| gpt-oss-20B | 85.4% | – |

How that maps onto this registry:

- **`qwen38-27b-c1-mtp`** (the current `ldr.env` default) is a dense 27B of the
  next generation: the closest analogue to the top row.
- **`qwen36-moe-c1-mtp`** is a 35B MoE with ~3B active parameters per token. Same
  family name as the top row, very different reasoning capacity per token, and
  multi-hop research is exactly where that shows. Don't assume the benchmark
  transfers.
- **`qwen38-vllm`** decodes fastest, but its 32k context is tight for
  synthesising many sources.
- **`qwen38-flash-next-*`** is the largest model, but it occupies both cards and
  host RAM, evicts everything else, and prefills slowly.

Compare models on questions you already know the answer to; your own
two-model comparison beats a leaderboard for this workload.

## A research profile

Research is a different workload from coding: many medium prefills, some of them
concurrent, rather than a few long ones. A profile can be derived from an existing
entry in `models.py` (illustrative; not in the registry and not measured):

```python
from dataclasses import replace

QWEN36_RESEARCH = replace(
    QWEN36_MOE,
    id="qwen36-research",
    name="Qwen3.6-35B-A3B · research",
    context=131072,
    parallel=2,                # LDR summarises pages concurrently; each request gets 65,536
    kv_type="q8_0",            # two long slots of f16 KV is a lot; never q4 (tool-call JSON)
    ubatch=4096, batch=8192,   # prefill-bound workload; +42% prefill measured on this model
    ttl=7200,                  # long gaps between steps; an idle eviction stalls a run
)
```

The inherited `vram_gb=21` does not include a 131k-token KV cache; measure with
`lai run` and set it before relying on the card-size check.

`lai check` will note that `parallel=2` halves the per-request window; that is
intended, and opencode's limit is derived correctly. Each choice is a
hypothesis: `parallel=2` assumes LDR issues concurrent completions (if it is
strictly sequential, `parallel=1` returns the full window for free), and the
ubatch gain was measured on a synthetic prefill, not a research run.

## What to expect

A detailed run reads 20–40 pages at 3–8k tokens each: **100–300k tokens of
prefill** before synthesis. At measured prefill rates that is roughly
40 s–2 min on the Qwen3.6 MoE (~2,500 t/s), 1–3 min on the vLLM 27B
(~1,600 t/s), and much longer on Flash-Next. Generation time on top depends on
the model and its reasoning effort. LDR's own estimates (1–5 min for quick
research, 10–30 min for a report) come from faster hardware.

None of these figures come from an actual research run. On the first run, watch
`lai logs -f`, record the real prefill rate and wall time here, and use them as
the baseline for tuning.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| LDR connects to `localhost:8080` instead of SearXNG | Instance URL variable not applied; LDR fell back to its default | Check `ldr.env`; start with `bin/ldr` |
| SearXNG returns HTML | `json` missing from `search.formats` | Edit `settings.yml`, `docker restart searxng` |
| SearXNG returns 403 | `server.limiter` is on | `limiter: false` |
| A UI setting won't stick | An environment variable overrides it | `env \| grep -i <setting>`; remove it from `ldr.env` |
| Every request fails after a registry change | `LDR_LLM_MODEL` names a renamed model | `lai check` warns; update `ldr.env` |
| Research stalls mid-run | Model unloaded on idle TTL | Raise the model's `ttl` |
| Reports thin out over weeks | SearXNG engines serving captchas | Query SearXNG directly; drop the failing engine, raise delays |
| Poor report, good sources | Model | Try a larger or dense model |
| Poor report, thin sources | Search | Add Brave or Tavily as a second engine |

The last two rows are the important split: **check the sources before blaming
the model.** Retrieval quality dominates model quality in a research agent, and
the instinct is always to reach for a bigger model first.

## Open questions

- Real prefill rate and wall time for each research mode on the current models.
- Whether LDR issues concurrent completions (decides `parallel`).
- Whether downloaded library documents live inside the SQLCipher file or beside
  it, which decides how fast `ldr-data/` grows and whether it needs its own
  backup (`ls -la ldr-data` answers it).
- Encryption is on (the default). If inspectability ever matters more than
  at-rest encryption on an already-encrypted drive,
  `LDR_BOOTSTRAP_ALLOW_UNENCRYPTED=true` uses plain SQLite.

## Next step: research as an opencode tool

LDR ships an MCP server (`pip install "local-deep-research[mcp]"`, binary
`ldr-mcp`) exposing tools such as `quick_research` and `generate_report`. The
earlier blocker, that `lai gen` would overwrite a hand-edited `opencode.json`, is
gone: add an entry beside `searxng` in `Settings.opencode_extra["mcp"]`, with an
absolute path to `.venv-ldr/bin/ldr-mcp` and the `ldr.env` values in its
`environment`, then `lai gen` and `opencode mcp list`. Not yet done or tested.
