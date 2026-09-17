# Development

How the code is organised and how to extend it. For using the stack, see
[operations.md](operations.md).

## Principles

**The registry is data; the package is the contract.** `models.py` holds only
`Settings` and a list of `Model`s (helpers that build them are fine). Everything
a field means, and which flag or config key it becomes, is documented once, on
the field in `lai/schema.py`.

**Generation is pure.** `lai/render.py` turns a registry into file contents and
never writes. `lai/cli.py` does the writing. The whole output of `lai gen` is
`render.artifacts()`, so tests assert on it directly.

**The machine is behind two seams.** `lai/host.py` (files, globs, the llama.cpp
build number) and `lai/proxy.py` (llama-swap's HTTP API, starting and stopping
it). Checks and rendering take a host; smoke rungs take a proxy. Tests replace
both, so the suite runs with no GPUs, models, oneAPI, or network.

**A blocking rule needs a recorded failure.** `lai check` errors encode things
that went wrong on this hardware. A rule without evidence is an opinion and
should be a warning. The rule's message states the consequence (with the
measured number where there is one), and `docs/findings.md` holds the full
record.

**The runtime is stdlib-only.** No virtual environment is needed to operate the
stack, which matters when the environment is what's broken. That is why
`lai/yamlout.py` exists instead of a PyYAML dependency; PyYAML is used in tests
to prove the emitter's output parses back to the same data.

**Silent failures get loud checks.** The recurring failure mode in this project
is a configuration that runs and is subtly wrong (wrong model answering, context
limits that disagree, images stripped, a flag silently overridden). Prefer
generating both sides of an interface from one value, and checking it, over
documenting that two values must match.

## Code map

```
bin/lai          puts the repo on sys.path, calls lai.cli.main
models.py        SETTINGS and MODELS

lai/
  schema.py      Settings, Model, allowed values. No I/O.
  registry.py    load(models.py) -> Registry; active(), get(), pinned_to()
  host.py        Host: find() weights, exists(), read_text(), llama_build()
  checks.py      Finding, Severity; rule functions; REGISTRY_RULES, COMMON_RULES, ENGINE_RULES
  render.py      artifacts(); llama-server args; llama-swap YAML and matrix router;
                 templates; opencode.json
  yamlout.py     dump(): the subset of YAML the proxy config needs
  templates/     llama-env.sh, vllm-launcher.sh, llama-swap.service (@PLACEHOLDER@ syntax)
  proxy.py       Proxy (HTTP client), ProxyService (systemd or pidfile), parse_running()
  smoke.py       rungs, ladder(), run()
  cli.py         App (lazy registry/host/proxy/service), cmd_* handlers, build_parser()
  term.py        colour and message helpers
  paths.py       repository and install locations

tests/
  conftest.py    FakeHost, make_model(), make_registry()
  test_*.py      one file per module; test_registry.py covers the real models.py
```

Data flow for `lai gen`:

```
models.py ──registry.load──► Registry ──checks.run(registry, host)──► findings
                                 │
                                 └──render.artifacts(registry, host, gen_dir)──► [Artifact]
                                                                                    │
                                                              cli writes changed files, installs opencode.json
```

## Extending

### A new `Model` field

1. Add it to `Model` in `lai/schema.py` with a default and a docstring or comment
   naming what it becomes (flag, llama-swap key, opencode key).
2. Emit it: `llama_server_args()` for a llama-server flag, `swap_model_entry()`
   for llama-swap, `opencode_model_entry()` for opencode.
3. If it is a llama-server flag, add every spelling of the flag to
   `FIELD_OWNED_FLAGS` in `lai/checks.py`, so `extra_args` can't silently
   override it.
4. If it has a closed set of values, add them to `schema.py` and to
   `known_values()`; `Literal` types are not enforced at runtime.
5. Test the rendering in `tests/test_render.py`.

### A new check rule

1. Write a generator function in `lai/checks.py`: `rule(registry, host)` for
   whole-registry invariants, or `rule(model, registry, host)` for per-model
   ones. Yield `error(...)` or `warn(...)`; `run()` attaches the model id.
2. Append it to `REGISTRY_RULES`, `COMMON_RULES`, or the engine's tuple in
   `ENGINE_RULES`. `test_every_rule_is_registered` fails if you forget.
3. Add a test that triggers it and, where it makes sense, one showing a nearby
   valid configuration stays clean.
4. Record the failure behind it in `docs/findings.md`.

Read from the machine only through `host`. If a rule needs something `Host`
doesn't offer, add a method to `Host` and to `FakeHost` in `tests/conftest.py`.

### A new command

1. Write `cmd_<name>(app, args) -> int` in `lai/cli.py`. Use `app.registry`,
   `app.host`, `app.proxy`, `app.service`; they are built lazily. Raise
   `LaiError` for user-facing failures.
2. Register it in `build_parser()` with help text, and add it to
   `test_parser_knows_every_command`.
3. Document it in the command table in `docs/operations.md`.

### A new smoke rung

1. Write `rung(ctx: SmokeContext) -> str` in `lai/smoke.py`. Return the pass
   message; raise `RungFailed(message, hint)` or `RungSkipped(reason)`. Don't
   print.
2. Add it to `ladder()`. Order is cheapest and most basic first; the long prefill
   stays last.
3. Test it against `ScriptedProxy` in `tests/test_smoke.py`, and add a row to the
   ladder table in `docs/operations.md` saying what its failure means.

### A new engine

The vLLM engine is the example to follow:

1. Add the name to `ENGINES` in `schema.py`.
2. Give `backend_command()` a branch. An engine that isn't a native SYCL binary
   should get its own launcher template rather than running through
   `llama-env.sh`.
3. Add engine-specific rules to `ENGINE_RULES`. The llama.cpp rules don't run for
   other engines.
4. Make smoke rungs that depend on llama.cpp endpoints skip for it (see
   `context_matches_client`).
5. Handle `lai run` in `foreground_argv()`.

### Templates

Templates use `@KEY@` placeholders filled by `fill_template()`, which raises on
any placeholder left unfilled. The hard-won comments in them are part of the
documentation; keep them when editing. `test_generated_scripts_pass_shellcheck`
renders the real registry and runs shellcheck over every generated script and
`bin/ldr`.

## Testing and linting

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

CI (`.github/workflows/ci.yml`) runs both on Python 3.11 and 3.13.
`ruff format` is not enforced: several tables (for example `FIELD_OWNED_FLAGS`)
are aligned by hand for reading.

`tests/test_registry.py` pins the set of routing keys. Changing a key breaks
opencode sessions and `ldr.env`, so that test should only change in a commit
that means to.

## Live verification

Tests can't exercise the hardware. After changing rendering or process control,
verify on the machine:

```bash
lai check && lai gen --dry-run     # read the diff
lai gen && lai env && lai restart
lai smoke qwen36-moe-c1-mtp         # fast
lai smoke <the model the change affects>
```
