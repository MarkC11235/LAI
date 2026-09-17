# Instructions for coding agents

LAI generates and validates a local LLM serving stack from `models.py`. Read
`README.md`, then `docs/development.md` before changing code.

## Invariants

- **Never edit `gen/` or `~/.config/opencode/opencode.json`.** Both are generated.
  Change `models.py` or the code in `lai/`, then regenerate.
- **The runtime is stdlib-only.** No third-party imports in `lai/` or `bin/`.
  Test and lint tools belong in the `dev` extras.
- **`lai/render.py` stays pure**: no file writes, no network. Machine access goes
  through `lai/host.py` or `lai/proxy.py`.
- **Routing keys (`Model.id`) are an interface.** Don't rename them unless asked;
  `tests/test_registry.py` pins them.
- **A `lai check` ERROR needs a recorded failure** in `docs/findings.md`.
  Unproven concerns are warnings.
- **Values in `models.py` came from measurements.** Don't "optimise" batch sizes,
  split modes, KV types, or load modes without new measurements; the reasons are
  in comments and in `docs/findings.md`.

## Before finishing a change

```bash
pytest && ruff check .
```

Update the docs a change affects: field docs in `lai/schema.py`, commands in
`docs/operations.md`, measurements in `docs/findings.md`.

## Commands with side effects

These act on the live machine: `lai gen` (installs the opencode config),
`lai up|down|restart|service`, `lai load|unload`, `lai run`, `lai smoke`
(loads models), and anything touching Docker. Run them only when asked.
`lai check`, `lai ls`, `lai gen --dry-run` and the test suite are safe.
