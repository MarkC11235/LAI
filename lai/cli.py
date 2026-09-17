"""`lai`: command-line entry point.

Handlers are thin: they load what they need from `App`, call into the modules
that do the work (checks, render, proxy, smoke), and format the result. Each
handler returns an exit status. Anything that should stop the command with a
message raises `LaiError`.

Adding a command: write `cmd_<name>(app, args) -> int`, then register it in
`build_parser()`. The parser's help text is the command's documentation, and
docs/operations.md is the place for when and why to use it.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from functools import cached_property
from pathlib import Path

from lai import LaiError, checks, paths, registry, render, smoke, term
from lai.host import Host
from lai.proxy import Proxy, ProxyService, parse_running
from lai.registry import Registry
from lai.schema import Model

DEFAULT_RUN_PORT = 8080  # kept free on this machine for foreground debugging


class App:
    """Lazily built collaborators, so `lai ui` never imports the registry and
    `lai check` never opens a socket."""

    def __init__(self, registry_file: Path = paths.REGISTRY_FILE, gen_dir: Path = paths.GEN_DIR) -> None:
        self.registry_file = registry_file
        self.gen_dir = gen_dir

    @cached_property
    def registry(self) -> Registry:
        return registry.load(self.registry_file)

    @cached_property
    def host(self) -> Host:
        return Host(self.registry.settings, self.gen_dir / "llama-env.sh")

    @cached_property
    def proxy(self) -> Proxy:
        return Proxy(self.registry.settings.base_url)

    @cached_property
    def service(self) -> ProxyService:
        return ProxyService(
            self.registry.settings,
            self.proxy,
            config=self.gen_dir / "llama-swap.yaml",
            pidfile=self.gen_dir / "llama-swap.pid",
            logfile=self.gen_dir / "llama-swap.log",
            unit=paths.SYSTEMD_UNIT,
        )

    def require_proxy(self) -> None:
        if not self.proxy.healthy():
            raise LaiError(f"llama-swap is not answering on {self.proxy.base_url}. Start it: lai up")


# ===========================================================================
# Registry and generation
# ===========================================================================


def cmd_ls(app: App, args: argparse.Namespace) -> int:
    """Every registered model: file present, card, state, per-request context."""
    reg = app.registry
    loaded = {item.get("model") for item in parse_running(app.proxy.running())}
    width = max(len(m.id) for m in reg.models) + 2

    header = f"{'ID':<{width}}{'CTX':>8}{'VRAM':>7}  {'CARD':<7}{'FILE':<6}{'STATE':<8}NAME"
    print(f"{term.BOLD}{header}{term.RESET}")
    for model in reg.models:
        present = app.host.find(model.weights) is not None
        if not model.enabled:
            state, colour = "off", term.DIM
        elif model.id in loaded:
            state, colour = "LOADED", term.GREEN
        else:
            state, colour = "idle", term.DIM
        file_cell = _cell("ok" if present else "--", 6, term.GREEN if present else term.RED)
        print(
            f"{model.id:<{width}}{model.slot_context:>8}{model.vram_gb:>6.0f}G  "
            f"{model.device or 'all':<7}{file_cell}{_cell(state, 8, colour)}{model.name}"
        )

    groups = [(card, [m.id for m in reg.pinned_to(card)]) for card in reg.settings.cards]
    populated = [(card, ids) for card, ids in groups if ids]
    print()
    if len(populated) > 1:
        print(term.DIM + "co-resident: " + " + ".join(f"one of {card} [{', '.join(ids)}]"
                                                      for card, ids in populated) + term.RESET)
    print(f"{term.DIM}models on 'all' cards run alone{term.RESET}")
    return 0


def cmd_check(app: App, args: argparse.Namespace) -> int:
    findings = checks.run(app.registry, app.host)
    _print_findings(findings, len(app.registry.active()))
    return 1 if checks.has_errors(findings) else 0


def cmd_gen(app: App, args: argparse.Namespace) -> int:
    findings = checks.run(app.registry, app.host)
    _print_findings(findings, len(app.registry.active()))
    if checks.has_errors(findings) and not args.force:
        raise LaiError("check failed: fix models.py, or re-run with --force")

    artifacts = render.artifacts(app.registry, app.host, app.gen_dir)
    opencode_dest = app.registry.settings.opencode_config
    opencode_text = next(a.content for a in artifacts if a.path.name == "opencode.json")

    if args.dry_run:
        changed = sum(_print_diff(a.path, a.content) for a in artifacts)
        if not args.no_opencode:
            changed += _print_diff(opencode_dest, opencode_text)
        print(f"\n{changed} file(s) would change")
        return 0

    app.gen_dir.mkdir(parents=True, exist_ok=True)
    swap_config_changed = False
    for artifact in artifacts:
        if _write_if_changed(artifact.path, artifact.content, executable=artifact.executable):
            term.info(f"wrote {artifact.path}")
            swap_config_changed |= artifact.path.name != "opencode.json"
    _remove_stale_launchers(app.gen_dir, {a.path for a in artifacts})

    if args.no_opencode:
        term.info(f"not installing {opencode_dest} (--no-opencode)")
    else:
        _install_opencode(opencode_dest, opencode_text)

    if swap_config_changed:
        print()
        term.info("proxy config changed: `lai restart` to apply")
    return 0


# ===========================================================================
# Proxy lifecycle
# ===========================================================================


def cmd_up(app: App, args: argparse.Namespace) -> int:
    app.service.start()
    return cmd_status(app, args) if app.service.uses_systemd() else 0


def cmd_down(app: App, args: argparse.Namespace) -> int:
    app.service.stop()
    return 0


def cmd_restart(app: App, args: argparse.Namespace) -> int:
    app.service.stop()
    time.sleep(1)
    return cmd_up(app, args)


def cmd_status(app: App, args: argparse.Namespace) -> int:
    if not app.proxy.healthy():
        print(f"{term.RED}down{term.RESET} {app.proxy.base_url}")
        return 1
    print(f"{term.GREEN}up{term.RESET}   {app.proxy.base_url}  ({app.service.manager()})")
    return cmd_ps(app, args)


def cmd_ps(app: App, args: argparse.Namespace) -> int:
    response = app.proxy.running()
    if response.status == 404:
        term.warn("/running returned 404: this llama-swap predates that endpoint")
        return 1
    if not response.ok:
        return 1
    items = parse_running(response)
    if not items:
        print(f"{term.DIM}no models loaded{term.RESET}")
    for item in items:
        detail = {k: v for k, v in item.items() if k != "model"}
        print(f"  {term.GREEN}*{term.RESET} {item.get('model')}  {term.DIM}{json.dumps(detail)}{term.RESET}")
    return 0


def cmd_service(app: App, args: argparse.Namespace) -> int:
    """Install and start the systemd --user unit, so the proxy survives logout and reboot."""
    config = app.gen_dir / "llama-swap.yaml"
    if not config.exists():
        raise LaiError("no generated config. Run `lai gen` first.")
    unit = paths.SYSTEMD_UNIT
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(render.systemd_unit(app.registry.settings, app.service.binary(), config))
    term.info(f"wrote {unit}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", "llama-swap"], check=False)
    print()
    term.info("enable lingering so the user manager runs without a login session:")
    print(f"    sudo loginctl enable-linger {os.environ.get('USER', '$USER')}")
    return 0


def cmd_ui(app: App, args: argparse.Namespace) -> int:
    print(f"{app.proxy.base_url}/ui/")
    return 0


def cmd_logs(app: App, args: argparse.Namespace) -> int:
    app.require_proxy()
    if not args.follow:
        print(app.proxy.request("/logs", timeout=10).text)
        return 0
    try:
        for line in app.proxy.stream_logs():
            sys.stdout.write(line)
            sys.stdout.flush()
    except OSError as exc:
        raise LaiError(f"log stream failed: {exc}") from exc
    return 0


# ===========================================================================
# One model
# ===========================================================================


def cmd_load(app: App, args: argparse.Namespace) -> int:
    app.require_proxy()
    model = app.registry.get(args.model)
    term.info(f"loading {model.id} ({model.vram_gb:.0f} GB): a cold load takes a while")
    started = time.monotonic()
    response = app.proxy.chat(model.id, [{"role": "user", "content": "hi"}], max_tokens=1,
                              timeout=app.registry.settings.health_timeout_s)
    if not response.ok:
        raise LaiError(f"HTTP {response.status}: {response.text[:400]}")
    term.info(f"{model.id} resident after {time.monotonic() - started:.0f}s")
    return 0


def cmd_unload(app: App, args: argparse.Namespace) -> int:
    app.require_proxy()
    if args.model:
        app.registry.get(args.model)  # fail on a typo before asking the proxy
    response = app.proxy.unload(args.model)
    if response.status in (200, 204):
        term.info(f"unloaded {args.model or 'all models'}")
        return 0
    if response.status == 404:
        raise LaiError("unload endpoint returned 404: it has moved between llama-swap releases")
    raise LaiError(f"HTTP {response.status}: {response.text[:400]}")


def cmd_run(app: App, args: argparse.Namespace) -> int:
    """Run one model's generated command in the foreground.

    When a backend dies during startup llama-swap can only say "upstream command
    exited prematurely". This is the same command with nothing redirected, so
    the real error reaches the terminal.
    """
    model = app.registry.get(args.model)
    if args.port == app.registry.settings.listen_port:
        raise LaiError(f"port {args.port} is llama-swap's own listen port")
    argv = foreground_argv(app, model, args.port)
    print(f"{term.DIM}{render.shell_join(argv)}{term.RESET}\n", flush=True)
    os.execv(argv[0], argv)
    return 0  # not reached


def foreground_argv(app: App, model: Model, port: int) -> list[str]:
    if model.engine == "vllm":
        launcher = paths.vllm_launcher(app.gen_dir, model.id)
        if not launcher.exists():
            raise LaiError("no generated launcher. Run `lai gen` first.")
        return [str(launcher), str(port)]

    wrapper = app.gen_dir / "llama-env.sh"
    if not os.access(wrapper, os.X_OK):
        raise LaiError(f"{wrapper} is missing or not executable. Run `lai gen` first.")
    weights = app.host.find(model.weights)
    if weights is None:
        raise LaiError(f"no file matches {app.registry.settings.model_dir / model.weights}")
    mmproj = app.host.find(model.mmproj) if model.mmproj else None
    groups = render.llama_server_args(model, app.registry.settings, weights, mmproj)
    return [str(wrapper)] + [token.replace("${PORT}", str(port)) for group in groups for token in group]


def cmd_env(app: App, args: argparse.Namespace) -> int:
    """Run the environment wrapper with --version: the fastest bisect available.

    A version string means the environment is fine and the problem is one model's
    arguments. Anything else means no model can start.
    """
    wrapper = app.gen_dir / "llama-env.sh"
    if not wrapper.exists():
        raise LaiError("no generated wrapper. Run `lai gen` first.")
    print(f"{term.DIM}{wrapper} --version{term.RESET}")
    result = subprocess.run([str(wrapper), "--version"], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    print(output or "(no output at all)")
    # bash prefixes its own errors with the script path, so the path appearing in
    # the output means the wrapper failed rather than llama-server answering.
    if result.returncode != 0 or not output or str(wrapper) in output:
        raise LaiError(f"wrapper exited {result.returncode}: the shared environment is broken, "
                       "not any one model. Nothing will start until this passes.")
    term.info("wrapper ok: oneAPI resolves and llama-server runs")
    return 0


def cmd_smoke(app: App, args: argparse.Namespace) -> int:
    app.require_proxy()
    context = smoke.SmokeContext(
        model=app.registry.get(args.model),
        settings=app.registry.settings,
        proxy=app.proxy,
        prefill_timeout_s=args.timeout,
    )
    return 0 if smoke.run(context) else 1


# ===========================================================================
# Parser and main
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lai",
        description="Local AI control: generate, validate, run and test the inference stack.",
        epilog="Workflow after editing models.py: lai check && lai gen && lai restart && lai smoke <id>",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, handler, help_text: str, aliases: tuple[str, ...] = ()) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text, aliases=list(aliases))
        sub.set_defaults(handler=handler)
        return sub

    add("ls", cmd_ls, "list models: file present, card, loaded state", aliases=("list",))
    add("check", cmd_check, "validate models.py against known failures")

    gen = add("gen", cmd_gen, "generate gen/* and install opencode.json")
    gen.add_argument("--dry-run", action="store_true", help="show a diff of what would change; write nothing")
    gen.add_argument("--no-opencode", action="store_true", help="do not install the opencode config")
    gen.add_argument("--force", action="store_true", help="generate despite check failures")

    add("up", cmd_up, "start llama-swap")
    add("down", cmd_down, "stop llama-swap and every backend")
    add("restart", cmd_restart, "stop, then start llama-swap")
    add("status", cmd_status, "is the proxy up, and what is loaded")
    add("ps", cmd_ps, "models loaded right now")
    add("service", cmd_service, "install the systemd --user unit")
    add("ui", cmd_ui, "print the llama-swap web UI URL")

    logs = add("logs", cmd_logs, "proxy and backend logs")
    logs.add_argument("-f", "--follow", action="store_true", help="stream until interrupted")

    load = add("load", cmd_load, "load a model now instead of on first request")
    load.add_argument("model")

    unload = add("unload", cmd_unload, "free VRAM now (all models when no id is given)")
    unload.add_argument("model", nargs="?")

    run = add("run", cmd_run, "run one model in the foreground, bypassing the proxy (debugging)")
    run.add_argument("model")
    run.add_argument("--port", type=int, default=DEFAULT_RUN_PORT)

    add("env", cmd_env, "test the oneAPI/conda environment wrapper on its own")

    smoke_cmd = add("smoke", cmd_smoke, "acceptance ladder; run before any client uses a model")
    smoke_cmd.add_argument("model")
    smoke_cmd.add_argument("--timeout", type=int, default=300, help="seconds for the long-prefill rung")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `lai ls | head` is not an error
    try:
        return args.handler(App(), args)
    except LaiError as exc:
        term.error(str(exc))
        return 1
    except KeyboardInterrupt:
        return 130


# ===========================================================================
# Helpers
# ===========================================================================


def _cell(text: str, width: int, colour: str) -> str:
    """Pad to the visible width before colouring, so escape codes don't skew columns."""
    return f"{colour}{text:<{width}}{term.RESET if colour else ''}"


def _print_findings(findings: list[checks.Finding], model_count: int) -> None:
    for finding in findings:
        colour = term.RED if finding.severity is checks.Severity.ERROR else term.YELLOW
        print(f"{colour}{finding.severity.value}{term.RESET} {finding}")
    if not checks.has_errors(findings):
        print(f"{term.GREEN}ok{term.RESET} {model_count} enabled models, no blocking problems")


def _print_diff(path: Path, new: str) -> bool:
    old = path.read_text() if path.exists() else ""
    if old == new:
        return False
    diff = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=str(path) if old else "/dev/null", tofile=str(path),
    )
    sys.stdout.writelines(diff)
    return True


def _write_if_changed(path: Path, content: str, *, executable: bool) -> bool:
    changed = not path.exists() or path.read_text() != content
    if changed:
        path.write_text(content)
    if executable:
        path.chmod(0o755)
    return changed


def _remove_stale_launchers(gen_dir: Path, current: set[Path]) -> None:
    for launcher in gen_dir.glob("vllm-*.sh"):
        if launcher not in current:
            launcher.unlink()
            term.info(f"removed {launcher} (model no longer enabled)")


def _install_opencode(dest: Path, content: str) -> None:
    if dest.exists() and dest.read_text() == content:
        term.info(f"{dest} unchanged")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Back up only when something would be lost: hand edits made since the
        # last `lai gen` belong in Settings.opencode_extra instead.
        backup = dest.with_name(f"{dest.name}.bak.{int(time.time())}")
        shutil.copy2(dest, backup)
        term.info(f"backed up the previous config to {backup}")
    dest.write_text(content)
    term.info(f"installed {dest} (restart opencode if a model was added)")


if __name__ == "__main__":
    sys.exit(main())
