"""Command wiring against a temporary registry and gen directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from lai import LaiError, cli

REGISTRY = '''
from pathlib import Path
from lai.schema import Model, Settings

ROOT = Path({root!r})
SETTINGS = Settings(
    llama_server=ROOT / "bin/llama-server",
    oneapi_setvars=ROOT / "setvars.sh",
    model_dir=ROOT / "models",
    opencode_config=ROOT / "config/opencode.json",
    default_model="a",
    small_model="a",
)
MODELS = [
    Model(id="a", name="A", weights="a.gguf", context=65536, vram_gb=20, device="SYCL1"),
    Model(id="v", name="V", weights="v/config.json", context=32768, vram_gb=28, device="SYCL0",
          engine="vllm", launcher='vllm serve --max-model-len "$CONTEXT" --served-model-name "$MODEL_ID"'),
]
'''


@pytest.fixture
def app(tmp_path: Path) -> cli.App:
    for relative in ("bin/llama-server", "setvars.sh", "models/a.gguf", "models/v/config.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    (tmp_path / "models.py").write_text(REGISTRY.format(root=str(tmp_path)))
    return cli.App(registry_file=tmp_path / "models.py", gen_dir=tmp_path / "gen")


def gen_args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"dry_run": False, "no_opencode": False, "force": False, **overrides})


def test_gen_writes_artifacts_and_installs_opencode(app, capsys):
    assert cli.cmd_gen(app, gen_args()) == 0
    assert sorted(p.name for p in app.gen_dir.iterdir()) == [
        "llama-env.sh", "llama-swap.yaml", "opencode.json", "vllm-v.sh"]
    assert os.access(app.gen_dir / "vllm-v.sh", os.X_OK)
    installed = app.registry.settings.opencode_config
    assert installed.read_text() == (app.gen_dir / "opencode.json").read_text()
    assert "lai restart" in capsys.readouterr().out


def test_second_gen_changes_nothing_and_makes_no_backup(app, capsys):
    cli.cmd_gen(app, gen_args())
    capsys.readouterr()
    cli.cmd_gen(app, gen_args())
    out = capsys.readouterr().out
    assert "wrote" not in out and "lai restart" not in out and "unchanged" in out
    assert not list(app.registry.settings.opencode_config.parent.glob("*.bak.*"))


def test_hand_edited_opencode_config_is_backed_up(app):
    cli.cmd_gen(app, gen_args())
    installed = app.registry.settings.opencode_config
    installed.write_text("{}\n")
    cli.cmd_gen(app, gen_args())
    backups = list(installed.parent.glob("opencode.json.bak.*"))
    assert len(backups) == 1 and backups[0].read_text() == "{}\n"


def test_dry_run_writes_nothing(app, capsys):
    assert cli.cmd_gen(app, gen_args(dry_run=True)) == 0
    assert not app.gen_dir.exists()
    assert "+++ " in capsys.readouterr().out


def test_gen_refuses_on_check_errors_unless_forced(app):
    (app.registry.settings.model_dir / "a.gguf").unlink()
    with pytest.raises(LaiError, match="check failed"):
        cli.cmd_gen(app, gen_args())
    assert cli.cmd_gen(app, gen_args(force=True)) == 0


def test_stale_vllm_launchers_are_removed(app):
    app.gen_dir.mkdir()
    stale = app.gen_dir / "vllm-removed-model.sh"
    stale.write_text("")
    cli.cmd_gen(app, gen_args(no_opencode=True))
    assert not stale.exists()


def test_foreground_argv_substitutes_the_port(app):
    cli.cmd_gen(app, gen_args(no_opencode=True))
    argv = cli.foreground_argv(app, app.registry.get("a"), 8080)
    assert argv[0].endswith("gen/llama-env.sh")
    assert argv[argv.index("--port") + 1] == "8080"
    assert cli.foreground_argv(app, app.registry.get("v"), 8081)[1] == "8081"


def test_run_refuses_the_proxy_port(app):
    args = argparse.Namespace(model="a", port=9090)
    with pytest.raises(LaiError, match="listen port"):
        cli.cmd_run(app, args)


def test_parser_knows_every_command():
    parser = cli.build_parser()
    for argv in (["ls"], ["list"], ["check"], ["gen", "--dry-run"], ["up"], ["down"], ["restart"],
                 ["status"], ["ps"], ["service"], ["ui"], ["logs", "-f"], ["load", "a"], ["unload"],
                 ["run", "a", "--port", "8081"], ["env"], ["smoke", "a", "--timeout", "600"]):
        assert callable(parser.parse_args(argv).handler)
