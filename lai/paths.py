"""Filesystem layout of the repository and of the files lai installs."""

from pathlib import Path
import os
import socket

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"

REGISTRY_FILE = REPO_ROOT / "models.py"
"""The fallback registry, used when no host file below matches this machine."""
HOSTS_DIR = REPO_ROOT / "hosts"
"""Per-machine registries, named `<hostname>.py`, so one checkout serves every box."""
REGISTRY_ENV = "LAI_REGISTRY"

def registry_file(environ: dict[str, str] | None = None, hostname: str | None = None) -> Path:
    """The registry this machine uses: $LAI_REGISTRY if set, else
    hosts/<hostname>.py if it exists, else models.py."""
    environ = os.environ if environ is None else environ
    override = environ.get(REGISTRY_ENV)
    if override:
        return Path(override).expanduser()
    host_file = HOSTS_DIR / f"{hostname or socket.gethostname()}.py"
    return host_file if host_file.exists() else REGISTRY_FILE


LDR_ENV_FILE = REPO_ROOT / "ldr.env"

GEN_DIR = REPO_ROOT / "gen"
"""Everything `lai gen` writes. Never edited by hand; not committed."""
SWAP_CONFIG = GEN_DIR / "llama-swap.yaml"
ENV_WRAPPER = GEN_DIR / "llama-env.sh"
OPENCODE_GENERATED = GEN_DIR / "opencode.json"
PROXY_PIDFILE = GEN_DIR / "llama-swap.pid"
PROXY_LOG = GEN_DIR / "llama-swap.log"

SYSTEMD_UNIT = Path.home() / ".config/systemd/user/llama-swap.service"


def vllm_launcher(gen_dir: Path, model_id: str) -> Path:
    """Where the generated container launcher for a vLLM model lives."""
    return gen_dir / f"vllm-{model_id}.sh"
