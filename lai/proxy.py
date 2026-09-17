"""llama-swap: the HTTP API client, and starting/stopping the proxy itself.

llama-swap's management endpoints have moved between releases. The ones used
here are `GET /health`, `GET /running`, `GET /logs`, `GET /logs/stream` and
`POST /api/models/unload[/<id>]`. A 404 from any of them means the installed
llama-swap is older or newer than this code expects.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from lai import LaiError, term
from lai.schema import Settings


@dataclass(frozen=True)
class Response:
    status: int  # 0 when there was no HTTP response at all (refused, timed out)
    text: str

    @property
    def ok(self) -> bool:
        return self.status == 200

    def json(self):
        return json.loads(self.text)


class Proxy:
    """Client for llama-swap's OpenAI-compatible and management APIs."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def request(self, path: str, *, method: str = "GET", body: dict | None = None,
                timeout: float = 30) -> Response:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as reply:
                return Response(reply.status, reply.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            return Response(exc.code, exc.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return Response(0, str(exc))

    def healthy(self) -> bool:
        return self.request("/health", timeout=3).ok

    def list_models(self) -> Response:
        return self.request("/v1/models", timeout=15)

    def props(self, model_id: str) -> Response:
        return self.request(f"/props?model={urllib.parse.quote(model_id)}", timeout=20)

    def chat(self, model_id: str, messages: list[dict], *, max_tokens: int, timeout: float,
             tools: list[dict] | None = None) -> Response:
        body: dict = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        return self.request("/v1/chat/completions", method="POST", body=body, timeout=timeout)

    def running(self) -> Response:
        return self.request("/running", timeout=5)

    def unload(self, model_id: str | None = None) -> Response:
        path = "/api/models/unload" + (f"/{urllib.parse.quote(model_id)}" if model_id else "")
        return self.request(path, method="POST", timeout=60)

    def stream_logs(self) -> Iterator[str]:
        with urllib.request.urlopen(self.base_url + "/logs/stream", timeout=None) as reply:
            for line in reply:
                yield line.decode("utf-8", "replace")


def parse_running(response: Response) -> list[dict]:
    """/running as a list of {"model": id, ...}; tolerant of both payload shapes seen."""
    if not response.ok:
        return []
    try:
        data = response.json()
    except json.JSONDecodeError:
        return []
    items = data.get("running", []) if isinstance(data, dict) else data
    return [item if isinstance(item, dict) else {"model": str(item)} for item in items or []]


class ProxyService:
    """Runs llama-swap: through the systemd --user unit when it is installed,
    otherwise as a detached process tracked by a pidfile."""

    def __init__(self, settings: Settings, proxy: Proxy, *, config: Path, pidfile: Path,
                 logfile: Path, unit: Path) -> None:
        self.settings = settings
        self.proxy = proxy
        self.config = config
        self.pidfile = pidfile
        self.logfile = logfile
        self.unit = unit

    @staticmethod
    def binary() -> str:
        found = shutil.which("llama-swap") or str(Path.home() / "bin/llama-swap")
        if not Path(found).exists():
            raise LaiError(
                "llama-swap not found on PATH or in ~/bin. Download the linux amd64 "
                "release from https://github.com/mostlygeek/llama-swap/releases"
            )
        return found

    @property
    def listen(self) -> str:
        return f"{self.settings.listen_host}:{self.settings.listen_port}"

    def uses_systemd(self) -> bool:
        return self.unit.exists()

    def systemd_active(self) -> bool:
        if not self.uses_systemd():
            return False
        result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", "llama-swap"],
                                capture_output=True)
        return result.returncode == 0

    def pid(self) -> int | None:
        try:
            pid = int(self.pidfile.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (OSError, ValueError):
            return None

    def manager(self) -> str:
        """How the running proxy was started, for `lai status`."""
        if self.systemd_active():
            return "systemd --user"
        pid = self.pid()
        return f"pid {pid}" if pid else "started outside lai"

    def start(self) -> None:
        if not self.config.exists():
            raise LaiError("no generated config. Run `lai gen` first.")
        if self.uses_systemd():
            subprocess.run(["systemctl", "--user", "start", "llama-swap"], check=False)
            time.sleep(1)
            return
        if self.pid():
            term.info("already running")
            return
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        with open(self.logfile, "ab") as log:
            child = subprocess.Popen(
                [self.binary(), "--config", str(self.config), "--listen", self.listen],
                stdout=log, stderr=log, start_new_session=True,
            )
        self.pidfile.write_text(str(child.pid))
        for _ in range(40):
            time.sleep(0.25)
            if self.proxy.healthy():
                term.info(f"llama-swap up on {self.proxy.base_url} (pid {child.pid})")
                return
        term.warn(f"started pid {child.pid}, but /health has not answered yet; see {self.logfile}")

    def stop(self) -> None:
        if self.uses_systemd():
            # The unit's cgroup takes every backend process down with it.
            subprocess.run(["systemctl", "--user", "stop", "llama-swap"], check=False)
            term.info("stopped (systemd)")
            return
        pid = self.pid()
        if not pid:
            self.pidfile.unlink(missing_ok=True)
            term.info("not running")
            return
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            time.sleep(0.25)
            if self.pid() is None:
                break
        else:
            # Kill the whole group, not just the leader: an orphaned llama-server
            # keeps its VRAM and the next load fails for no visible reason.
            term.warn("did not exit on SIGTERM; killing the process group")
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        self.pidfile.unlink(missing_ok=True)
        stray = subprocess.run(["pgrep", "-c", "-x", "llama-server"], capture_output=True, text=True)
        if stray.stdout.strip() not in ("", "0"):
            term.warn(f"{stray.stdout.strip()} llama-server process(es) still hold VRAM: "
                      "pkill -x llama-server")
        term.info("stopped")
