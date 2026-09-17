"""Smoke rungs against a scripted proxy: each failure mode is recognised."""

from __future__ import annotations

import json
import struct
import zlib

import pytest
from conftest import make_model, make_registry

from lai import smoke
from lai.proxy import Response


def ok(payload: dict) -> Response:
    return Response(200, json.dumps(payload))


def reply(content="", **message) -> Response:
    return ok({"choices": [{"message": {"content": content, **message}}], "usage": {"prompt_tokens": 18000}})


class ScriptedProxy:
    """Answers chat requests from a function of the request, and records them."""

    def __init__(self, chat=None, models=("m",), props=None):
        self._chat = chat or (lambda **_: reply("hello there my good friend"))
        self._models = models
        self._props = props
        self.requests: list[dict] = []

    def list_models(self):
        return ok({"data": [{"id": i} for i in self._models]})

    def props(self, model_id):
        return self._props or Response(404, "")

    def chat(self, model_id, messages, *, max_tokens, timeout, tools=None):
        request = {"messages": messages, "max_tokens": max_tokens, "tools": tools}
        self.requests.append(request)
        return self._chat(**request)


def context(proxy, **model_overrides):
    registry = make_registry(make_model(**model_overrides))
    return smoke.SmokeContext(registry.models[0], registry.settings, proxy, prefill_timeout_s=5)


def test_unregistered_model_fails_first_rung():
    with pytest.raises(smoke.RungFailed, match="not in /v1/models"):
        smoke.registered(context(ScriptedProxy(models=("other",))))


def test_premature_exit_suggests_bisecting():
    proxy = ScriptedProxy(chat=lambda **_: Response(502, "upstream command exited prematurely"))
    with pytest.raises(smoke.RungFailed) as failure:
        smoke.loads_and_completes(context(proxy))
    assert "lai env" in failure.value.hint and "lai run m" in failure.value.hint


def test_context_mismatch_fails():
    props = ok({"default_generation_settings": {"n_ctx": 8192}})
    with pytest.raises(smoke.RungFailed, match="n_ctx=8192"):
        smoke.context_matches_client(context(ScriptedProxy(props=props)))


def test_context_match_passes_with_parallel_slots():
    props = ok({"default_generation_settings": {"n_ctx": 32768}})
    assert "32768" in smoke.context_matches_client(context(ScriptedProxy(props=props), parallel=2))


def test_context_rung_skips_vllm():
    with pytest.raises(smoke.RungSkipped):
        smoke.context_matches_client(context(ScriptedProxy(), engine="vllm"))


def test_tool_call_with_broken_arguments_fails():
    call = {"function": {"name": "bash", "arguments": '{"command": "ls'}}
    proxy = ScriptedProxy(chat=lambda **_: reply(tool_calls=[call]))
    with pytest.raises(smoke.RungFailed) as failure:
        smoke.tool_call_parses(context(proxy))
    assert "KV" in failure.value.hint


def test_tool_call_missing_reports_the_reasoning_field():
    proxy = ScriptedProxy(chat=lambda **_: reply(reasoning="<function=bash>"))
    with pytest.raises(smoke.RungFailed) as failure:
        smoke.tool_call_parses(context(proxy, reasoning_field="reasoning"))
    assert "<function=bash>" in failure.value.hint


def test_tool_rung_skips_when_tools_disabled():
    with pytest.raises(smoke.RungSkipped):
        smoke.tool_call_parses(context(ScriptedProxy(), tools=False))


def colour_of(request) -> str:
    image_url = request["messages"][0]["content"][1]["image_url"]["url"]
    return "red" if image_url == red_url() else "blue"


def red_url():
    import base64
    return "data:image/png;base64," + base64.b64encode(smoke.solid_png((220, 20, 20))).decode()


def test_vision_passes_when_colours_are_named_in_reasoning_only():
    proxy = ScriptedProxy(chat=lambda **r: reply("", reasoning_content=f"it is {colour_of(r)}"))
    assert "both identified" in smoke.sees_images(context(proxy, vision=True, mmproj="p"))


def test_vision_fails_when_the_model_guesses_one_colour():
    proxy = ScriptedProxy(chat=lambda **_: reply("red"))
    with pytest.raises(smoke.RungFailed, match="solid blue image not identified"):
        smoke.sees_images(context(proxy, vision=True, mmproj="p"))


def test_prefill_timeout_prints_evidence_commands():
    proxy = ScriptedProxy(chat=lambda **_: Response(0, "timed out"))
    with pytest.raises(smoke.RungFailed) as failure:
        smoke.long_prefill(context(proxy))
    assert "xpu-smi" in failure.value.hint and "gdb" in failure.value.hint


def test_ladder_includes_vision_only_for_vision_models():
    assert len(smoke.ladder(make_model())) == 5
    assert len(smoke.ladder(make_model(vision=True))) == 6


def test_run_stops_at_first_failure(capsys):
    proxy = ScriptedProxy(models=("other",))
    assert smoke.run(context(proxy)) is False
    assert proxy.requests == []  # nothing after rung 1 ran
    assert "stopped at rung 1" in capsys.readouterr().out


def test_full_ladder_passes(capsys):
    call = {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}

    def chat(messages, tools, **_):
        if tools:
            return reply(tool_calls=[call])
        return reply("hello")

    assert smoke.run(context(ScriptedProxy(chat=chat))) is True
    assert "all rungs passed" in capsys.readouterr().out


def test_solid_png_is_a_valid_image():
    png = smoke.solid_png((1, 2, 3), size=4)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, depth, colour_type = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, colour_type) == (4, 4, 8, 2)
    idat_length = struct.unpack(">I", png[33:37])[0]
    pixels = zlib.decompress(png[41:41 + idat_length])
    assert pixels == (b"\x00" + bytes((1, 2, 3)) * 4) * 4
