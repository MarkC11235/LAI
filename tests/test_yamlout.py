"""The emitter must produce YAML that parses back to exactly the input."""

from __future__ import annotations

import pytest
import yaml

from lai import yamlout

AWKWARD_STRINGS = [
    "", "yes", "no", "on", "off", "null", "~", "true", "1", "0x10", "1e3", "08",
    "a: b", "#comment", "- item", "quote \" and \\ backslash", "tab\there",
    "${PORT}", "per_layer_token_embd\\.weight=CPU", "Qwen3.8 · medium", "@AT",
]


@pytest.mark.parametrize("text", AWKWARD_STRINGS)
def test_strings_survive(text):
    assert yaml.safe_load(yamlout.dump({"k": text})) == {"k": text}


@pytest.mark.parametrize("key", ["yes", "on", "a b", "${MODEL_ID}:high", "1", "qwen38-27b-c0-mtp"])
def test_keys_survive(key):
    assert yaml.safe_load(yamlout.dump({key: 1})) == {key: 1}


@pytest.mark.parametrize("number", [0, -1, 1800, 0.0, 0.95, 1.0, 1e-05, 2.5e10, -3.5])
def test_numbers_keep_their_type(number):
    loaded = yaml.safe_load(yamlout.dump({"n": number}))["n"]
    assert loaded == number and type(loaded) is type(number)


def test_nested_structures_round_trip():
    document = {
        "models": {
            "a": {
                "cmd": "/gen/llama-env.sh\n--port ${PORT}\n-ot 'x\\.y=CPU'\n",
                "ttl": -1,
                "capabilities": {"tools": True, "in": ["text", "image"], "out": ["text"]},
                "setParams": {"chat_template_kwargs": {"reasoning_effort": "low"}},
                "empty_map": {},
                "empty_list": [],
                "none": None,
            }
        },
        "list_of_maps": [{"a": 1, "b": [1, 2]}, {"c": {"d": "e"}}],
        "nested_lists": [[1, 2], ["x"]],
    }
    assert yaml.safe_load(yamlout.dump(document)) == document


def test_multiline_commands_use_literal_blocks():
    text = yamlout.dump({"cmd": "line one\nline two\n"})
    assert "cmd: |\n  line one\n  line two\n" in text


@pytest.mark.parametrize("value", ["no trailing newline\nsecond", "two trailing\n\n", " leading space\nx\n"])
def test_strings_unsafe_for_literal_blocks_are_quoted(value):
    text = yamlout.dump({"cmd": value})
    assert "|" not in text
    assert yaml.safe_load(text) == {"cmd": value}


def test_header_becomes_comments():
    text = yamlout.dump({"a": 1}, header="first\n\nsecond")
    assert text.startswith("# first\n#\n# second\n\na: 1\n")
