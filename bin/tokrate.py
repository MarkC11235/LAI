#!/usr/bin/env python3
"""Measure prefill and decode tokens/sec for any OpenAI-compatible endpoint.

Engine-agnostic: it times the SSE stream from the client side, so a llama.cpp
entry and a vLLM entry are measured by exactly the same clock and the numbers
are directly comparable. Nothing is read from llama.cpp's `timings` or vLLM's
`metrics`, so neither engine's reporting quirks matter.

    tokrate.py qwen36-q6 qwen38-vllm
    tokrate.py --prompt-tokens 18000 --max-tokens 512 --runs 5 qwen38-vllm
    tokrate.py --reuse-prompt qwen38-vllm      # measure the prefix-cache-hit case

Defaults point at llama-swap on :9090, which also means the model loads on
demand. Point --url at a backend port to bypass the proxy entirely.

stdlib only. Python 3.9+.

Three measurement traps this avoids, all of which produce plausible-looking
wrong answers:

  * Cold load leaking into the timing. The first request per model loads
    weights; it is run, reported separately, and excluded from the median.

  * Prefix cache faking a fast prefill. Each run gets a freshly randomized
    prompt so the prefill is real work. --reuse-prompt measures the cached
    path deliberately, which is a different and also useful number.

  * Counting chunks instead of tokens. With speculative decoding a single SSE
    chunk can carry several accepted tokens, so chunk-rate understates decode
    by the acceptance factor. Token counts come from `usage`; chunk counting is
    only a fallback when the server sends no usage block.
"""

import argparse
import json
import random
import statistics
import sys
import time
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

# Filler vocabulary. Content is irrelevant -- generation length is pinned by
# max_tokens and ignore_eos -- but varied words keep the tokenizer honest and
# defeat prefix caching between runs.
WORDS = """arbor beacon cinder delta ember fathom girder hollow ingot jetty
kernel lattice marrow nimbus onyx pylon quarry rivet socket tundra umbra vellum
whisker xenon yarrow zephyr anvil bramble copper drift epoch flint granite
harbor isthmus juniper kelp lumen mortar nectar obsidian plinth quill runnel
slate thistle undertow vector willow""".split()


def build_prompt(target_tokens, seed):
    rng = random.Random(seed)
    # ~0.75 words per token is close enough for common English tokenizers; the
    # real count is read back from usage.prompt_tokens and reported.
    words = max(8, int(target_tokens * 0.75))
    filler = " ".join(rng.choice(WORDS) for _ in range(words))
    return (
        "Reference list (ignore its contents entirely):\n\n"
        + filler
        + "\n\nWrite one continuous paragraph about memory hierarchies."
    )


def measure(url, model, prompt, max_tokens, timeout):
    """One streaming request. Returns a dict of timings, or raises."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,  # reproducible, and keeps spec-decode acceptance stable
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # without this the model may stop at 20 tokens
    }
    req = urlrequest.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    resp = urlrequest.urlopen(req, timeout=timeout)

    first = last = None
    chunks = 0
    usage = {}

    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        body = line[5:].strip()
        if not body or body == b"[DONE]":
            continue
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        now = time.monotonic()
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta")
            # A delta holding only `role` is the preamble; an empty one belongs
            # to the stop chunk. Neither is generated text.
            if isinstance(delta, dict) and any(k != "role" for k in delta):
                if first is None:
                    first = now
                last = now
                chunks += 1
                break

    end = time.monotonic()
    if first is None:
        raise RuntimeError("no content received")

    out_tokens = usage.get("completion_tokens") or chunks
    in_tokens = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0

    ttft = first - start
    span = last - first

    return {
        "ttft": ttft,
        "total": end - start,
        "in_tokens": in_tokens,
        "cached": cached,
        "out_tokens": out_tokens,
        "chunks": chunks,
        "counted_from_usage": bool(usage.get("completion_tokens")),
        # Prefill excludes cached tokens: a prefix hit is not work done.
        "prefill": (in_tokens - cached) / ttft if in_tokens and ttft > 0 else None,
        # Inter-token only, so prefill never inflates it.
        "decode": (out_tokens - 1) / span if span > 0 and out_tokens > 1 else None,
    }


def fmt(value, width=8, places=1):
    return "n/a".rjust(width) if value is None else f"{value:{width}.{places}f}"


def run_model(args, model):
    print(f"\n=== {model} ===", flush=True)

    seed = [0]

    def prompt_for():
        # Fresh prompt each run unless the cached path is what we want.
        seed[0] = 1234 if args.reuse_prompt else seed[0] + 1
        return build_prompt(args.prompt_tokens, seed[0])

    try:
        cold = measure(args.url, model, prompt_for(), args.max_tokens, args.timeout)
    except HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:300]
        print(f"  FAILED {err.code}: {detail}")
        return
    except (URLError, RuntimeError, TimeoutError) as err:
        print(f"  FAILED: {err}")
        return

    print(
        f"  cold (load + first request): {cold['total']:.1f}s"
        f"   [excluded from medians]"
    )

    runs = []
    for i in range(args.runs):
        try:
            result = measure(args.url, model, prompt_for(), args.max_tokens, args.timeout)
        except Exception as err:  # noqa: BLE001 - one bad run shouldn't kill the sweep
            print(f"  run {i + 1}: FAILED: {err}")
            continue
        runs.append(result)
        print(
            f"  run {i + 1}: ttft {result['ttft']:6.2f}s"
            f"  prefill {fmt(result['prefill'], 8, 0)} tok/s"
            f"  decode {fmt(result['decode'], 6, 1)} tok/s"
            f"  ({result['in_tokens']} in"
            + (f", {result['cached']} cached" if result['cached'] else "")
            + f", {result['out_tokens']} out)",
            flush=True,
        )

    if not runs:
        return

    def median_of(key):
        values = [r[key] for r in runs if r[key] is not None]
        return statistics.median(values) if values else None

    print(
        f"  MEDIAN  ttft {statistics.median(r['ttft'] for r in runs):6.2f}s"
        f"  prefill {fmt(median_of('prefill'), 8, 0)} tok/s"
        f"  decode {fmt(median_of('decode'), 6, 1)} tok/s"
    )

    if not runs[0]["counted_from_usage"]:
        print(
            "  NOTE: server sent no usage block; token counts came from chunk\n"
            "        counting, which understates decode under speculative decoding."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="+", help="routing keys to measure")
    parser.add_argument("--url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--reuse-prompt",
        action="store_true",
        help="send an identical prompt every run to measure the prefix-cache-hit path",
    )
    args = parser.parse_args()

    print(f"endpoint {args.url}")
    print(
        f"~{args.prompt_tokens} prompt tokens, {args.max_tokens} generated, "
        f"{args.runs} runs, "
        + ("reused prompt (cache hits expected)" if args.reuse_prompt else "fresh prompt each run")
    )

    for model in args.models:
        run_model(args, model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
