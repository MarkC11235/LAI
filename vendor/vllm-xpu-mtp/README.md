# vLLM XPU MTP patches (vendored)

`qwen38-vllm` runs multi-token-prediction speculative decoding on a nightly
vLLM XPU image. On the Arc Pro B70 that needs two monkeypatches, applied inside
the container before `vllm serve` starts (see the launcher in `models.py`):

| File | Purpose |
|---|---|
| `patch_mtp_nightly.py` | MTP support for the pinned nightly image; its behaviour is gated on `B70_MTP_BF16_DRAFT=1` |
| `patch_mtp_boundary.py` | The cookbook's second MTP patch, required alongside the first |

They come from SergiioB's
[intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook),
`patches/`. They are vendored here rather than mounted from a clone because the
stack depends on them: a cleaned-up `~/projects` would otherwise break the model
with no connection to the change.

The launcher refuses to start if either file is missing.

## Populating or updating

```bash
COOKBOOK=~/projects/intel-arc-pro-b70-inference-cookbook
git -C "$COOKBOOK" pull
cp "$COOKBOOK"/patches/patch_mtp_nightly.py "$COOKBOOK"/patches/patch_mtp_boundary.py .
git -C "$COOKBOOK" rev-parse HEAD > SOURCE_COMMIT
sha256sum patch_*.py > SHA256SUMS
```

Then read both files. They execute inside a container with the GPU attached
and patch vLLM internals, and the source repository is small and
single-maintainer; the checksums are the only integrity record.

The patches target the image digest pinned in `models.py`. Changing the image
means re-validating MTP (acceptance rate in the startup log, then decode
throughput); see `docs/findings.md`.

Check the upstream repository's license before publishing these files.
