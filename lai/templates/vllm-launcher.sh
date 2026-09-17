#!/usr/bin/env bash
# @BANNER@
#
# Container launcher for an engine="vllm" model. llama-swap runs this with the
# backend port it allocated as $1.
#
# Not started through llama-env.sh: that wrapper prepares the host for native
# SYCL binaries (oneAPI, conda scrub). The container carries its own userspace.

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "FATAL: docker not on PATH" >&2
  exit 1
fi

# From models.py. Use these in the launcher body instead of repeating values:
# `lai check` confirms --served-model-name and --max-model-len match them.
# shellcheck disable=SC2034  # a body may not use every one
MODEL_ID=@MODEL_ID@
MODEL_DIR=@MODEL_DIR@
CONTEXT=@CONTEXT@
REPO_ROOT=@REPO_ROOT@   # for files vendored into this repository (vendor/)

# ---- launcher body (Model.launcher) -------------------------------------------
@BODY@
