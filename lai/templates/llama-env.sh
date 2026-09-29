#!/usr/bin/env bash
# @BANNER@
#
# Every llama-server process starts through this wrapper. It exists for the two
# environment failures that have broken this machine before:
#
#   1. conda's base env auto-activates and shadows the system libstdc++/libsycl.
#      llama-swap inherits whatever environment started it, so scrub
#      unconditionally.
#   2. Without oneAPI sourced, llama-server dies with
#      "libsvml.so: cannot open shared object file".
#
# Deliberately NOT `bash -l`: a login shell re-runs .bashrc and re-activates the
# conda env this script removes.

set -o pipefail

@ONEAPI_LINE@
LLAMA_SERVER=@LLAMA_SERVER@

strip_path() {
  local out="" p
  local IFS=:
  for p in ${1:-}; do
    case "$p" in
      *conda*|*mamba*|*miniforge*) continue ;;
      "") continue ;;
    esac
    out="${out:+$out:}$p"
  done
  printf '%s' "$out"
}

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL
unset PYTHONHOME PYTHONPATH
PATH="$(strip_path "${PATH:-}")"
LD_LIBRARY_PATH="$(strip_path "${LD_LIBRARY_PATH:-}")"
export PATH LD_LIBRARY_PATH

@ONEAPI_BLOCK@

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "FATAL: $LLAMA_SERVER missing or not executable" >&2
  exit 1
fi

exec "$LLAMA_SERVER" "$@"
