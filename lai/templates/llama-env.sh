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

ONEAPI_SETVARS=@ONEAPI_SETVARS@
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

if [[ ! -r "$ONEAPI_SETVARS" ]]; then
  echo "FATAL: $ONEAPI_SETVARS not readable" >&2
  exit 1
fi

# Deliberately NO `set -u`, and stderr is NOT discarded.
#
# Intel's setvars.sh references variables that may be unset (ZSH_VERSION,
# SETVARS_CALL, ...). Under `set -u` the shell exits 127 before llama-server
# runs, and with stderr sent to /dev/null the failure is silent: llama-swap can
# only report "upstream command exited prematurely". Both mistakes were in the
# first version of this file.
#
# stdout is dropped because setvars is chatty; stderr is the only place a real
# failure will appear.
# shellcheck disable=SC1090,SC1091
if ! source "$ONEAPI_SETVARS" --force >/dev/null; then
  echo "FATAL: sourcing $ONEAPI_SETVARS failed" >&2
  exit 1
fi

export ZES_ENABLE_SYSMAN=1

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "FATAL: $LLAMA_SERVER missing or not executable" >&2
  exit 1
fi

exec "$LLAMA_SERVER" "$@"
