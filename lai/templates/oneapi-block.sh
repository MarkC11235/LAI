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

