"""LAI (local AI): a generated, validated LLM inference stack for one workstation.

Module map (each module's docstring has the details):

    schema      the registry contract: Settings and Model
    registry    load models.py and query it
    host        everything read from the machine, behind one seam for tests
    checks      `lai check`: rules that reject configs known to fail
    render      `lai gen`: pure functions from registry to generated files
    yamlout     minimal YAML emitter, so the runtime stays stdlib-only
    proxy       llama-swap HTTP client and process control
    smoke       `lai smoke`: live acceptance ladder for one model
    cli         argument parsing and command handlers
"""


class LaiError(Exception):
    """A user-facing failure: printed as `error: <message>`, exit status 1."""
