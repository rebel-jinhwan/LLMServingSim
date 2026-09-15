"""``llmservingsim <command> [args...]`` — the console entry point.

Each command runs the subpackage's own ``__main__``, so
``llmservingsim serving --help`` and ``llmservingsim serving --help``
are the same program.
"""

import runpy
import sys

COMMANDS = {
    "serving": "llmservingsim.serving",
    "profiler": "llmservingsim.profiler",
    "bench": "llmservingsim.bench",
    "workloads": "llmservingsim.workloads.generators",
}

USAGE = f"usage: llmservingsim {{{','.join(COMMANDS)}}} [args...]"


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command not in COMMANDS:
        if command in (None, "-h", "--help"):
            print(USAGE)
            return
        sys.exit(f"{USAGE}\nllmservingsim: unknown command {command!r}")
    sys.argv = [f"llmservingsim {command}", *sys.argv[2:]]
    runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
