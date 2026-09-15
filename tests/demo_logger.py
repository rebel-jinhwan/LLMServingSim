"""Every logger surface at once, for looking at: ``python tests/demo_logger.py``.

A visual smoke test, not an assertion, so tests/run.py does not pick it up
(the name is not ``test_*``). Was the ``__main__`` block of
``serving/core/logger.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmservingsim.serving.core.logger import (  # noqa: E402
    configure_logger, get_logger, print_banner, progress, stage)


def main() -> None:
    configure_logger(level="DEBUG")
    print_banner()
    log = get_logger("Demo", node_id=0, instance_id=1)
    log.info("iteration 0 finished, exposed communication 0 cycles.")
    log.debug("debug info for scheduling decision.")
    log.warning("KV cache usage above 80%.")
    log.error("ASTRA-Sim pipe read timeout.")
    log.success("simulation wrapped up")
    log.summary("TTFT mean: 7.71 s  |  TPOT mean: 55.8 ms")
    with stage("example work"):
        import time as _t
        _t.sleep(0.2)
    with progress("cooking", total=3) as bar:
        for _ in range(3):
            import time as _t
            _t.sleep(0.1)
            bar.advance()


if __name__ == "__main__":
    main()
