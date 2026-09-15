"""Run every check under ``tests/``: ``python tests/run.py [name ...]``.

Each ``tests/test_*.py`` holds plain ``test_*()`` functions that assert. No
framework and no dependency: importing a module and calling its functions is
the whole runner. pytest picks the same files up unchanged if you have it.
A check that cannot run here raises ``unittest.SkipTest``.

These are the unit checks. Simulator behaviour is validated separately, by
``./serving/validate.sh`` against recorded results.
"""

from __future__ import annotations

import importlib
import sys
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    names = argv or [p.stem for p in sorted(Path(__file__).parent.glob("test_*.py"))]
    failed, skipped, passed = [], [], 0
    for name in names:
        module = importlib.import_module(f"tests.{name}")
        for attr in sorted(vars(module)):
            if not attr.startswith("test_"):
                continue
            label = f"{name}.{attr}"
            try:
                getattr(module, attr)()
            except unittest.SkipTest as e:
                print(f"SKIP {label}: {e}")
                skipped.append(label)
            except Exception:
                print(f"FAIL {label}")
                traceback.print_exc()
                failed.append(label)
            else:
                print(f"PASS {label}")
                passed += 1
    print(f"\n{passed} passed, {len(skipped)} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
