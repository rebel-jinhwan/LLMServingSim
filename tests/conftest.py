"""Put the repository root on sys.path so ``import serving`` works from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
