"""Make this uninstalled workspace importable for local unit tests only."""

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".cache" / "matplotlib"))
os.environ.setdefault("KERAS_HOME", str(WORKSPACE / ".cache" / "keras"))
sys.path.insert(0, str(WORKSPACE))
