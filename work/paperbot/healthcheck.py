#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

path = Path(os.getenv("AI_SIGNAL_PATH", "/shared/signal.json"))
try:
    signal = json.loads(path.read_text(encoding="utf-8"))
    fresh = time.time() <= float(signal["expires_at"]) and time.time() - float(signal["generated_at"]) <= 1500
    valid = signal["action"] in {"buy", "hold", "sell"} and 0 <= float(signal["confidence"]) <= 1
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
    fresh = valid = False
sys.exit(0 if fresh and valid else 1)
