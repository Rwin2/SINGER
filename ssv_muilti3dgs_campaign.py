#!/usr/bin/env python3
"""Root-level entry point — delegates to the coruscant notebook CLI."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "notebooks" / "ssv_muilti3dgs_campaign_coruscant.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(SCRIPT)] + sys.argv[1:]))
