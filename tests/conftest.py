import sys
from pathlib import Path

# Project root = parent of this tests/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to sys.path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
