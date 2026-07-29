"""
Script: Get Stats

Computes dataset column inventory and baseline KPIs (historical PID operation).

Usage:
    python scripts/get_stats.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import get_stats

if __name__ == "__main__":
    get_stats()
