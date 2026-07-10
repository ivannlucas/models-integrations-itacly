"""
Script: Data Processing

Generates raw data from the simulator, filters production records
and creates temporal splits.

Usage:
    python scripts/data_processing.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import data_processing

if __name__ == "__main__":
    data_processing()
