"""
Script: Train

Trains the MLP model, evaluates on test set and saves all artifacts.

Usage:
    python scripts/train.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import train

if __name__ == "__main__":
    train()
