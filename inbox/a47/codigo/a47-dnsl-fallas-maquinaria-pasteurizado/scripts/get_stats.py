import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_get_stats

if __name__ == "__main__":
    run_get_stats()
