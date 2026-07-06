# utils/ui.py
# Description: Shared module for console user interface elements, like colored logging.

import sys


class Colors:
    """Contains ANSI escape codes for coloring text in the terminal."""
    HEADER = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Configuration for different log levels
LOG_LEVEL_CONFIG = {
    "INFO": {"prefix": "[INFO]", "color": Colors.GREEN},
    "ERROR": {"prefix": "[ERROR]", "color": Colors.RED},
    "WARNING": {"prefix": "[WARNING]", "color": Colors.YELLOW},
    "HEADER": {"prefix": "---", "color": Colors.HEADER},
    "SUCCESS": {"prefix": "[SUCCESS]", "color": Colors.GREEN},
}

# Color mapping for status messages
STATUS_COLORS = {
    "PASS": Colors.GREEN,
    "FAIL": Colors.RED,
    "MEJORABLE": Colors.YELLOW,
    "OK": Colors.GREEN,
    "KO": Colors.RED,
    "DEFAULT": Colors.ENDC
}


def log_message(message: str, level: str = "INFO", stream=sys.stderr):
    """
    Prints a formatted message to the console with colors and prefixes.

    Args:
        message (str): The message to print.
        level (str): The log level (e.g., "INFO", "ERROR"). It determines the
                     color and prefix of the message.
        stream: The output stream (defaults to sys.stderr).
    """
    config = LOG_LEVEL_CONFIG.get(level.upper(), {"prefix": "", "color": ""})
    color = config["color"]
    prefix = config["prefix"]
    print(f"{color}{Colors.BOLD}{prefix}{Colors.ENDC}{color} {message}{Colors.ENDC}", file=stream)
