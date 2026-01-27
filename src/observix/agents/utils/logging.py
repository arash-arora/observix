import logging
import sys
from rich.logging import RichHandler

def setup_logging(level: int = logging.INFO) -> None:
    """Sets up rich logging for the agents package."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def get_logger(name: str) -> logging.Logger:
    """Returns a logger for the given name."""
    return logging.getLogger(name)
