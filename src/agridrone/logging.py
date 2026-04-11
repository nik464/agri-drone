"""
logging.py - Structured logging setup using loguru.
"""
import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | str | None = None,
    rotation: str = "100 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru logger for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file; if None, logs to stderr only
        rotation: Log rotation size (e.g., "100 MB")
        retention: Retention policy (e.g., "7 days")
    """
    # Remove default handler
    logger.remove()

    # Add stdout handler (always)
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level,
    )

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz",
        )

    logger.info(f"Logging initialized: level={log_level}, file={log_file}")


def get_logger():
    """Get loguru logger instance."""
    return logger
