import sys
from pathlib import Path
from loguru import logger

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default logger
logger.remove()

def _safe_print(msg: str) -> None:
    try:
        sys.stdout.write(msg)
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        sys.stdout.flush()

# Console Logger
logger.add(
    sink=_safe_print,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)

# File Logger
logger.add(
    LOG_DIR / "app.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
)

__all__ = ["logger"]