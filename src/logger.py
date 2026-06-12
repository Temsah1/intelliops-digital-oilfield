"""logger.py — Centralised logging configuration."""
import logging
import sys
from pathlib import Path

LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

COLOURS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET":    "\033[0m",
}

class ColourFormatter(logging.Formatter):
    def format(self, record):
        colour = COLOURS.get(record.levelname, COLOURS["RESET"])
        record.levelname = f"{colour}{record.levelname}{COLOURS['RESET']}"
        return super().format(record)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColourFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(ch)
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "oilfield.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(fh)
    logger.propagate = False
    return logger
