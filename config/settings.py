import os
import sys
import logging
from pathlib import Path

try:
    from loguru import logger

    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    import logging

    logger = logging.getLogger("tfgic")

APP_NAME = "tfgic"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_data_dir(subdir=None):
    if subdir:
        data_dir = DEFAULT_DATA_DIR / subdir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level="INFO"):
    if HAS_LOGURU:
        logger.remove()

        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            filter=lambda record: record["extra"].get("name", name) == name,
            level=level,
        )

        if log_file:
            log_path = Path(log_file)
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                filter=lambda record: record["extra"].get("name", name) == name,
                level=level,
                rotation="10 MB",
                compression="zip",
            )

        named_logger = logger.bind(name=name)
        return named_logger
    else:
        if isinstance(level, str):
            level = getattr(logging, level)

        std_logger = logging.getLogger(name)
        std_logger.setLevel(level)

        for handler in std_logger.handlers[:]:
            std_logger.removeHandler(handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        std_logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            std_logger.addHandler(file_handler)

        return std_logger


config_logger = setup_logger(name=f"{APP_NAME}.config")
