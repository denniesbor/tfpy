"""
Author: Dennies Bor
Role:   Configures standard logging and path constants for the tfgic application.
"""

import sys
import logging
from pathlib import Path

import torch

APP_NAME = "tfgic"
BASE_DIR = Path(__file__).resolve().parent.parent

# All outputs live under data/ to keep the working directory clean
DEFAULT_DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DEFAULT_DATA_DIR / "models"
RESULTS_DIR = DEFAULT_DATA_DIR / "results"

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_data_dir(subdir=None):
    """Return (and create) a data directory, optionally under a subdirectory."""
    data_dir = DEFAULT_DATA_DIR / subdir if subdir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level="INFO"):
    """Configure and return a named logger with console and optional file output."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Resolve best available device: CUDA → MPS → CPU
def _resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()

# Ensure output directories exist at import time
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

config_logger = setup_logger(name=f"{APP_NAME}.config")
