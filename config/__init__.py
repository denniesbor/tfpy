"""
Author: Dennies Bor
Role:   Package initialisation and public exports for tfgic.
"""

from .settings import (
    setup_logger,
    get_data_dir,
    config_logger,
    APP_NAME,
    DEFAULT_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DEVICE,
)

__all__ = [
    "setup_logger",
    "get_data_dir",
    "config_logger",
    "APP_NAME",
    "DEFAULT_DATA_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "DEVICE",
]
