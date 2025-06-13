"""
Configuration module for GIC calculations.

Provides utilities for:
- Data directory management
- Custom logger setup using Loguru
"""

import os
import sys
import logging
from pathlib import Path

# Try to import loguru, fallback to standard logging if not available
try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    import logging
    logger = logging.getLogger("tfgic")

# Application name - used for the logger
APP_NAME = "tfgic"

# Path settings
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_data_dir(subdir=None):
    """
    Get the path to a data directory, creating it if it doesn't exist.

    Parameters
    ----------
    subdir : str or Path, optional
        Subdirectory within the data directory, if None returns the main data directory

    Returns
    -------
    Path
        Path to the requested data directory
    """
    if subdir:
        data_dir = DEFAULT_DATA_DIR / subdir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level="INFO"):
    """
    Set up a custom logger with console and optional file output, 
    using Loguru if available or falling back to standard logging.

    Parameters
    ----------
    name : str, optional
        Logger name, default is APP_NAME
    log_file : str or Path, optional
        Path to log file, if None no file logging is set up
    level : str or int, optional
        Logging level, default is "INFO" for Loguru or logging.INFO

    Returns
    -------
    logger
        Configured logger instance (loguru.logger or logging.Logger)
    """
    if HAS_LOGURU:
        # Remove any existing handlers
        logger.remove()
        
        # Add console handler with appropriate format
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            filter=lambda record: record["extra"].get("name", name) == name,
            level=level,
        )
        
        # Add file handler if requested
        if log_file:
            # Ensure the log directory exists
            log_path = Path(log_file)
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                filter=lambda record: record["extra"].get("name", name) == name,
                level=level,
                rotation="10 MB",  # Rotate when the file reaches 10MB
                compression="zip",  # Compress rotated files
            )
        
        # Create a logger with the specified name
        named_logger = logger.bind(name=name)
        return named_logger
    else:
        # Fall back to standard logging if Loguru is not available
        # Convert string level to logging level if needed
        if isinstance(level, str):
            level = getattr(logging, level)
            
        # Create logger
        std_logger = logging.getLogger(name)
        std_logger.setLevel(level)

        # Remove existing handlers if present
        for handler in std_logger.handlers[:]:
            std_logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        std_logger.addHandler(console_handler)

        # Create file handler if requested
        if log_file:
            # Ensure the log directory exists
            log_path = Path(log_file)
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            std_logger.addHandler(file_handler)

        return std_logger


# Create a logger for the configuration module
config_logger = setup_logger(name=f"{APP_NAME}.config")