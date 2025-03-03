#!/usr/bin/env python3
"""
Logging Module for YouTube Frame Extractor

This module provides centralized logging functionality for the YouTube Frame Extractor package.
It supports both console and file logging, with rich formatting options for console output.

Usage:
    from youtube_frame_extractor.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("Error occurred", exc_info=True)  # With exception traceback
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from logging.handlers import RotatingFileHandler

try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# Track configured loggers to avoid duplicate handlers
_configured_loggers: Dict[str, logging.Logger] = {}

# Global console instance for rich logging
_rich_console = Console(stderr=True) if RICH_AVAILABLE else None

def _get_config() -> Dict[str, Any]:
    """
    Get logging configuration from settings.
    
    Returns:
        Dictionary with logging configuration
    """
    try:
        from .config import get_settings
        settings = get_settings()
        return {
            'level': getattr(settings.logging, 'level', DEFAULT_LOG_LEVEL),
            'format': getattr(settings.logging, 'format', DEFAULT_LOG_FORMAT),
            'file': getattr(settings.logging, 'file', None),
            'console': getattr(settings.logging, 'console', True),
            'rich_formatting': getattr(settings.logging, 'rich_formatting', RICH_AVAILABLE)
        }
    except (ImportError, AttributeError):
        return {
            'level': DEFAULT_LOG_LEVEL,
            'format': DEFAULT_LOG_FORMAT,
            'file': None,
            'console': True,
            'rich_formatting': RICH_AVAILABLE
        }

def configure_logger(
    logger: logging.Logger,
    level: Optional[Union[int, str]] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    rich_formatting: bool = True,
    force_reconfigure: bool = False
) -> logging.Logger:
    """
    Configure a logger with handlers and formatting.
    
    Args:
        logger: Logger to configure
        level: Log level (e.g., logging.INFO, 'INFO')
        log_format: Log message format
        log_file: Path to log file (if None, no file logging)
        console: Whether to enable console logging
        rich_formatting: Whether to use rich formatting for console output
        force_reconfigure: Force reconfiguration even if already configured
        
    Returns:
        Configured logger
    """
    # Skip if already configured unless forced
    if logger.name in _configured_loggers and not force_reconfigure:
        return logger
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Get defaults from config
    config = _get_config()
    
    # Use provided values or defaults
    level = level or config['level']
    log_format = log_format or config['format']
    log_file = log_file or config['file']
    console = console if console is not None else config['console']
    rich_formatting = rich_formatting if rich_formatting is not None else config['rich_formatting']
    
    # Convert string level to constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
    
    # Set logger level
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console:
        if rich_formatting and RICH_AVAILABLE:
            # Use rich handler for beautiful console output
            console_handler = RichHandler(
                console=_rich_console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                omit_repeated_times=False
            )
            # Rich handler uses a different format
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            # Use standard console handler
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
        
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler to avoid huge log files
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            # Don't fail if file logging can't be set up
            sys.stderr.write(f"Warning: Couldn't set up file logging to {log_file}: {str(e)}\n")
    
    # Remember that this logger is configured
    _configured_loggers[logger.name] = logger
    
    return logger

def get_logger(
    name: str,
    level: Optional[Union[int, str]] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    console: Optional[bool] = None,
    rich_formatting: Optional[bool] = None
) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (overrides default)
        log_format: Log message format (overrides default)
        log_file: Path to log file (overrides default)
        console: Whether to enable console logging (overrides default)
        rich_formatting: Whether to use rich formatting (overrides default)
        
    Returns:
        Configured logger
    """
    if name in _configured_loggers:
        logger = _configured_loggers[name]
        # Update settings if specified
        if any(param is not None for param in [level, log_format, log_file, console, rich_formatting]):
            logger = configure_logger(
                logger,
                level=level,
                log_format=log_format,
                log_file=log_file,
                console=console,
                rich_formatting=rich_formatting,
                force_reconfigure=True
            )
        return logger
    
    # Create new logger
    logger = logging.getLogger(name)
    return configure_logger(
        logger,
        level=level,
        log_format=log_format,
        log_file=log_file,
        console=console,
        rich_formatting=rich_formatting
    )

def set_global_log_level(level: Union[int, str]) -> None:
    """
    Set the log level for all configured loggers.
    
    Args:
        level: Log level to set
    """
    # Convert string level to constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
    
    # Update root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)
    
    # Update all configured loggers
    for logger_name, logger in _configured_loggers.items():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

def enable_debug_mode() -> None:
    """
    Enable debug mode for all loggers.
    This sets the log level to DEBUG for all configured loggers.
    """
    set_global_log_level(logging.DEBUG)

def get_package_logger() -> logging.Logger:
    """
    Get the main package logger.
    
    Returns:
        Package-level logger
    """
    return get_logger("youtube_frame_extractor")

# Configure the package-level logger
package_logger = get_package_logger()