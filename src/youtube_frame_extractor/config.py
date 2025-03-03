#!/usr/bin/env python3
"""
Configuration Module for YouTube Frame Extractor

This module provides a centralized configuration system for the YouTube Frame Extractor
package using Pydantic for data validation and environment variable loading.

Configuration can be specified through:
1. Environment variables (prefixed with YFE_)
2. A configuration file (YAML or JSON)
3. Default values specified in this module

Example environment variables:
    YFE_OUTPUT_DIR=/path/to/output
    YFE_DEFAULT_INTERVAL=2.5
    YFE_MAX_CONCURRENT_DOWNLOADS=3
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from pydantic import BaseSettings, Field, validator
import yaml

# Default paths
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/youtube_frame_extractor/output")
DEFAULT_TEMP_DIR = os.path.expanduser("~/youtube_frame_extractor/temp")
DEFAULT_CACHE_DIR = os.path.expanduser("~/youtube_frame_extractor/cache")

class ExtractionMethod(str, Enum):
    """Extraction methods supported by the package."""
    BROWSER = "browser"
    DOWNLOAD = "download"

class VLMModel(str, Enum):
    """Supported VLM models."""
    CLIP_BASE = "openai/clip-vit-base-patch16"
    CLIP_LARGE = "openai/clip-vit-large-patch14"

class BrowserSettings(BaseSettings):
    """Browser-specific settings."""
    headless: bool = True
    browser_type: str = "chrome"
    selenium_timeout: int = 30  # seconds
    wait_time: float = 0.5  # seconds between actions
    user_agent: Optional[str] = None
    binary_location: Optional[str] = None
    extensions: List[str] = []
    arguments: List[str] = ["--mute-audio", "--disable-infobars", "--disable-extensions"]
    disable_gpu: bool = True
    default_interval: float = 2.0  # seconds between frame captures
    default_max_frames: int = 50

    class Config:
        env_prefix = "YFE_BROWSER_"

class DownloadSettings(BaseSettings):
    """Download-specific settings."""
    use_ytdlp: bool = True
    use_ffmpeg: bool = True
    temp_dir: str = DEFAULT_TEMP_DIR
    keep_video: bool = False
    default_format: str = "bestvideo[height<=720]+bestaudio/best[height<=720]"
    default_frame_rate: float = 1.0  # frames per second
    default_max_frames: int = 100
    timeout: int = 300  # seconds
    max_concurrent_downloads: int = 3
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds

    class Config:
        env_prefix = "YFE_DOWNLOAD_"

class VLMSettings(BaseSettings):
    """Vision Language Model settings."""
    default_model: VLMModel = VLMModel.CLIP_BASE
    device: str = "cuda" if any('cuda' in d.lower() for d in os.popen('python -c "import torch; print(torch.cuda.is_available())"').read()) else "cpu"
    default_threshold: float = 0.3
    cache_dir: str = DEFAULT_CACHE_DIR
    batch_size: int = 16
    preload_models: bool = False
    timeout: int = 30  # seconds

    class Config:
        env_prefix = "YFE_VLM_"

class StorageSettings(BaseSettings):
    """Storage settings."""
    output_dir: str = DEFAULT_OUTPUT_DIR
    use_cloud_storage: bool = False
    cloud_provider: Optional[str] = None  # "aws", "gcp", "azure"
    aws_bucket: Optional[str] = None
    aws_region: Optional[str] = None
    gcs_bucket: Optional[str] = None
    azure_container: Optional[str] = None

    class Config:
        env_prefix = "YFE_STORAGE_"

class LoggingSettings(BaseSettings):
    """Logging settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True
    rich_formatting: bool = True

    @validator("level")
    def validate_log_level(cls, v):
        """Validate and convert log level string to logging constant."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        if isinstance(v, str) and v.upper() in levels:
            return levels[v.upper()]
        if isinstance(v, int) and v in levels.values():
            return v
        raise ValueError(f"Invalid log level: {v}. Must be one of {list(levels.keys())}")

    class Config:
        env_prefix = "YFE_LOGGING_"

class BatchSettings(BaseSettings):
    """Batch processing settings."""
    default_workers: int = 3
    max_workers: int = 10
    generate_report: bool = True
    report_format: str = "markdown"  # "markdown", "html", "json"
    timeout: int = 3600  # seconds

    class Config:
        env_prefix = "YFE_BATCH_"

class Settings(BaseSettings):
    """Main settings class for YouTube Frame Extractor."""
    # General settings
    version: str = "1.0.0"
    default_method: ExtractionMethod = ExtractionMethod.BROWSER
    user_agent: str = "YouTube Frame Extractor/1.0.0"
    
    # Sub-settings
    browser: BrowserSettings = BrowserSettings()
    download: DownloadSettings = DownloadSettings()
    vlm: VLMSettings = VLMSettings()
    storage: StorageSettings = StorageSettings()
    logging: LoggingSettings = LoggingSettings()
    batch: BatchSettings = BatchSettings()

    # Additional settings
    config_file: Optional[str] = None
    
    class Config:
        env_prefix = "YFE_"
        case_sensitive = False
        
    def __init__(self, **data: Any):
        """Initialize settings with optional config file loading."""
        super().__init__(**data)
        
        # Load from config file if specified
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_config_file()
    
    def _load_from_config_file(self) -> None:
        """Load settings from a configuration file."""
        path = Path(self.config_file)
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            # Update settings from file
            self._update_from_dict(config_data)
        
        except Exception as e:
            logging.warning(f"Error loading config file {self.config_file}: {str(e)}")
    
    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Recursively update settings from a dictionary."""
        for key, value in data.items():
            if key in self.__fields__:
                if isinstance(value, dict) and hasattr(self, key) and hasattr(getattr(self, key), '__fields__'):
                    # Recursively update nested settings
                    nested_obj = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if nested_key in nested_obj.__fields__:
                            setattr(nested_obj, nested_key, nested_value)
                else:
                    # Update top-level setting
                    setattr(self, key, value)
    
    def save_to_file(self, path: Union[str, Path], format: str = 'yaml') -> None:
        """Save current settings to a configuration file."""
        path = Path(path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert settings to dict
        settings_dict = self.dict()
        
        try:
            if format.lower() == 'yaml':
                with open(path, 'w') as f:
                    yaml.dump(settings_dict, f, default_flow_style=False)
            elif format.lower() == 'json':
                with open(path, 'w') as f:
                    json.dump(settings_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")
        
        except Exception as e:
            logging.error(f"Error saving settings to {path}: {str(e)}")
            raise

# Create a global settings instance
settings = Settings()

def load_settings(config_file: Optional[str] = None) -> Settings:
    """
    Load settings from a config file and/or environment variables.
    
    Args:
        config_file: Path to a YAML or JSON configuration file
    
    Returns:
        Settings instance
    """
    global settings
    
    if config_file:
        settings = Settings(config_file=config_file)
    
    # Ensure directories exist
    os.makedirs(settings.storage.output_dir, exist_ok=True)
    os.makedirs(settings.download.temp_dir, exist_ok=True)
    os.makedirs(settings.vlm.cache_dir, exist_ok=True)
    
    return settings

def get_settings() -> Settings:
    """
    Get the current settings instance.
    
    Returns:
        Settings instance
    """
    return settings