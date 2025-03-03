#!/usr/bin/env python3
"""
Extractors package for YouTube Frame Extractor

This package contains extractor implementations for retrieving frames from YouTube videos:
- BaseExtractor (abstract class)
- BrowserExtractor (Selenium-based)
- DownloadExtractor (yt-dlp + ffmpeg-based)
"""

from .base import BaseExtractor
from .browser import BrowserExtractor
from .download import DownloadExtractor

__all__ = [
    "BaseExtractor",
    "BrowserExtractor",
    "DownloadExtractor",
]
