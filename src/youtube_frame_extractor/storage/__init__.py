#!/usr/bin/env python3
"""
Storage package for YouTube Frame Extractor

This package offers:
- CloudStorage: AWS S3 / GCP handling
- LocalStorage: Local file system handling
"""

from .cloud import CloudStorage
from .local import LocalStorage

__all__ = [
    "CloudStorage",
    "LocalStorage",
]
