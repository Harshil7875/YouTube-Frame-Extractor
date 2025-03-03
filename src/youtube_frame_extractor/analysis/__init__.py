#!/usr/bin/env python3
"""
Analysis package for YouTube Frame Extractor

This package provides classes and methods for advanced analysis of frames, including:
- CLIP-based semantic search (clip.py)
- Object detection (object_detection.py)
- OCR (ocr.py)
- A generic VLM interface (vlm.py)
"""

from .clip import CLIPAnalyzer
from .object_detection import ObjectDetector
from .ocr import OCRAnalyzer
from .vlm import VLMAnalyzer

__all__ = [
    "CLIPAnalyzer",
    "ObjectDetector",
    "OCRAnalyzer",
    "VLMAnalyzer",
]
