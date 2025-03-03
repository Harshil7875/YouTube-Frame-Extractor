#!/usr/bin/env python3
"""
Utility functions and helpers for the YouTube Frame Extractor project.

Provides:
- browser.py: Selenium driver setup & JavaScript utilities
- concurrency.py: Parallel map & chunked map
- image.py: Basic image operations (load/save, resize, MSE, SSIM)
- video.py: Video metadata, OpenCV-based extraction, scene change detection
"""

from .browser import (
    create_driver,
    quit_driver,
    wait_for_element,
    execute_javascript,
    capture_screenshot_as_base64,
    highlight_element,
)
from .concurrency import (
    parallel_map,
    chunked_parallel_map,
)
from .image import (
    load_image,
    save_image,
    resize_image,
    convert_to_grayscale,
    image_to_numpy,
    compute_mse,
    compute_ssim,
)
from .video import (
    get_video_metadata,
    extract_frames_opencv,
    scene_change_detection,
)

__all__ = [
    # browser
    "create_driver",
    "quit_driver",
    "wait_for_element",
    "execute_javascript",
    "capture_screenshot_as_base64",
    "highlight_element",
    # concurrency
    "parallel_map",
    "chunked_parallel_map",
    # image
    "load_image",
    "save_image",
    "resize_image",
    "convert_to_grayscale",
    "image_to_numpy",
    "compute_mse",
    "compute_ssim",
    # video
    "get_video_metadata",
    "extract_frames_opencv",
    "scene_change_detection",
]
