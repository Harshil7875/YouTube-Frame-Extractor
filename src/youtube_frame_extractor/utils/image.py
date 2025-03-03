#!/usr/bin/env python3
"""
Image Utility Module for YouTube Frame Extractor

This module provides helper functions for common image operations:
- Loading and saving images
- Resizing, grayscale conversion
- Calculating image differences (MSE, SSIM)
"""

import os
import math
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Union, Optional

try:
    import cv2  # for SSIM or advanced comparisons
except ImportError:
    cv2 = None

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import ImageProcessingError

logger = get_logger(__name__)
settings = get_settings()


def load_image(path: str) -> Image.Image:
    """
    Load an image from disk into a PIL Image.

    Args:
        path: Local file path to the image.

    Returns:
        PIL Image object.

    Raises:
        ImageProcessingError: If the file cannot be opened or is not found.
    """
    if not os.path.exists(path):
        msg = f"Image file not found: {path}"
        logger.error(msg)
        raise ImageProcessingError(msg)

    try:
        image = Image.open(path)
        image.load()  # Ensures the image is read completely
        logger.debug(f"Loaded image '{path}' size={image.size}, mode={image.mode}")
        return image
    except Exception as e:
        msg = f"Error loading image '{path}': {str(e)}"
        logger.error(msg)
        raise ImageProcessingError(msg)


def save_image(image: Image.Image, path: str, format: Optional[str] = None) -> None:
    """
    Save a PIL Image to disk.

    Args:
        image: PIL Image instance.
        path: Destination file path.
        format: Override image format (e.g., "JPEG", "PNG"), otherwise inferred from extension.

    Raises:
        ImageProcessingError: If saving fails.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        image.save(path, format=format)
        logger.debug(f"Saved image to '{path}' (format={format})")
    except Exception as e:
        msg = f"Error saving image to '{path}': {str(e)}"
        logger.error(msg)
        raise ImageProcessingError(msg)


def resize_image(image: Image.Image, size: Tuple[int, int], keep_aspect: bool = False) -> Image.Image:
    """
    Resize an image to the given size.

    Args:
        image: PIL Image.
        size: (width, height) target dimensions.
        keep_aspect: If True, resizes while maintaining aspect ratio (may add borders).

    Returns:
        Resized PIL Image.
    """
    if keep_aspect:
        logger.debug(f"Resizing with aspect ratio to fit size={size}")
        image = ImageOps.contain(image, size)  # shrinks or fits while maintaining ratio
    else:
        logger.debug(f"Resizing directly to size={size}")
        image = image.resize(size, Image.ANTIALIAS)
    return image


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert an image to grayscale mode.

    Args:
        image: PIL Image.

    Returns:
        Grayscale PIL Image.
    """
    logger.debug(f"Converting image to grayscale (original mode={image.mode})")
    return image.convert("L")


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy array (RGB or grayscale).

    Args:
        image: PIL Image.

    Returns:
        NumPy array with shape (H, W, 3) or (H, W).
    """
    return np.array(image)


def compute_mse(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute the Mean Squared Error (MSE) between two images.

    Args:
        img1: First PIL Image.
        img2: Second PIL Image.

    Returns:
        MSE value (0 indicates identical images).
    """
    if img1.size != img2.size or img1.mode != img2.mode:
        logger.debug("Images differ in size or mode, converting second image to match first.")
        img2 = img2.resize(img1.size).convert(img1.mode)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    mse = float(np.mean((arr1 - arr2) ** 2))
    logger.debug(f"MSE between images: {mse:.4f}")
    return mse


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images using OpenCV if available.

    Args:
        img1: First PIL Image.
        img2: Second PIL Image.

    Returns:
        SSIM value in the range [-1, 1] (1 indicates identical images).
        Returns -1.0 if OpenCV is not installed or an error occurs.
    """
    if cv2 is None:
        logger.warning("OpenCV not installed; returning -1.0 for SSIM.")
        return -1.0

    # Convert both images to grayscale for SSIM
    gray1 = convert_to_grayscale(img1)
    gray2 = convert_to_grayscale(img2)

    if gray1.size != gray2.size:
        logger.debug("Images differ in size; resizing second image to match first.")
        gray2 = gray2.resize(gray1.size)

    arr1 = np.array(gray1)
    arr2 = np.array(gray2)

    try:
        # OpenCV expects images in uint8
        if arr1.dtype != np.uint8:
            arr1 = arr1.astype(np.uint8)
        if arr2.dtype != np.uint8:
            arr2 = arr2.astype(np.uint8)

        # SSIM from OpenCV 4.5+ can be via `cv2.quality.QualitySSIM_compute`, but let's do old approach:
        (score, _) = cv2.quality.QualitySSIM_compute(arr1, arr2)
        # score is a tuple with a single float element
        ssim_val = float(score[0])
        logger.debug(f"SSIM between images: {ssim_val:.4f}")
        return ssim_val
    except Exception as e:
        logger.warning(f"Error computing SSIM: {str(e)}")
        return -1.0
