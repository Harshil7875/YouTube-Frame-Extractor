#!/usr/bin/env python3
"""
Generic Vision Language Model (VLM) Module for YouTube Frame Extractor

This module defines a unifying interface to handle text-image similarity
across different VLM backends (e.g., CLIP, BLIP, etc.).

Currently, it integrates with the CLIPAnalyzer from clip.py by default,
but can be extended to support additional VLMs.
"""

from typing import Optional, Union
from PIL import Image

from .clip import CLIPAnalyzer
# If you have more VLM backends, import them here, e.g. from .blip import BLIPAnalyzer, etc.

from ..exceptions import VLMError
from ..logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class BaseVLM:
    """
    Abstract base class for Vision Language Models.

    Subclasses must implement `calculate_similarity(image, text)`.
    """

    def calculate_similarity(self, image: Image.Image, text: str) -> float:
        """
        Calculate a similarity score between an image and a text query.

        Args:
            image: A PIL Image.
            text: A textual query or description.

        Returns:
            A float representing similarity (range depends on the model).

        Raises:
            VLMError: If something goes wrong in the inference process.
        """
        raise NotImplementedError("Subclasses must implement calculate_similarity()")


class VLMAnalyzer(BaseVLM):
    """
    A unifying VLM analyzer that can switch between different backends
    (like CLIP, BLIP, etc.) based on configuration or a specified model name.

    Usage:
        analyzer = VLMAnalyzer(model_name="openai/clip-vit-base-patch16")
        similarity = analyzer.calculate_similarity(pil_image, "A dog playing frisbee")
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the VLM analyzer with a chosen model.

        Args:
            model_name: Name of the model to load (e.g. "openai/clip-vit-large-patch14").
            device: "cuda" or "cpu". If None, it uses settings.vlm.device.
        """
        self.model_name = model_name or settings.vlm.default_model.value
        self.device = device or settings.vlm.device
        self.backend = None

        # For now, we assume everything is CLIP-based unless you expand to more VLMs.
        logger.info(f"Initializing VLMAnalyzer with model '{self.model_name}' on device '{self.device}'")

        # If you had multiple backends, you'd do a conditional here:
        # e.g. if "clip" in self.model_name.lower(): self.backend = <some CLIP class>
        #      elif "blip" in self.model_name.lower(): self.backend = BLIPAnalyzer(...)
        # else:
        #      raise VLMError("Unsupported or unknown VLM model: ...")

        try:
            # We'll just use CLIP as a default
            self.backend = CLIPAnalyzer(model_name=self.model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to initialize VLM backend for '{self.model_name}': {str(e)}")
            raise VLMError(f"Failed to initialize VLM model '{self.model_name}': {str(e)}")

        logger.info(f"VLMAnalyzer successfully loaded backend for '{self.model_name}'")

    def calculate_similarity(self, image: Image.Image, text: str) -> float:
        """
        Calculate similarity using the chosen VLM backend.

        Args:
            image: A PIL Image.
            text: A text description/query.

        Returns:
            A float representing similarity. Typically in range [-1.0, +1.0] for CLIP.

        Raises:
            VLMError: If the backend fails or is not set.
        """
        if not self.backend:
            raise VLMError("VLM backend not initialized")

        return self.backend.calculate_similarity(image, text)

    def batch_calculate_similarity(self, images, text: str) -> list:
        """
        Optionally, calculate similarity for a list of images.
        Some models can handle batching more efficiently.

        Args:
            images: A list of PIL Images.
            text: A text description/query.

        Returns:
            A list of floats representing similarity for each image.
        """
        if not self.backend:
            raise VLMError("VLM backend not initialized")
        if not images:
            return []

        # If your CLIPAnalyzer doesn't support direct batching, you could simply loop:
        results = []
        for img in images:
            sim = self.backend.calculate_similarity(img, text)
            results.append(sim)
        return results
