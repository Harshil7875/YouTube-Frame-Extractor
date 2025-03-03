#!/usr/bin/env python3
"""
CLIP-based Analysis for YouTube Frame Extractor

This module provides an analyzer class for OpenAI's CLIP model,
allowing you to calculate similarity scores between text prompts
and images (PIL format).
"""

import os
import torch
from PIL import Image

try:
    import clip  # Official CLIP library from GitHub: https://github.com/openai/CLIP
except ImportError:
    raise ImportError(
        "CLIP library not found. Please install it from GitHub:\n"
        "pip install git+https://github.com/openai/CLIP.git"
    )

from ..exceptions import ModelLoadError, VLMError
from ..config import get_settings
from ..logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CLIPAnalyzer:
    """
    A wrapper for the OpenAI CLIP model that calculates text-image similarity scores.

    Usage:
        analyzer = CLIPAnalyzer(model_name="openai/clip-vit-large-patch14")
        similarity = analyzer.calculate_similarity(pil_image, "A photo of a cat")
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the CLIP model and its corresponding tokenizer.

        Args:
            model_name: Name of the CLIP model variant to load (e.g. "openai/clip-vit-base-patch16").
                       If None, it uses the default model from settings.vlm.default_model.
            device: "cuda" or "cpu". If None, it uses settings.vlm.device.
        """
        self.model_name = model_name or settings.vlm.default_model.value
        self.device = device or settings.vlm.device

        logger.info(f"Initializing CLIPAnalyzer with model '{self.model_name}' on device '{self.device}'")

        try:
            # Load the model and preprocess from the official CLIP repo
            self.model, self.preprocess = clip.load(self.model_name, device=self.device, download_root=settings.vlm.cache_dir)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model '{self.model_name}': {str(e)}")
            raise ModelLoadError(model_name=self.model_name, reason=str(e))

        logger.info(f"Successfully loaded CLIP model '{self.model_name}'")

    def calculate_similarity(self, image: Image.Image, text: str) -> float:
        """
        Calculate the similarity score between a PIL image and a text prompt.

        Args:
            image: A PIL Image.
            text: A text string to compare against.

        Returns:
            A float indicating the cosine similarity (range roughly -1.0 to +1.0).
        """
        if image is None:
            raise VLMError("No image provided for CLIP analysis", model_name=self.model_name)
        if not text:
            raise VLMError("No text prompt provided for CLIP analysis", model_name=self.model_name)

        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # shape: [1, 3, H, W]

            # Tokenize text
            text_tokens = clip.tokenize([text]).to(self.device)  # shape: [1, token_length]

            # Forward pass in CLIP
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = (image_features * text_features).sum().item()

            return similarity

        except Exception as e:
            logger.error(f"Error during CLIP analysis: {str(e)}")
            raise VLMError(f"Error during CLIP analysis: {str(e)}", model_name=self.model_name)
