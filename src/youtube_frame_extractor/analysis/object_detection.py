#!/usr/bin/env python3
"""
Object Detection for YouTube Frame Extractor

This module provides a class to perform object detection on frames
using a pre-trained model (e.g., Faster R-CNN) from torchvision.
"""

import torch
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn
)
from PIL import Image

from ..exceptions import ObjectDetectionError, ModelLoadError
from ..config import get_settings
from ..logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ObjectDetector:
    """
    A wrapper for a pre-trained object detection model (e.g., Faster R-CNN).
    It predicts bounding boxes, labels, and confidence scores for objects in an image.

    Example usage:
        detector = ObjectDetector(model_name="fasterrcnn_resnet50_fpn")
        results = detector.detect_objects(pil_image, score_threshold=0.5)
    """

    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn", device: str = None, score_threshold: float = 0.5):
        """
        Initialize the detector with a chosen pre-trained model.

        Args:
            model_name: Name of the pre-trained detection model
                       ("fasterrcnn_resnet50_fpn" or "fasterrcnn_mobilenet_v3_large_fpn", etc.)
            device: "cuda" or "cpu". If None, uses settings.vlm.device or fallback to CPU.
            score_threshold: Default confidence score threshold to filter predictions.
        """
        self.model_name = model_name
        self.device = device or settings.vlm.device
        self.score_threshold = score_threshold

        logger.info(f"Initializing ObjectDetector with model '{model_name}' on device '{self.device}'")

        try:
            # Load a standard torchvision detection model
            if model_name.lower() == "fasterrcnn_resnet50_fpn":
                self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            elif model_name.lower() == "fasterrcnn_mobilenet_v3_large_fpn":
                self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            self.model.eval()
            self.model.to(self.device)

        except Exception as e:
            logger.error(f"Error loading object detection model '{model_name}': {str(e)}")
            raise ModelLoadError(model_name=model_name, reason=str(e))

        logger.info(f"Successfully loaded object detection model '{model_name}'")

        # Define a standard transform (you may adjust for your pipeline)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect_objects(self, image: Image.Image, score_threshold: float = None):
        """
        Run inference on a single PIL image to detect objects.

        Args:
            image: A PIL Image to analyze.
            score_threshold: Optional threshold for confidence scores. Defaults to self.score_threshold.

        Returns:
            A list of dictionaries, each containing:
                - "label": Integer label ID (or class name if you map it)
                - "score": Confidence score (float)
                - "bbox": [x_min, y_min, x_max, y_max]

        Raises:
            ObjectDetectionError: If inference fails or inputs are invalid.
        """
        if not image:
            raise ObjectDetectionError("No image provided for object detection")

        threshold = score_threshold if score_threshold is not None else self.score_threshold

        try:
            # Preprocess
            tensor_img = self.transform(image).to(self.device).unsqueeze(0)  # shape: [1, C, H, W]

            with torch.no_grad():
                outputs = self.model(tensor_img)

            # outputs is a list of dicts (for each image)
            predictions = outputs[0]

            # Filter by score
            filtered_results = []
            for label, score, bbox in zip(
                predictions["labels"],
                predictions["scores"],
                predictions["boxes"]
            ):
                if score >= threshold:
                    result = {
                        "label": label.item(),
                        "score": float(score.item()),
                        "bbox": [float(x) for x in bbox.tolist()]
                    }
                    filtered_results.append(result)

            logger.debug(f"Raw detections: {len(predictions['labels'])}, "
                         f"passing threshold {threshold}: {len(filtered_results)}")

            return filtered_results

        except Exception as e:
            logger.error(f"Error running object detection: {str(e)}")
            raise ObjectDetectionError(f"Error running object detection: {str(e)}")

    def batch_detect(self, images, score_threshold: float = None):
        """
        Run inference on a batch of images at once.

        Args:
            images: A list of PIL Images.
            score_threshold: Optional threshold for confidence scores.

        Returns:
            A list (same length as images) of lists of detection dictionaries.
        """
        if not images:
            return []

        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Preprocess all images together
        batch_tensors = []
        for img in images:
            if not img:
                batch_tensors.append(None)
                continue
            batch_tensors.append(self.transform(img))

        # Filter out None images
        valid_indices = [i for i, t in enumerate(batch_tensors) if t is not None]
        valid_tensors = [batch_tensors[i] for i in valid_indices]

        if not valid_tensors:
            return [[] for _ in images]  # no valid images, return empty results

        try:
            input_batch = torch.stack(valid_tensors).to(self.device)  # shape: [N, C, H, W]
            with torch.no_grad():
                outputs = self.model(input_batch)

            # Construct final results
            results = [[] for _ in images]
            for idx, output in zip(valid_indices, outputs):
                filtered_results = []
                for label, score, bbox in zip(
                    output["labels"], output["scores"], output["boxes"]
                ):
                    if score >= threshold:
                        filtered_results.append({
                            "label": label.item(),
                            "score": float(score.item()),
                            "bbox": [float(x) for x in bbox.tolist()]
                        })

                results[idx] = filtered_results

            return results

        except Exception as e:
            logger.error(f"Error running batch object detection: {str(e)}")
            raise ObjectDetectionError(f"Error in batch object detection: {str(e)}")
