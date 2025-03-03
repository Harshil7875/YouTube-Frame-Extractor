#!/usr/bin/env python3
"""
OCR (Optical Character Recognition) Module for YouTube Frame Extractor

This module provides functionality to extract text from images using Tesseract.
It wraps pytesseract for easy integration into the broader framework.
"""

import os
from typing import Optional, List, Dict, Any

import pytesseract
from PIL import Image

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import OCRError

logger = get_logger(__name__)
settings = get_settings()


class OCRAnalyzer:
    """
    A class that provides OCR (Optical Character Recognition) functionality
    using Tesseract via pytesseract.

    Example usage:
        analyzer = OCRAnalyzer(lang="eng", psm=3)
        text = analyzer.extract_text_from_image(pil_image)
    """

    def __init__(self, lang: str = "eng", psm: int = 3, oem: int = 3):
        """
        Initialize the OCR analyzer.

        Args:
            lang: Language code (e.g., 'eng', 'fra'). Defaults to 'eng'.
            psm: Page segmentation mode. Defaults to 3 (Fully automatic).
            oem: OCR Engine mode. Defaults to 3 (Default, based on what is available).
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem

        # If TESSERACT_CMD is set in environment or config, use that
        # pytesseract.pytesseract.tesseract_cmd = ...
        # You can do something like:
        tesseract_cmd = os.environ.get("TESSERACT_CMD", "")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        logger.info(f"OCRAnalyzer initialized with lang='{self.lang}', psm={self.psm}, oem={self.oem}")

    def extract_text_from_image(self, image: Image.Image, config_params: Optional[str] = None) -> str:
        """
        Extract text from a single PIL image.

        Args:
            image: A PIL Image from which to extract text.
            config_params: Additional configuration string for Tesseract (optional).

        Returns:
            A string containing the recognized text.

        Raises:
            OCRError: If OCR fails or no text is extracted.
        """
        if not image:
            raise OCRError("No image provided for OCR")

        # Build the Tesseract config
        # Example of a typical config: f"--psm {self.psm} --oem {self.oem}"
        config_str = config_params or f"--psm {self.psm} --oem {self.oem}"

        # Add language
        lang_code = self.lang

        try:
            logger.debug(f"Running OCR with config='{config_str}', lang='{lang_code}'")
            text = pytesseract.image_to_string(image, lang=lang_code, config=config_str)
            logger.debug(f"OCR output (trimmed): {text[:60]!r} ...")

            return text.strip() if text else ""

        except Exception as e:
            logger.error(f"Error during OCR: {str(e)}")
            raise OCRError(f"Error during OCR: {str(e)}")

    def batch_extract_text(
        self, 
        images: List[Image.Image], 
        config_params: Optional[str] = None
    ) -> List[str]:
        """
        Extract text from a batch of images in a single loop.

        Args:
            images: A list of PIL Images.
            config_params: Additional configuration string for Tesseract (optional).

        Returns:
            A list of recognized text strings, one for each image.
        """
        if not images:
            return []

        results = []
        for idx, img in enumerate(images):
            if img is None:
                results.append("")
                continue

            try:
                text = self.extract_text_from_image(img, config_params=config_params)
                results.append(text)
            except OCRError as e:
                logger.warning(f"OCR failed for image index {idx}: {str(e)}")
                results.append("")

        return results

    def extract_data(self, image: Image.Image, output_type: str = "dict", config_params: Optional[str] = None) -> Any:
        """
        Extract structured data from an image, such as bounding boxes or HOCR data.

        Args:
            image: A PIL Image.
            output_type: One of the PyTesseract Output types, e.g. 'dict', 'data.frame', 'string', 'hocr', 'alto'.
            config_params: Additional Tesseract config.

        Returns:
            The extracted data in the specified format (dict, string, etc.).

        Raises:
            OCRError: If extraction fails.
        """
        import pytesseract as pt
        from pytesseract import Output

        # Map string to Output.<type> if needed
        output_map = {
            "dict": Output.DICT,
            "data.frame": Output.DATAFRAME,
            "string": None,      # The default 'string' is image_to_string
            "hocr": Output.HOCR,
            "alto": Output.ALTO
        }
        if output_type not in output_map:
            raise OCRError(f"Unsupported output type: {output_type}")

        if not image:
            raise OCRError("No image provided for structured OCR data extraction")

        config_str = config_params or f"--psm {self.psm} --oem {self.oem}"
        lang_code = self.lang

        try:
            logger.debug(f"Extracting OCR data with output_type={output_type}, lang={lang_code}")
            if output_type == "string":
                return pt.image_to_string(image, lang=lang_code, config=config_str)
            else:
                ocr_data = pt.image_to_data(
                    image,
                    lang=lang_code,
                    config=config_str,
                    output_type=output_map[output_type]
                )
                return ocr_data

        except Exception as e:
            logger.error(f"Error extracting OCR data: {str(e)}")
            raise OCRError(f"Error extracting OCR data: {str(e)}")
