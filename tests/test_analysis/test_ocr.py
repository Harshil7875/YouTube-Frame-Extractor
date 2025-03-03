import pytest
from PIL import Image, ImageDraw
from youtube_frame_extractor.analysis.ocr import OCRAnalyzer
from youtube_frame_extractor.exceptions import OCRError

@pytest.fixture(scope="module")
def text_image():
    # Create a simple image with text "Test OCR" using PIL's drawing functions.
    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    text = "Test OCR"
    # Position the text roughly in the middle.
    draw.text((10, 40), text, fill="black")
    return img

@pytest.fixture(scope="module")
def ocr_analyzer():
    # Initialize OCRAnalyzer with default parameters.
    return OCRAnalyzer(lang="eng", psm=6)

def test_extract_text_from_image(ocr_analyzer, text_image):
    extracted_text = ocr_analyzer.extract_text_from_image(text_image)
    # The OCR output should contain either "Test" or "OCR" (or both).
    assert "Test" in extracted_text or "OCR" in extracted_text

def test_extract_text_from_invalid_image(ocr_analyzer):
    with pytest.raises(OCRError):
        ocr_analyzer.extract_text_from_image(None)

def test_batch_extract_text(ocr_analyzer, text_image):
    images = [text_image, text_image, None]
    results = ocr_analyzer.batch_extract_text(images)
    assert len(results) == 3
    # Valid images should produce non-empty results.
    assert results[0] != ""
    assert results[1] != ""
    # For a None image, expect an empty string.
    assert results[2] == ""

def test_extract_data_output_type(ocr_analyzer, text_image):
    # Test extraction of structured OCR data using output type "dict"
    data = ocr_analyzer.extract_data(text_image, output_type="dict")
    # Ensure the output is a dictionary and contains typical OCR keys like "text".
    assert isinstance(data, dict)
    assert "text" in data  # pytesseract returns a dict with keys like "level", "page_num", "text", etc.
