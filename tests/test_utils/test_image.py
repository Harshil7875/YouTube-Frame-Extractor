import pytest
from PIL import Image
from youtube_frame_extractor.utils import image

# Assume that utils/image.py defines a function:
# def convert_to_grayscale(img: Image.Image) -> Image.Image:
#     return img.convert("L")

def test_convert_to_grayscale():
    # Create a simple RGB image.
    img = Image.new("RGB", (100, 100), color="red")
    
    # Convert the image to grayscale using the utility function.
    gray_img = image.convert_to_grayscale(img)
    
    # Check if the output image is in grayscale mode ('L').
    assert gray_img.mode == "L"
    # Optionally, verify the image size remains the same.
    assert gray_img.size == img.size
