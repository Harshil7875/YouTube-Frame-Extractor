import pytest
from PIL import Image

from youtube_frame_extractor.analysis.clip import CLIPAnalyzer
from youtube_frame_extractor.exceptions import VLMError

@pytest.fixture(scope="module")
def dummy_image():
    # Create a simple white image of size 224x224
    return Image.new("RGB", (224, 224), color="white")

@pytest.fixture(scope="module")
def clip_analyzer():
    # Create a CLIPAnalyzer instance using the base model on CPU for testing.
    return CLIPAnalyzer(model_name="openai/clip-vit-base-patch16", device="cpu")

def test_calculate_similarity_returns_float(clip_analyzer, dummy_image):
    query = "a plain white image"
    similarity = clip_analyzer.calculate_similarity(dummy_image, query)
    assert isinstance(similarity, float)
    # Cosine similarity should be between -1 and 1.
    assert -1.0 <= similarity <= 1.0

def test_calculate_similarity_invalid_image(clip_analyzer):
    query = "any text"
    with pytest.raises(VLMError):
        clip_analyzer.calculate_similarity(None, query)

def test_calculate_similarity_invalid_query(clip_analyzer, dummy_image):
    with pytest.raises(VLMError):
        clip_analyzer.calculate_similarity(dummy_image, "")
