import pytest
from PIL import Image

from youtube_frame_extractor.analysis.object_detection import ObjectDetector
from youtube_frame_extractor.exceptions import ObjectDetectionError

@pytest.fixture(scope="module")
def blank_image():
    # Create a blank white image (likely with no detectable objects)
    return Image.new("RGB", (224, 224), color="white")

@pytest.fixture(scope="module")
def object_detector():
    # Initialize the ObjectDetector using a common detection model on CPU.
    return ObjectDetector(model_name="fasterrcnn_resnet50_fpn", device="cpu", score_threshold=0.5)

def test_detect_objects_returns_list(object_detector, blank_image):
    results = object_detector.detect_objects(blank_image)
    assert isinstance(results, list)
    # If any detections are made, each detection should have the expected keys.
    for detection in results:
        assert "label" in detection
        assert "score" in detection
        assert "bbox" in detection

def test_detect_objects_invalid_input(object_detector):
    with pytest.raises(ObjectDetectionError):
        object_detector.detect_objects(None)

def test_batch_detect(object_detector, blank_image):
    images = [blank_image, blank_image, None]
    results = object_detector.batch_detect(images)
    # The results list should have the same length as the input images.
    assert len(results) == len(images)
    # For a None image, expect an empty list.
    assert results[2] == []
    # For valid images, each result should be a list (which might be empty or contain detections).
    for i in [0, 1]:
        assert isinstance(results[i], list)
