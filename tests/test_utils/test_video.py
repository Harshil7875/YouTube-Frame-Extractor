import pytest
from youtube_frame_extractor.utils import video

# Dummy helper to simulate a frame.
def dummy_frame(time):
    return {"time": time, "data": "dummy"}

def test_detect_scene_changes():
    # Create dummy frames with constant content (simulate no scene changes).
    frames = [dummy_frame(t) for t in range(10)]
    
    # Assume detect_scene_changes returns a list of frame indices where scene changes occur.
    # With dummy identical frames, we expect no scene changes if threshold is set high.
    scene_changes = video.detect_scene_changes(frames, method="mse", threshold=1000.0)
    
    # For our dummy data, scene_changes should be an empty list.
    assert isinstance(scene_changes, list)
    assert len(scene_changes) == 0

def test_extract_scene_keyframes():
    # Create dummy frames.
    frames = [{"time": i} for i in range(5)]
    # Simulate scene change indices.
    scene_change_indices = [1, 3]
    
    # Assume extract_scene_keyframes returns the frames corresponding to the provided indices.
    key_frames = video.extract_scene_keyframes(frames, scene_change_indices)
    
    # Assert that the number of key frames equals the number of scene changes.
    assert len(key_frames) == len(scene_change_indices)
    
    # Check that each key frame's time matches the expected frame.
    for key_frame, idx in zip(key_frames, scene_change_indices):
        assert key_frame["time"] == frames[idx]["time"]
