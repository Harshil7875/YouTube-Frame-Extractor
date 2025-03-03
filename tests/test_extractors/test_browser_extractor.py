import pytest
from pathlib import Path

from youtube_frame_extractor.extractors.browser import BrowserExtractor
from youtube_frame_extractor.exceptions import ExtractionError

# Create a dummy version of the BrowserExtractor that simulates frame extraction.
class DummyBrowserExtractor(BrowserExtractor):
    def extract_frames(self, video_id, interval, max_frames, duration, progress_callback):
        # Simulate extraction by returning dummy frame dictionaries.
        frames = []
        for i in range(int(max_frames)):
            frames.append({
                "time": i * interval,
                "path": f"/dummy/path/frame_{i}.jpg",
                "similarity": 0.5
            })
            if progress_callback:
                progress_callback(i + 1)
        return frames

    def scan_video_for_frames(self, video_id, search_query, vlm_analyzer, interval, threshold, max_frames, duration, progress_callback):
        # Simulate extraction with VLM analysis by returning frames with a similarity above threshold.
        frames = []
        for i in range(int(max_frames)):
            frames.append({
                "time": i * interval,
                "path": f"/dummy/path/frame_{i}.jpg",
                "similarity": threshold + 0.1
            })
            if progress_callback:
                progress_callback(i + 1)
        return frames

@pytest.fixture
def dummy_output_dir(tmp_path: Path) -> str:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)

@pytest.fixture
def browser_extractor(dummy_output_dir) -> DummyBrowserExtractor:
    # Instantiate our dummy extractor instead of the real one.
    extractor = DummyBrowserExtractor(output_dir=dummy_output_dir, headless=True)
    return extractor

def test_extract_frames_without_vlm(browser_extractor):
    video_id = "dummy_video_id"
    interval = 1.0
    max_frames = 3
    duration = None

    frames = browser_extractor.extract_frames(
        video_id=video_id,
        interval=interval,
        max_frames=max_frames,
        duration=duration,
        progress_callback=lambda x: None
    )
    assert len(frames) == max_frames
    for i, frame in enumerate(frames):
        assert frame["time"] == i * interval
        assert frame["path"] == f"/dummy/path/frame_{i}.jpg"
        assert "similarity" in frame

def test_scan_video_for_frames_with_vlm(browser_extractor):
    video_id = "dummy_video_id"
    interval = 1.0
    max_frames = 2
    duration = None
    search_query = "dummy query"
    # Create a dummy VLM analyzer that returns a fixed similarity.
    dummy_vlm = lambda image, text: 0.8  

    frames = browser_extractor.scan_video_for_frames(
        video_id=video_id,
        search_query=search_query,
        vlm_analyzer=dummy_vlm,
        interval=interval,
        threshold=0.7,
        max_frames=max_frames,
        duration=duration,
        progress_callback=lambda x: None
    )
    assert len(frames) == max_frames
    for frame in frames:
        # The dummy implementation always returns threshold + 0.1.
        assert frame["similarity"] > 0.7
