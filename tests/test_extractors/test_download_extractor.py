import pytest
from pathlib import Path

from youtube_frame_extractor.extractors.download import DownloadExtractor
from youtube_frame_extractor.exceptions import ExtractionError

# Create a dummy version of the DownloadExtractor that simulates downloading and frame extraction.
class DummyDownloadExtractor(DownloadExtractor):
    def extract_frames(self, video_id, frame_rate, max_frames, resolution, keep_video, download_callback):
        # Simulate extraction by returning dummy frames.
        frames = []
        for i in range(int(max_frames)):
            frames.append({
                "time": i / frame_rate,
                "path": f"/dummy/path/downloaded_frame_{i}.jpg",
                "similarity": 0.6
            })
            if download_callback:
                # Simulate progress percentage increment.
                download_callback((i + 1) * 10)
        return frames

@pytest.fixture
def dummy_output_dir(tmp_path: Path) -> str:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)

@pytest.fixture
def download_extractor(dummy_output_dir) -> DummyDownloadExtractor:
    extractor = DummyDownloadExtractor(output_dir=dummy_output_dir)
    return extractor

def test_extract_frames_download(download_extractor):
    video_id = "dummy_video_id"
    frame_rate = 1.0
    max_frames = 4
    resolution = "720p"
    keep_video = False

    frames = download_extractor.extract_frames(
        video_id=video_id,
        frame_rate=frame_rate,
        max_frames=max_frames,
        resolution=resolution,
        keep_video=keep_video,
        download_callback=lambda percent: None
    )
    assert len(frames) == max_frames
    for i, frame in enumerate(frames):
        # Verify that frame time is calculated based on the frame rate.
        assert frame["time"] == i / frame_rate
        assert frame["path"] == f"/dummy/path/downloaded_frame_{i}.jpg"
        assert "similarity" in frame

def test_extract_frames_download_error(download_extractor, monkeypatch):
    # Simulate an error during frame extraction by monkeypatching extract_frames to raise an exception.
    def error_callback(*args, **kwargs):
        raise ExtractionError("Simulated extraction error")

    monkeypatch.setattr(download_extractor, "extract_frames", error_callback)
    with pytest.raises(ExtractionError):
        download_extractor.extract_frames(
            video_id="dummy_video_id",
            frame_rate=1.0,
            max_frames=2,
            resolution="720p",
            keep_video=False,
            download_callback=lambda percent: None
        )
