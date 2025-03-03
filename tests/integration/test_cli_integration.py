import json
import os
import pytest
from pathlib import Path
from typer.testing import CliRunner

# Import the CLI app from the package.
from youtube_frame_extractor import cli

runner = CliRunner()


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Fixture to create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def load_metadata(metadata_path: Path) -> dict:
    """Helper function to load and return JSON metadata."""
    with open(metadata_path, "r") as f:
        return json.load(f)


def test_browser_extraction_cli(temp_output_dir: Path):
    """
    Integration test for the browser-based extraction CLI command.
    It invokes the command with a known YouTube video id and verifies
    that frames are (at least partially) processed and metadata is saved.
    """
    video_id = "dQw4w9WgXcQ"
    max_frames = "2"
    interval = "1.0"
    
    result = runner.invoke(
        cli.app,
        [
            "browser",
            video_id,
            "--output-dir",
            str(temp_output_dir),
            "--max-frames",
            max_frames,
            "--interval",
            interval,
        ],
    )
    # Ensure the command finished successfully.
    assert result.exit_code == 0, result.output

    # Verify that metadata was saved.
    metadata_file = temp_output_dir / f"{video_id}_metadata.json"
    assert metadata_file.exists(), "Metadata file should exist after browser extraction."
    
    metadata = load_metadata(metadata_file)
    assert metadata.get("video_id") == video_id
    assert metadata.get("extraction_method") == "browser"
    assert "frame_count" in metadata


def test_download_extraction_cli(temp_output_dir: Path):
    """
    Integration test for the download-based extraction CLI command.
    It invokes the command and checks that the metadata is generated correctly.
    """
    video_id = "dQw4w9WgXcQ"
    max_frames = "2"
    frame_rate = "1.0"
    
    result = runner.invoke(
        cli.app,
        [
            "download",
            video_id,
            "--output-dir",
            str(temp_output_dir),
            "--max-frames",
            max_frames,
            "--frame-rate",
            frame_rate,
        ],
    )
    # Ensure the command finished successfully.
    assert result.exit_code == 0, result.output

    metadata_file = temp_output_dir / f"{video_id}_metadata.json"
    assert metadata_file.exists(), "Metadata file should exist after download extraction."
    
    metadata = load_metadata(metadata_file)
    assert metadata.get("video_id") == video_id
    assert metadata.get("extraction_method") == "download"
    assert "frame_count" in metadata


def test_batch_processing_cli(tmp_path: Path):
    """
    Integration test for the batch processing CLI command.
    It submits two video IDs for processing and verifies that each video's output
    directory contains a metadata file.
    """
    output_dir = tmp_path / "batch_output"
    output_dir.mkdir()
    video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0"]
    
    # Join video IDs into a single argument (Typer supports multiple arguments)
    args = ["batch"] + video_ids + [
        "--output-dir",
        str(output_dir),
        "--max-frames",
        "1",
        "--workers",
        "1",
    ]
    
    result = runner.invoke(cli.app, args)
    # Ensure the command finished successfully.
    assert result.exit_code == 0, result.output

    # For each video, verify that the metadata file exists.
    for vid in video_ids:
        video_dir = output_dir / vid
        metadata_file = video_dir / f"{vid}_metadata.json"
        assert metadata_file.exists(), f"Metadata file for video {vid} should exist."
        
        metadata = load_metadata(metadata_file)
        assert metadata.get("video_id") == vid
        # Either extraction was successful or a proper error message was recorded.
        assert "frame_count" in metadata


def test_vlm_analysis_cli(temp_output_dir: Path):
    """
    Integration test for the VLM-based analysis CLI command.
    It verifies that invoking the VLM command produces metadata with the correct structure.
    """
    video_id = "dQw4w9WgXcQ"
    query = "person singing"
    max_frames = "2"
    
    result = runner.invoke(
        cli.app,
        [
            "vlm",
            video_id,
            query,
            "--output-dir",
            str(temp_output_dir),
            "--max-frames",
            max_frames,
        ],
    )
    # Ensure the command finished successfully.
    assert result.exit_code == 0, result.output

    metadata_file = temp_output_dir / f"{video_id}_metadata.json"
    assert metadata_file.exists(), "Metadata file should exist after VLM analysis."

    metadata = load_metadata(metadata_file)
    assert metadata.get("video_id") == video_id
    # VLM analysis should include the query in the metadata if provided.
    if "query" in metadata:
        assert metadata["query"] == query
