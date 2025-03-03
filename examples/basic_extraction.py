#!/usr/bin/env python3
"""
Basic Frame Extraction Example

This example demonstrates how to extract frames from a YouTube video
using both browser-based and download-based methods.

Usage:
    python examples/basic_extraction.py --video-id dQw4w9WgXcQ --method browser --interval 2 --frames 10
    python examples/basic_extraction.py --video-id dQw4w9WgXcQ --method download --frame-rate 0.5
"""

import argparse
import logging
import os
import sys
print("sys.path:", sys.path)

from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to sys.path to import the package modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from youtube_frame_extractor.extractors.browser import BrowserExtractor
    from youtube_frame_extractor.extractors.download import DownloadExtractor
except ImportError as e:
    print("ERROR: Could not import YouTube Frame Extractor package modules.")
    print("Exception:", e)
    print("Make sure you're running this script from the project root directory and that PYTHONPATH includes the 'src' directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("basic_extraction")


def extract_frames_browser(
    video_id: str, output_dir: str, interval: float, max_frames: int
) -> List[Dict[str, Any]]:
    """
    Extract frames using the browser-based method.
    
    Args:
        video_id: YouTube video ID.
        output_dir: Directory to save extracted frames.
        interval: Interval between frame captures in seconds.
        max_frames: Maximum number of frames to extract.
        
    Returns:
        List of dictionaries containing frame data and metadata.
    """
    logger.info(f"Starting browser-based extraction for video: {video_id}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        extractor = BrowserExtractor(output_dir=output_dir, headless=True)
        frames = extractor.extract_frames(
            video_id=video_id,
            interval=interval,
            max_frames=max_frames
        )
        logger.info(f"Browser extraction completed: {len(frames)} frames extracted.")
        return frames
    except Exception as e:
        logger.error(f"Error during browser extraction: {str(e)}")
        return []


def extract_frames_download(
    video_id: str, output_dir: str, frame_rate: float, max_frames: int
) -> List[Dict[str, Any]]:
    """
    Extract frames using the download-based method.
    
    Args:
        video_id: YouTube video ID.
        output_dir: Directory to save extracted frames.
        frame_rate: Number of frames to extract per second.
        max_frames: Maximum number of frames to extract.
        
    Returns:
        List of dictionaries containing frame data and metadata.
    """
    logger.info(f"Starting download-based extraction for video: {video_id}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        extractor = DownloadExtractor(output_dir=output_dir)
        frames = extractor.extract_frames(
            video_id=video_id,
            frame_rate=frame_rate,
            max_frames=max_frames
        )
        logger.info(f"Download extraction completed: {len(frames)} frames extracted.")
        return frames
    except Exception as e:
        logger.error(f"Error during download extraction: {str(e)}")
        return []


def display_results(frames: List[Dict[str, Any]], video_id: str) -> None:
    """
    Display information about the extracted frames.
    
    Args:
        frames: List of frame data dictionaries.
        video_id: YouTube video ID.
    """
    if not frames:
        logger.warning("No frames were extracted.")
        return
    
    logger.info(f"Extracted {len(frames)} frames from video {video_id}")
    
    for i, frame_data in enumerate(frames[:5]):
        time_info = f"Time: {frame_data.get('time', 'N/A')}s"
        logger.info(f"Frame {i + 1}: {frame_data.get('path', 'N/A')} ({time_info})")
    
    if len(frames) > 5:
        logger.info(f"... and {len(frames) - 5} more frames.")
    
    # Determine the output directory from the first frame's path
    frame_dir = os.path.abspath(frames[0]['path']).rsplit(os.sep, 1)[0]
    logger.info(f"Frames saved in directory: {frame_dir}")


def main() -> None:
    """
    Main entry point for the basic extraction example.
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from a YouTube video using browser or download methods."
    )
    parser.add_argument("--video-id", type=str, required=True, help="YouTube video ID")
    parser.add_argument(
        "--method",
        choices=["browser", "download"],
        default="browser",
        help="Frame extraction method to use (default: browser)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./example_output",
        help="Directory to save extracted frames (default: ./example_output)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Interval between frames for browser-based extraction (seconds)"
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=0.5,
        help="Frames per second for download-based extraction"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Maximum number of frames to extract"
    )
    
    args = parser.parse_args()
    
    if args.method == "browser":
        frames = extract_frames_browser(
            video_id=args.video_id,
            output_dir=args.output_dir,
            interval=args.interval,
            max_frames=args.frames
        )
    else:
        frames = extract_frames_download(
            video_id=args.video_id,
            output_dir=args.output_dir,
            frame_rate=args.frame_rate,
            max_frames=args.frames
        )
    
    display_results(frames, args.video_id)

if __name__ == "__main__":
    main()
