#!/usr/bin/env python3
"""
Video Utility Module for YouTube Frame Extractor

Provides helper functions to:
- Extract basic metadata (duration, resolution) using ffmpeg or OpenCV
- Extract frames via OpenCV (optional)
- Perform simple scene detection or advanced checks if needed
"""

import os
import subprocess
from typing import Optional, Union, List, Dict

try:
    import cv2
except ImportError:
    cv2 = None

import ffmpeg
from PIL import Image

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import UtilityError, FFmpegError, VideoUnavailableError

logger = get_logger(__name__)
settings = get_settings()


def get_video_metadata(video_path: str) -> Dict[str, Union[float, int, str]]:
    """
    Retrieve metadata from a video file using ffmpeg.probe.

    Args:
        video_path: Path to the local video file.

    Returns:
        A dictionary with keys like 'duration', 'width', 'height', 'streams'.

    Raises:
        FFmpegError: If ffmpeg probe fails or the file is invalid.
    """
    if not os.path.exists(video_path):
        msg = f"Video file not found: {video_path}"
        logger.error(msg)
        raise VideoUnavailableError(video_id=video_path, reason="File not found")

    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe.get("format", {})
        duration = float(format_info.get("duration", 0.0))

        # Basic video stream info
        width, height = 0, 0
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                break

        metadata = {
            "duration": duration,
            "width": width,
            "height": height,
            "streams": probe.get("streams", [])
        }
        logger.debug(f"Video metadata for '{video_path}': {metadata}")
        return metadata

    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        logger.error(f"Error probing video file with ffmpeg: {stderr}")
        raise FFmpegError(
            f"Error probing video '{video_path}': {stderr}",
            command="ffprobe"
        )


def extract_frames_opencv(
    video_path: str,
    output_dir: str,
    step_frame: int = 30,
    max_frames: Optional[int] = None,
    start_frame: int = 0
) -> List[str]:
    """
    Extract frames from a local video using OpenCV at a fixed step (every N frames).
    This is an alternative to using ffmpeg directly.

    Args:
        video_path: Path to the local video file.
        output_dir: Directory to save extracted frames.
        step_frame: Extract every Nth frame (default=30).
        max_frames: Limit the total number of frames extracted (None for no limit).
        start_frame: Frame index to start from (defaults to 0).

    Returns:
        A list of file paths for the extracted frames (in chronological order).

    Raises:
        UtilityError: If OpenCV is not installed or the video cannot be read.
    """
    if cv2 is None:
        msg = "OpenCV not installed, cannot extract frames with OpenCV."
        logger.error(msg)
        raise UtilityError(msg)

    if not os.path.exists(video_path):
        msg = f"Video file not found: {video_path}"
        logger.error(msg)
        raise UtilityError(msg)

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"OpenCV failed to open video: {video_path}"
        logger.error(msg)
        raise UtilityError(msg)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"Total frames in '{video_path}': {frame_count}")

    extracted_files = []
    current_frame = start_frame
    extracted = 0

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or read error

        # Current position might not exactly match 'current_frame' but for reference
        if (current_frame - start_frame) % step_frame == 0:
            # Extract and save frame
            frame_filename = f"frame_{current_frame:06d}.jpg"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            extracted_files.append(output_path)
            extracted += 1

            if max_frames is not None and extracted >= max_frames:
                logger.debug(f"Reached max_frames={max_frames}, stopping extraction.")
                break

        current_frame += 1
        if current_frame >= frame_count:
            break

    cap.release()
    logger.info(f"Extracted {len(extracted_files)} frames from '{video_path}' into '{output_dir}'")
    return extracted_files


def scene_change_detection(
    video_path: str,
    threshold: float = 30.0,
    method: str = "mse"
) -> List[int]:
    """
    Perform simple scene change detection by comparing consecutive frames.

    Args:
        video_path: Local video file path.
        threshold: Threshold for difference (MSE or SSIM-based).
        method: "mse" or "ssim". (MSE: higher = more different, SSIM: lower = more different)

    Returns:
        A list of frame indices where scene changes are detected.

    Raises:
        UtilityError: If OpenCV is not installed or video can't be processed.
    """
    if cv2 is None:
        msg = "OpenCV not installed, cannot do scene change detection."
        logger.error(msg)
        raise UtilityError(msg)

    if not os.path.exists(video_path):
        msg = f"Video file not found: {video_path}"
        logger.error(msg)
        raise UtilityError(msg)

    from ..utils.image import compute_mse, compute_ssim, image_to_numpy
    from ..utils.image import convert_to_grayscale

    changes = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise UtilityError(f"Failed to open video for scene detection: {video_path}")

    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        cap.release()
        return changes  # no frames

    # Convert first frame to PIL for consistency in measure
    prev_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_index += 1
        current_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if method.lower() == "mse":
            diff = compute_mse(prev_pil, current_pil)
            # if difference is large, scene change
            if diff >= threshold:
                changes.append(frame_index)
        else:  # ssim
            ssim_val = compute_ssim(prev_pil, current_pil)
            # if SSIM is low => more difference; threshold means if ssim < (some value)
            # interpret 'threshold' differently for SSIM (like 0.7 => scene change)
            # We'll interpret threshold as "scene change if ssim < threshold"
            if 0 <= ssim_val < threshold:
                changes.append(frame_index)

        prev_pil = current_pil

    cap.release()
    logger.info(f"Detected {len(changes)} scene changes in '{video_path}' using method={method}.")
    return changes
