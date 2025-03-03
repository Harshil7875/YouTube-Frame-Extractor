#!/usr/bin/env python3
"""
Download-based YouTube Frame Extractor

This module implements a frame extractor that downloads YouTube videos
using yt-dlp and extracts frames at specified intervals using ffmpeg.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import time
import json

import ffmpeg
import numpy as np
from PIL import Image

try:
    import yt_dlp as ytdlp
except ImportError:
    # Fallback to youtube-dl, but with warning
    try:
        import youtube_dl as ytdlp
        import warnings
        warnings.warn(
            "yt-dlp not found, falling back to youtube-dl. Some features may not work correctly. "
            "Consider installing yt-dlp for better performance and fewer issues with rate limiting.",
            ImportWarning
        )
    except ImportError:
        raise ImportError(
            "Neither yt-dlp nor youtube-dl is installed. Please install yt-dlp with: pip install yt-dlp"
        )

from .base import BaseExtractor
from ..exceptions import (
    DownloadExtractionError, VideoUnavailableError, 
    FFmpegError, YtDlpError, VideoPrivateError,
    VideoGeoBlockedError
)
from ..logger import get_logger
from ..config import get_settings

# Get logger
logger = get_logger(__name__)
settings = get_settings()

class YtdlpLogger:
    """
    Logger for yt-dlp progress messages.
    Converts yt-dlp callbacks to our own callback format.
    """
    
    def __init__(self, download_callback: Optional[Callable[[float], None]] = None):
        """
        Initialize the logger.
        
        Args:
            download_callback: Optional callback function that receives download progress (0-100)
        """
        self.download_callback = download_callback
        self.last_percent = 0
    
    def debug(self, msg):
        """Handle debug messages."""
        if msg.startswith('[debug]'):
            logger.debug(msg)
    
    def info(self, msg):
        """Handle info messages."""
        if msg.startswith('[info]'):
            logger.info(msg[7:])  # Strip [info] prefix
    
    def warning(self, msg):
        """Handle warning messages."""
        if msg.startswith('[warning]'):
            logger.warning(msg[10:])  # Strip [warning] prefix
    
    def error(self, msg):
        """Handle error messages."""
        if msg.startswith('[error]'):
            logger.error(msg[8:])  # Strip [error] prefix
        else:
            logger.error(msg)
    
    def progress_hook(self, d):
        """
        Process download progress information.
        
        Args:
            d: Progress dictionary from yt-dlp
        """
        if d['status'] == 'downloading' and self.download_callback:
            if 'total_bytes' in d and d['total_bytes'] > 0:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
            elif 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
            else:
                # If we can't calculate percent, just report -1
                percent = -1
            
            # Only call callback if progress has changed significantly (avoid excessive updates)
            if percent >= 0 and (abs(percent - self.last_percent) >= 1 or percent >= 100):
                self.download_callback(percent)
                self.last_percent = percent
        
        elif d['status'] == 'finished' and self.download_callback:
            # Mark as 100% when download finishes
            self.download_callback(100)


class DownloadExtractor(BaseExtractor):
    """
    Download-based frame extraction from YouTube videos.
    This approach downloads the entire video and extracts frames using ffmpeg.
    """
    
    def __init__(self, output_dir: str = "output", temp_dir: Optional[str] = None):
        """
        Initialize the download-based extractor.
        
        Args:
            output_dir: Directory to save extracted frames
            temp_dir: Directory for temporary downloaded videos
        """
        super().__init__(output_dir)
        
        # Set temp directory - use system temp if not specified
        if temp_dir:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
        else:
            self.temp_dir = settings.download.temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"Initialized download-based extractor with temp directory: {self.temp_dir}")
        
        # Check for ffmpeg
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> None:
        """
        Check if ffmpeg is available in the system path.
        Raises an exception if not found.
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                logger.warning("ffmpeg check returned non-zero exit code, but continuing anyway")
        except FileNotFoundError:
            logger.error("ffmpeg not found in system path")
            raise FFmpegError(
                "ffmpeg not found. Please install ffmpeg and make sure it's in your system path."
            )
    
    def download_video(
        self, 
        video_id: str, 
        output_path: Optional[str] = None,
        resolution: str = "720p",
        download_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Download a YouTube video using yt-dlp.
        
        Args:
            video_id: YouTube video ID
            output_path: Path to save the downloaded video
            resolution: Preferred video resolution
            download_callback: Optional callback function for progress updates
            
        Returns:
            Path to the downloaded video file
        """
        video_url = self.get_video_url(video_id)
        
        # Create a temporary directory if output_path not specified
        temp_dir = None
        if not output_path:
            temp_dir = tempfile.mkdtemp(dir=self.temp_dir)
            output_path = os.path.join(temp_dir, f"{video_id}.mp4")
        
        # Set up yt-dlp options
        ytdlp_logger = YtdlpLogger(download_callback)
        
        format_spec = settings.download.default_format
        if resolution:
            if resolution.lower() == "720p":
                format_spec = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best"
            elif resolution.lower() == "1080p":
                format_spec = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[ext=mp4]/best"
            elif resolution.lower() == "480p":
                format_spec = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[ext=mp4]/best"
            elif resolution.lower() == "360p":
                format_spec = "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best[ext=mp4]/best"
        
        ydl_opts = {
            'format': format_spec,
            'outtmpl': output_path,
            'logger': ytdlp_logger,
            'progress_hooks': [ytdlp_logger.progress_hook],
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            with ytdlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first to catch unavailable videos early
                info = ydl.extract_info(video_url, download=False)
                
                if info.get('is_live', False):
                    raise VideoUnavailableError(
                        video_id, 
                        reason="Cannot extract frames from live videos"
                    )
                
                # Download the video
                logger.info(f"Downloading video {video_id} from YouTube")
                ydl.download([video_url])
                
                # Verify the downloaded file exists and is not empty
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise DownloadExtractionError(
                        f"Download completed but video file is missing or empty: {output_path}", 
                        video_id=video_id
                    )
                
                logger.info(f"Successfully downloaded video to {output_path}")
                return output_path
                
        except ytdlp.utils.DownloadError as e:
            error_message = str(e).lower()
            
            # Handle specific error types
            if "private video" in error_message:
                raise VideoPrivateError(video_id)
            elif "not available in your country" in error_message:
                raise VideoGeoBlockedError(video_id)
            elif "copyright infringement" in error_message:
                raise VideoUnavailableError(video_id, reason="Video removed due to copyright claim")
            elif "this video has been removed" in error_message:
                raise VideoUnavailableError(video_id, reason="Video has been removed")
            else:
                # Generic download error
                raise YtDlpError(
                    f"Failed to download video {video_id}: {str(e)}",
                    video_id=video_id,
                    error_output=str(e)
                )
                
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            raise DownloadExtractionError(
                f"Error downloading video {video_id}: {str(e)}",
                video_id=video_id,
                context={'error': str(e)}
            )
    
    def extract_frames_from_video(
        self,
        video_path: str,
        frame_rate: float = 1.0,
        max_frames: int = 100,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from a video file using ffmpeg.
        
        Args:
            video_path: Path to the video file
            frame_rate: Number of frames to extract per second
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end of video)
            output_dir: Directory to save extracted frames
            progress_callback: Optional callback function that receives the frame number
            
        Returns:
            List of dictionaries containing frame data and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not output_dir:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration using ffprobe
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            logger.info(f"Video duration: {duration:.2f} seconds")
        except ffmpeg.Error as e:
            logger.error(f"Error probing video file: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            raise FFmpegError(
                f"Error analyzing video file: {str(e)}",
                command="ffprobe"
            )
        
        # Calculate parameters
        start_time = start_time or 0
        end_time = min(end_time or duration, duration)
        
        # Calculate total frames based on duration and frame rate
        total_possible_frames = int((end_time - start_time) * frame_rate)
        num_frames = min(max_frames, total_possible_frames) if max_frames > 0 else total_possible_frames
        
        if num_frames <= 0:
            logger.warning(f"No frames to extract (duration: {duration:.2f}s, frame rate: {frame_rate} fps)")
            return []
        
        # Calculate actual frame interval to get evenly distributed frames
        frame_interval = (end_time - start_time) / num_frames
        
        logger.info(f"Extracting {num_frames} frames at {frame_rate} fps")
        
        # Create a temp directory for the extracted frames
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_frames_dir:
            try:
                # Build ffmpeg command
                output_pattern = os.path.join(temp_frames_dir, "frame_%04d.jpg")
                
                # Adjust actual fps based on frame_interval
                if frame_rate > 0:
                    fps_option = f"fps={frame_rate}"
                else:
                    fps_option = f"fps=1/{frame_interval}"
                
                # Build the ffmpeg command for extracting frames
                ffmpeg_cmd = (
                    ffmpeg
                    .input(video_path, ss=start_time, to=end_time)
                    .filter('fps', fps_option)
                    .output(output_pattern, start_number=1, **{'q:v': 2})  # q:v sets quality (2 is high quality)
                    .global_args('-loglevel', 'error', '-y')
                )
                
                # Run the command
                ffmpeg_cmd.run(capture_stdout=True, capture_stderr=True)
                
                # Collect and process the extracted frames
                frames = []
                frame_paths = sorted([
                    os.path.join(temp_frames_dir, f) 
                    for f in os.listdir(temp_frames_dir) 
                    if f.startswith("frame_") and f.endswith(".jpg")
                ])
                
                # Limit to max_frames if specified
                if max_frames > 0 and len(frame_paths) > max_frames:
                    # Select frames evenly distributed across the range
                    indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
                    frame_paths = [frame_paths[i] for i in indices]
                
                # Process each frame
                for i, frame_path in enumerate(frame_paths):
                    # Calculate approximate timestamp based on frame number and interval
                    timestamp = start_time + (i * frame_interval)
                    
                    # Open the image
                    try:
                        image = Image.open(frame_path)
                        
                        # Create output filename
                        output_filename = f"frame_{timestamp:.2f}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Save the image
                        image.save(output_path, "JPEG")
                        
                        # Create frame data
                        frame_data = {
                            "frame": image,
                            "path": output_path,
                            "time": timestamp,
                            "index": i
                        }
                        
                        frames.append(frame_data)
                        
                        # Call progress callback if provided
                        if progress_callback and i % 5 == 0:  # Update every 5 frames to avoid too many callbacks
                            progress_callback(i)
                    
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_path}: {str(e)}")
                
                logger.info(f"Successfully extracted {len(frames)} frames from {video_path}")
                return frames
            
            except ffmpeg.Error as e:
                stderr = e.stderr.decode() if hasattr(e, 'stderr') else "Unknown error"
                logger.error(f"FFmpeg error: {stderr}")
                raise FFmpegError(
                    f"Error extracting frames: {stderr}",
                    command=str(ffmpeg_cmd.compile())
                )
            
            except Exception as e:
                logger.error(f"Error extracting frames: {str(e)}")
                raise DownloadExtractionError(f"Error extracting frames: {str(e)}")
    
    def extract_frames(
        self,
        video_id: str,
        frame_rate: float = 1.0,
        max_frames: int = 100,
        resolution: str = "720p",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        keep_video: bool = False,
        download_callback: Optional[Callable[[float], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Download a YouTube video and extract frames at specified intervals.
        
        Args:
            video_id: YouTube video ID
            frame_rate: Number of frames to extract per second
            max_frames: Maximum number of frames to extract
            resolution: Video resolution to download
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)
            keep_video: Whether to keep the downloaded video file
            download_callback: Optional callback function for download progress
            progress_callback: Optional callback function for extraction progress
            
        Returns:
            List of dictionaries containing frame data and metadata
        """
        logger.info(f"Extracting frames from video {video_id} using download method")
        
        # Create output directory for this video
        video_output_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        try:
            # Download the video to a temporary location
            logger.info(f"Downloading video {video_id}")
            temp_dir = tempfile.mkdtemp(dir=self.temp_dir)
            video_path = self.download_video(
                video_id=video_id,
                output_path=os.path.join(temp_dir, f"{video_id}.mp4"),
                resolution=resolution,
                download_callback=download_callback
            )
            
            try:
                # Extract frames from the downloaded video
                logger.info(f"Extracting frames from downloaded video")
                frames = self.extract_frames_from_video(
                    video_path=video_path,
                    frame_rate=frame_rate,
                    max_frames=max_frames,
                    start_time=start_time,
                    end_time=end_time,
                    output_dir=video_output_dir,
                    progress_callback=progress_callback
                )
                
                # Save metadata
                metadata_path = os.path.join(video_output_dir, f"{video_id}_metadata.json")
                metadata = {
                    "video_id": video_id,
                    "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_rate": frame_rate,
                    "max_frames": max_frames,
                    "resolution": resolution,
                    "start_time": start_time,
                    "end_time": end_time,
                    "extracted_frames": len(frames),
                    "frames": [
                        {
                            "path": frame["path"],
                            "time": frame["time"],
                            "index": frame["index"]
                        }
                        for frame in frames
                    ]
                }
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Keep downloaded video if requested
                if keep_video:
                    # If keep_video is True, move the video to the output directory
                    kept_video_path = os.path.join(video_output_dir, f"{video_id}.mp4")
                    shutil.move(video_path, kept_video_path)
                    logger.info(f"Kept downloaded video at {kept_video_path}")
                
                return frames
            
            finally:
                # Clean up temporary files if not keeping the video
                if not keep_video and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
        
        except Exception as e:
            logger.error(f"Error in download extraction for video {video_id}: {str(e)}")
            # Re-raise with more context
            if isinstance(e, (VideoUnavailableError, YtDlpError, FFmpegError)):
                raise
            raise DownloadExtractionError(
                f"Error extracting frames from video {video_id}: {str(e)}",
                video_id=video_id
            )
    
    def scan_video_for_frames(
        self,
        video_id: str,
        search_query: str,
        vlm_analyzer,  # Type hint omitted to avoid circular import
        frame_rate: float = 1.0,
        max_frames: int = 100,
        threshold: float = 0.3,
        resolution: str = "720p",
        download_callback: Optional[Callable[[float], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Download a YouTube video and extract frames that match a specific description.
        
        Args:
            video_id: YouTube video ID
            search_query: Natural language description to search for
            vlm_analyzer: Vision Language Model analyzer instance
            frame_rate: Number of frames to extract per second
            max_frames: Maximum number of frames to extract
            threshold: Similarity threshold for matching frames
            resolution: Video resolution to download
            download_callback: Optional callback function for download progress
            progress_callback: Optional callback function for extraction progress
            
        Returns:
            List of dictionaries containing frame data, metadata, and similarity scores
        """
        # Import VLMAnalyzer here to avoid circular import
        logger.info(f"Scanning video {video_id} for frames matching: '{search_query}'")
        
        # Extract frames from the video
        frames = self.extract_frames(
            video_id=video_id,
            frame_rate=frame_rate,
            max_frames=max_frames,
            resolution=resolution,
            download_callback=download_callback,
            progress_callback=progress_callback
        )
        
        if not frames:
            logger.warning(f"No frames extracted from video {video_id}")
            return []
        
        try:
            # Calculate similarity scores for each frame
            filtered_frames = []
            
            for i, frame in enumerate(frames):
                # Get the image
                image = frame["frame"]
                
                # Calculate similarity between the image and the search query
                similarity = vlm_analyzer.calculate_similarity(image, search_query)
                
                # Add similarity score to the frame data
                frame["similarity"] = float(similarity)
                frame["query"] = search_query
                frame["threshold"] = threshold
                
                # If similarity is above threshold, add to filtered frames
                if similarity >= threshold:
                    filtered_frames.append(frame)
                
                # Call progress callback if provided
                if progress_callback and i % 5 == 0:  # Update every 5 frames
                    progress_callback(i)
            
            # Sort frames by similarity score (highest first)
            filtered_frames.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"Found {len(filtered_frames)} frames matching '{search_query}' above threshold {threshold}")
            
            # Update metadata with VLM results
            video_output_dir = os.path.join(self.output_dir, video_id)
            metadata_path = os.path.join(video_output_dir, f"{video_id}_vlm_metadata.json")
            
            metadata = {
                "video_id": video_id,
                "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "search_query": search_query,
                "frame_rate": frame_rate,
                "max_frames": max_frames,
                "threshold": threshold,
                "total_frames_analyzed": len(frames),
                "matching_frames": len(filtered_frames),
                "frames": [
                    {
                        "path": frame["path"],
                        "time": frame["time"],
                        "similarity": frame["similarity"],
                        "index": frame["index"]
                    }
                    for frame in filtered_frames
                ]
            }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return filtered_frames
            
        except Exception as e:
            logger.error(f"Error analyzing frames with VLM: {str(e)}")
            raise DownloadExtractionError(
                f"Error analyzing frames with VLM: {str(e)}",
                video_id=video_id
            )