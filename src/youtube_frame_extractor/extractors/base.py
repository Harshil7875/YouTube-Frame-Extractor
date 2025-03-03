#!/usr/bin/env python3
"""
Base Extractor for YouTube Frame Extractor

This module defines the abstract base class for all extractors
in the YouTube Frame Extractor package.
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import shutil

from PIL import Image
import numpy as np

from ..exceptions import ExtractionError
from ..logger import get_logger
from ..config import get_settings

# Get logger and settings
logger = get_logger(__name__)
settings = get_settings()

class BaseExtractor(ABC):
    """
    Abstract base class for all frame extractors.
    
    This class defines the common interface and shared functionality
    for all frame extraction methods (browser-based, download-based, etc.).
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the base extractor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized extractor with output directory: {output_dir}")
    
    @abstractmethod
    def extract_frames(self, video_id: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract frames from a YouTube video.
        
        This is the primary method that must be implemented by all extractors.
        
        Args:
            video_id: YouTube video ID
            **kwargs: Additional arguments specific to each extractor type
            
        Returns:
            List of dictionaries containing frame data and metadata
        """
        pass
    
    def save_frames(
        self, 
        frames: List[Dict[str, Any]], 
        video_id: str, 
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Save extracted frames to disk.
        
        Args:
            frames: List of dictionaries containing frame data
            video_id: YouTube video ID
            output_dir: Directory to save frames (overrides default)
            
        Returns:
            Updated list of frames with file paths
        """
        # Determine output directory
        if output_dir:
            video_output_dir = output_dir
        else:
            video_output_dir = os.path.join(self.output_dir, video_id)
        
        # Create directory if it doesn't exist
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Save each frame
        saved_frames = []
        for i, frame in enumerate(frames):
            try:
                # Get timestamp from frame data or use index
                timestamp = frame.get('time', i)
                
                # Generate filename
                filename = f"frame_{timestamp:.2f}.jpg"
                filepath = os.path.join(video_output_dir, filename)
                
                # Get the image from frame data
                image = None
                if 'frame' in frame and frame['frame'] is not None:
                    image = frame['frame']
                elif 'image' in frame and frame['image'] is not None:
                    image = frame['image']
                else:
                    logger.warning(f"No image data found in frame {i}")
                    continue
                
                # Convert to PIL Image if it's a numpy array
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                # Save the image
                if isinstance(image, Image.Image):
                    image.save(filepath, "JPEG")
                    
                    # Update frame data with path
                    frame['path'] = filepath
                    saved_frames.append(frame)
                else:
                    logger.warning(f"Invalid image format for frame {i}")
            
            except Exception as e:
                logger.error(f"Error saving frame {i}: {str(e)}")
        
        logger.info(f"Saved {len(saved_frames)} frames to {video_output_dir}")
        return saved_frames
    
    def get_video_url(self, video_id: str) -> str:
        """
        Get the YouTube URL for a video ID.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Full YouTube URL
        """
        return f"https://www.youtube.com/watch?v={video_id}"
    
    def scan_video_for_frames(
        self,
        video_id: str,
        search_query: str,
        vlm_analyzer,  # Type hint omitted to avoid circular import
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from a video that match a specific description using VLM.
        
        This is a convenience method that combines extraction and VLM analysis.
        
        Args:
            video_id: YouTube video ID
            search_query: Natural language description to search for
            vlm_analyzer: Vision Language Model analyzer instance
            **kwargs: Additional arguments for the extract_frames method
            
        Returns:
            List of frames that match the search query
        """
        # Default implementation that should be overridden by subclasses
        # for more efficient implementations
        
        # Extract frames
        frames = self.extract_frames(video_id=video_id, **kwargs)
        
        if not frames:
            logger.warning(f"No frames extracted from video {video_id}")
            return []
        
        # Get parameters
        threshold = kwargs.get('threshold', 0.3)
        
        try:
            # Calculate similarity scores for each frame
            filtered_frames = []
            
            for frame in frames:
                # Get the image
                image = None
                if 'frame' in frame and frame['frame'] is not None:
                    image = frame['frame']
                elif 'image' in frame and frame['image'] is not None:
                    image = frame['image']
                elif 'path' in frame and os.path.exists(frame['path']):
                    image = Image.open(frame['path'])
                
                if image is None:
                    logger.warning(f"No image data found in frame")
                    continue
                
                # Calculate similarity between the image and the search query
                similarity = vlm_analyzer.calculate_similarity(image, search_query)
                
                # Add similarity score to the frame data
                frame['similarity'] = float(similarity)
                frame['query'] = search_query
                frame['threshold'] = threshold
                
                # If similarity is above threshold, add to filtered frames
                if similarity >= threshold:
                    filtered_frames.append(frame)
            
            # Sort frames by similarity score (highest first)
            filtered_frames.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Found {len(filtered_frames)} frames matching '{search_query}' above threshold {threshold}")
            return filtered_frames
            
        except Exception as e:
            logger.error(f"Error analyzing frames with VLM: {str(e)}")
            raise ExtractionError(f"Error analyzing frames with VLM: {str(e)}")
    
    def cleanup(self) -> None:
        """
        Clean up any resources used by the extractor.
        
        This method should be called when the extractor is no longer needed.
        """
        # Default implementation does nothing
        # Subclasses should override this if they need to clean up resources
        pass
    
    def __enter__(self):
        """
        Support for context manager protocol.
        
        Returns:
            Self reference
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager protocol.
        
        Calls cleanup() when exiting a context manager block.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.cleanup()