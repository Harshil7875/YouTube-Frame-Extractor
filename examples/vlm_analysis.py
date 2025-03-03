#!/usr/bin/env python3
"""
VLM-based Frame Analysis Example

This example demonstrates how to use Vision Language Models (VLMs)
to find frames in YouTube videos that match specific natural language descriptions.

Usage:
    python vlm_analysis.py --video-id dQw4w9WgXcQ --query "person dancing" --threshold 0.3
    python vlm_analysis.py --video-id dQw4w9WgXcQ --query "close up of face" --method download
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for importing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.youtube_frame_extractor.extractors.browser import BrowserExtractor
    from src.youtube_frame_extractor.extractors.download import DownloadExtractor
    from src.youtube_frame_extractor.analysis.vlm import VLMAnalyzer
except ImportError:
    print("ERROR: Could not import YouTube Frame Extractor package.")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("vlm_analysis")

def extract_and_analyze_frames(
    video_id: str,
    query: str,
    method: str = "browser",
    output_dir: str = "./vlm_output",
    model_name: str = "openai/clip-vit-base-patch16",
    threshold: float = 0.3,
    interval: float = 1.0,
    frame_rate: float = 0.5,
    max_frames: int = 50,
) -> List[Dict[str, Any]]:
    """
    Extract frames from a YouTube video and analyze them using a VLM.
    
    Args:
        video_id: YouTube video ID
        query: Natural language description to search for
        method: Extraction method ('browser' or 'download')
        output_dir: Directory to save extracted frames
        model_name: Name of the VLM model to use
        threshold: Similarity threshold for considering a match
        interval: Interval between frame captures for browser method (seconds)
        frame_rate: Frames per second for download method
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of matched frames with metadata and similarity scores
    """
    logger.info(f"Extracting and analyzing frames from video {video_id} using {method} method")
    logger.info(f"Searching for frames matching description: '{query}'")
    
    # Create video-specific output directory
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Initialize the VLM analyzer
    try:
        logger.info(f"Initializing VLM analyzer with model: {model_name}")
        vlm_analyzer = VLMAnalyzer(model_name=model_name)
    except Exception as e:
        logger.error(f"Error initializing VLM analyzer: {str(e)}")
        return []
    
    # Extract frames
    try:
        frames = []
        
        if method == "browser":
            logger.info("Using browser-based extraction")
            extractor = BrowserExtractor(output_dir=video_output_dir, headless=True)
            
            # For browser method, we can use the scan_video_for_frames method
            frames = extractor.scan_video_for_frames(
                video_id=video_id,
                search_query=query,
                vlm_analyzer=vlm_analyzer,
                interval=interval,
                threshold=threshold,
                max_frames=max_frames
            )
        else:  # download method
            logger.info("Using download-based extraction")
            extractor = DownloadExtractor(output_dir=video_output_dir)
            
            # For download method, we extract frames first, then analyze them
            raw_frames = extractor.extract_frames(
                video_id=video_id,
                frame_rate=frame_rate,
                max_frames=max_frames
            )
            
            logger.info(f"Extracted {len(raw_frames)} frames, now analyzing with VLM")
            
            # Analyze each frame with the VLM
            for frame_data in raw_frames:
                image = frame_data.get("frame")
                if image is None:
                    continue
                
                # Calculate similarity between image and query
                similarity = vlm_analyzer.calculate_similarity(image, query)
                
                # Add similarity score to frame data
                frame_data["similarity"] = float(similarity)
                frame_data["query"] = query
                
                # If above threshold, add to matched frames
                if similarity >= threshold:
                    frames.append(frame_data)
            
            logger.info(f"Found {len(frames)} frames matching the query above threshold {threshold}")
        
        # Sort by similarity (highest first)
        frames.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting and analyzing frames: {str(e)}")
        return []

def save_results(frames: List[Dict[str, Any]], video_id: str, query: str, output_dir: str) -> str:
    """
    Save VLM analysis results to a JSON file.
    
    Args:
        frames: List of frame data dictionaries with similarity scores
        video_id: YouTube video ID
        query: The search query used
        output_dir: Directory to save the results
        
    Returns:
        Path to the saved results file
    """
    if not frames:
        logger.warning("No frames to save")
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a results dictionary
    results = {
        "video_id": video_id,
        "query": query,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_count": len(frames),
        "frames": []
    }
    
    # Add frame data
    for frame in frames:
        # Create a serializable frame entry
        frame_entry = {
            "path": frame.get("path", ""),
            "time": frame.get("time", 0),
            "similarity": frame.get("similarity", 0),
            "metadata": {
                k: v for k, v in frame.items() 
                if k not in ["frame", "path", "time", "similarity"]
                and not callable(v)
            }
        }
        results["frames"].append(frame_entry)
    
    # Save to JSON file
    filename = f"{video_id}_{query.replace(' ', '_')}_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return ""

def display_top_results(frames: List[Dict[str, Any]], top_n: int = 5):
    """
    Display information about the top matched frames.
    
    Args:
        frames: List of frame data dictionaries with similarity scores
        top_n: Number of top results to display
    """
    if not frames:
        logger.warning("No matching frames found")
        return
    
    logger.info(f"Top {min(top_n, len(frames))} matching frames:")
    
    for i, frame in enumerate(frames[:top_n]):
        similarity = frame.get("similarity", 0) * 100  # Convert to percentage
        time_str = f"{frame.get('time', 0):.2f}s" if "time" in frame else "N/A"
        path = frame.get("path", "N/A")
        
        logger.info(f"{i+1}. Similarity: {similarity:.1f}% | Time: {time_str} | Path: {path}")
    
    if len(frames) > top_n:
        logger.info(f"... and {len(frames) - top_n} more matching frames")

def main():
    """
    Main entry point for the VLM analysis example.
    """
    parser = argparse.ArgumentParser(
        description="Find frames in YouTube videos that match specific descriptions using VLM"
    )
    parser.add_argument("--video-id", type=str, required=True, help="YouTube video ID")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True,
        help="Natural language description to search for"
    )
    parser.add_argument(
        "--method", 
        choices=["browser", "download"], 
        default="browser",
        help="Frame extraction method"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./vlm_output",
        help="Directory to save frames and results"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/clip-vit-base-patch16",
        help="VLM model name"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.3,
        help="Similarity threshold (0.0 to 1.0)"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=1.0,
        help="Interval between frames for browser method (seconds)"
    )
    parser.add_argument(
        "--frame-rate", 
        type=float, 
        default=0.5,
        help="Frames per second for download method"
    )
    parser.add_argument(
        "--frames", 
        type=int, 
        default=50,
        help="Maximum number of frames to extract"
    )
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=5,
        help="Number of top results to display"
    )
    
    args = parser.parse_args()
    
    # Extract and analyze frames
    start_time = time.time()
    
    frames = extract_and_analyze_frames(
        video_id=args.video_id,
        query=args.query,
        method=args.method,
        output_dir=args.output_dir,
        model_name=args.model,
        threshold=args.threshold,
        interval=args.interval,
        frame_rate=args.frame_rate,
        max_frames=args.frames
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Display top results
    display_top_results(frames, args.top_n)
    
    # Save results to file
    if frames:
        results_file = save_results(
            frames=frames,
            video_id=args.video_id,
            query=args.query,
            output_dir=args.output_dir
        )
        if results_file:
            logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()