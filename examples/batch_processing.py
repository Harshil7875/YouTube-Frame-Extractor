#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates how to process multiple YouTube videos in batch,
with parallel processing, progress tracking, and error handling.

Usage:
    python batch_processing.py --video-ids dQw4w9WgXcQ 9bZkp7q19f0 --method browser
    python batch_processing.py --video-file video_ids.txt --method download
"""

import argparse
import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for importing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.youtube_frame_extractor.extractors.browser import BrowserExtractor
    from src.youtube_frame_extractor.extractors.download import DownloadExtractor
except ImportError:
    print("ERROR: Could not import YouTube Frame Extractor package.")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_processing.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("batch_processing")

def process_video(
    video_id: str,
    method: str,
    output_dir: str,
    interval: float = 2.0,
    frame_rate: float = 0.5,
    max_frames: int = 10,
    query: Optional[str] = None,
) -> Tuple[str, bool, List[Dict[str, Any]], Optional[str]]:
    """
    Process a single video for batch processing.
    
    Args:
        video_id: YouTube video ID
        method: Extraction method ('browser' or 'download')
        output_dir: Directory to save extracted frames
        interval: Interval between frame captures for browser method (seconds)
        frame_rate: Frames per second for download method
        max_frames: Maximum number of frames to extract
        query: Optional search query for VLM-based extraction
        
    Returns:
        Tuple containing (video_id, success_flag, frames_list, error_message)
    """
    # Create video-specific output directory
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    try:
        frames = []
        
        if method == "browser":
            extractor = BrowserExtractor(output_dir=video_output_dir, headless=True)
            frames = extractor.extract_frames(
                video_id=video_id,
                interval=interval,
                max_frames=max_frames
            )
        else:  # download method
            extractor = DownloadExtractor(output_dir=video_output_dir)
            frames = extractor.extract_frames(
                video_id=video_id,
                frame_rate=frame_rate,
                max_frames=max_frames
            )
            
        return video_id, True, frames, None
        
    except Exception as e:
        error_message = f"Error processing video {video_id}: {str(e)}"
        logger.error(error_message)
        return video_id, False, [], error_message

def batch_process_videos(
    video_ids: List[str],
    method: str,
    output_dir: str,
    interval: float = 2.0,
    frame_rate: float = 0.5,
    max_frames: int = 10,
    query: Optional[str] = None,
    max_workers: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple videos in batch using parallel execution.
    
    Args:
        video_ids: List of YouTube video IDs
        method: Extraction method ('browser' or 'download')
        output_dir: Directory to save extracted frames
        interval: Interval between frame captures for browser method (seconds)
        frame_rate: Frames per second for download method
        max_frames: Maximum number of frames to extract
        query: Optional search query for VLM-based extraction
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dictionary mapping video IDs to results information
    """
    logger.info(f"Starting batch processing of {len(video_ids)} videos")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    completed = 0
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_video = {
            executor.submit(
                process_video,
                video_id,
                method,
                output_dir,
                interval,
                frame_rate,
                max_frames,
                query
            ): video_id for video_id in video_ids
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            video_id = future_to_video[future]
            
            try:
                video_id, success, frames, error = future.result()
                
                results[video_id] = {
                    "success": success,
                    "frame_count": len(frames),
                    "error": error,
                    "frames": frames
                }
                
                completed += 1
                logger.info(f"Progress: {completed}/{len(video_ids)} videos processed")
                
                if success:
                    logger.info(f"Successfully extracted {len(frames)} frames from video {video_id}")
                else:
                    logger.error(f"Failed to process video {video_id}: {error}")
                
            except Exception as e:
                logger.error(f"Exception occurred while processing video {video_id}: {str(e)}")
                results[video_id] = {
                    "success": False,
                    "frame_count": 0,
                    "error": str(e),
                    "frames": []
                }
                
                completed += 1
                logger.info(f"Progress: {completed}/{len(video_ids)} videos processed")
    
    logger.info(f"Batch processing complete. Processed {len(video_ids)} videos")
    return results

def load_video_ids_from_file(file_path: str) -> List[str]:
    """
    Load YouTube video IDs from a text file.
    
    Args:
        file_path: Path to the file containing video IDs (one per line)
        
    Returns:
        List of YouTube video IDs
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            video_ids = [line.strip() for line in f.readlines()]
            video_ids = [vid for vid in video_ids if vid and not vid.startswith('#')]
            
        logger.info(f"Loaded {len(video_ids)} video IDs from {file_path}")
        return video_ids
        
    except Exception as e:
        logger.error(f"Error loading video IDs from {file_path}: {str(e)}")
        return []

def generate_summary_report(results: Dict[str, Dict[str, Any]], output_file: str):
    """
    Generate a summary report of the batch processing results.
    
    Args:
        results: Dictionary mapping video IDs to results information
        output_file: Path to save the summary report
    """
    try:
        with open(output_file, 'w') as f:
            f.write("# Batch Processing Summary Report\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Statistics
            total_videos = len(results)
            successful_videos = sum(1 for v in results.values() if v["success"])
            total_frames = sum(v["frame_count"] for v in results.values())
            
            f.write(f"Total videos processed: {total_videos}\n")
            f.write(f"Successful extractions: {successful_videos}\n")
            f.write(f"Failed extractions: {total_videos - successful_videos}\n")
            f.write(f"Total frames extracted: {total_frames}\n\n")
            
            # Video details
            f.write("## Video Details\n\n")
            for video_id, result in results.items():
                status = "✅ Success" if result["success"] else "❌ Failed"
                f.write(f"### Video: {video_id} - {status}\n\n")
                
                if result["success"]:
                    f.write(f"Frames extracted: {result['frame_count']}\n")
                    frames_str = ", ".join(f.get('path', 'unknown') for f in result["frames"][:3])
                    if result["frame_count"] > 3:
                        frames_str += f", ... and {result['frame_count'] - 3} more frames"
                    f.write(f"Sample frames: {frames_str}\n")
                else:
                    f.write(f"Error: {result['error']}\n")
                    
                f.write("\n")
            
        logger.info(f"Summary report generated: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")

def main():
    """
    Main entry point for the batch processing example.
    """
    parser = argparse.ArgumentParser(description="Process multiple YouTube videos in batch")
    
    # Video ID arguments (either directly or from file)
    video_id_group = parser.add_mutually_exclusive_group(required=True)
    video_id_group.add_argument(
        "--video-ids", 
        nargs="+", 
        help="List of YouTube video IDs to process"
    )
    video_id_group.add_argument(
        "--video-file", 
        type=str, 
        help="Path to file containing YouTube video IDs (one per line)"
    )
    
    # Processing parameters
    parser.add_argument(
        "--method", 
        choices=["browser", "download"], 
        default="browser",
        help="Frame extraction method"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./batch_output",
        help="Directory to save frames"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=2.0,
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
        default=10,
        help="Maximum number of frames to extract per video"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Optional search query for VLM-based extraction"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=3,
        help="Maximum number of concurrent workers"
    )
    parser.add_argument(
        "--report", 
        type=str, 
        default="batch_report.md",
        help="Path to save the summary report"
    )
    
    args = parser.parse_args()
    
    # Get the list of video IDs
    if args.video_ids:
        video_ids = args.video_ids
    else:
        video_ids = load_video_ids_from_file(args.video_file)
        
    if not video_ids:
        logger.error("No video IDs provided. Exiting.")
        sys.exit(1)
    
    # Process videos in batch
    start_time = time.time()
    
    results = batch_process_videos(
        video_ids=video_ids,
        method=args.method,
        output_dir=args.output_dir,
        interval=args.interval,
        frame_rate=args.frame_rate,
        max_frames=args.frames,
        query=args.query,
        max_workers=args.workers
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
    
    # Generate summary report
    generate_summary_report(results, args.report)
    
    # Print summary
    successful = sum(1 for r in results.values() if r["success"])
    logger.info(f"Summary: Successfully processed {successful}/{len(video_ids)} videos")
    logger.info(f"Total frames extracted: {sum(r['frame_count'] for r in results.values())}")
    logger.info(f"Check {args.report} for detailed report")

if __name__ == "__main__":
    main()