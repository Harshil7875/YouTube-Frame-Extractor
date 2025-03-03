#!/usr/bin/env python3
"""
Command Line Interface for YouTube Frame Extractor

This module provides the CLI for the YouTube Frame Extractor package,
supporting various extraction methods and analysis options.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

import typer
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .extractors.browser import BrowserExtractor
from .extractors.download import DownloadExtractor
from .analysis.vlm import VLMAnalyzer
from .logger import get_logger
from .config import settings

# Initialize Typer app
app = typer.Typer(
    name="YouTube Frame Extractor",
    help="Extract and analyze frames from YouTube videos",
    add_completion=False,
)

# Initialize rich console
console = Console()

# Get logger
logger = get_logger(__name__)

# Define commands
@app.command("browser")
def browser_extraction(
    video_id: str = typer.Argument(..., help="YouTube video ID"),
    output_dir: str = typer.Option("./output", help="Directory to save extracted frames"),
    interval: float = typer.Option(2.0, help="Interval between frame captures (seconds)"),
    max_frames: int = typer.Option(50, help="Maximum number of frames to extract"),
    duration: Optional[float] = typer.Option(None, help="Duration of video to process (None for full video)"),
    headless: bool = typer.Option(True, help="Run browser in headless mode"),
    query: Optional[str] = typer.Option(None, help="Natural language query for VLM-based extraction"),
    threshold: float = typer.Option(0.3, help="Similarity threshold for VLM-based matching (0.0 to 1.0)"),
    model_name: str = typer.Option("openai/clip-vit-base-patch16", help="VLM model to use"),
    save_metadata: bool = typer.Option(True, help="Save metadata along with extracted frames"),
):
    """
    Extract frames directly from the YouTube player using browser automation.
    This approach captures frames without downloading the full video.
    """
    start_time = time.time()
    
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize extractor
        extractor = BrowserExtractor(output_dir=output_dir, headless=headless)
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Initial task description
            task = progress.add_task(f"Extracting frames from video {video_id}", total=max_frames)
            
            # Check if we should use VLM for extraction
            if query:
                console.print(f"Using VLM to find frames matching: [bold cyan]'{query}'[/]")
                # Initialize VLM analyzer
                vlm_analyzer = VLMAnalyzer(model_name=model_name)
                
                # Extract frames with VLM analysis
                frames = extractor.scan_video_for_frames(
                    video_id=video_id,
                    search_query=query,
                    vlm_analyzer=vlm_analyzer,
                    interval=interval,
                    threshold=threshold,
                    max_frames=max_frames,
                    duration=duration,
                    progress_callback=lambda i: progress.update(task, completed=i, description=f"Analyzing frame {i}/{max_frames}")
                )
            else:
                # Extract frames without VLM
                frames = extractor.extract_frames(
                    video_id=video_id,
                    interval=interval,
                    max_frames=max_frames,
                    duration=duration,
                    progress_callback=lambda i: progress.update(task, completed=i, description=f"Extracting frame {i}/{max_frames}")
                )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Display results
        if frames:
            console.print(f"\n[bold green]Successfully extracted {len(frames)} frames in {elapsed_time:.2f} seconds[/]")
            
            # Print information about the first few frames
            console.print("\n[bold]Sample frames:[/]")
            for i, frame in enumerate(frames[:5]):
                time_str = f"Time: {frame.get('time', i):.2f}s" if 'time' in frame else ""
                similarity_str = f"Similarity: {frame.get('similarity', 0):.2f}" if 'similarity' in frame else ""
                path_str = f"Path: {frame.get('path', 'N/A')}"
                console.print(f"  Frame {i+1}: {time_str} {similarity_str} {path_str}")
            
            if len(frames) > 5:
                console.print(f"  ... and {len(frames) - 5} more frames")
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = os.path.join(output_dir, f"{video_id}_metadata.json")
                metadata = {
                    "video_id": video_id,
                    "extraction_method": "browser",
                    "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "threshold": threshold if query else None,
                    "frame_count": len(frames),
                    "frames": [
                        {k: v for k, v in frame.items() if k != 'frame' and not callable(v)}
                        for frame in frames
                    ]
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                console.print(f"\nMetadata saved to: [bold]{metadata_path}[/]")
        else:
            console.print("\n[bold yellow]No frames were extracted[/]")
        
        return frames
        
    except Exception as e:
        console.print(f"\n[bold red]Error extracting frames: {str(e)}[/]")
        logger.error(f"Error in browser extraction: {str(e)}", exc_info=True)
        return []

@app.command("download")
def download_extraction(
    video_id: str = typer.Argument(..., help="YouTube video ID"),
    output_dir: str = typer.Option("./output", help="Directory to save extracted frames"),
    frame_rate: float = typer.Option(1.0, help="Frames per second to extract"),
    max_frames: int = typer.Option(100, help="Maximum number of frames to extract"),
    resolution: str = typer.Option("720p", help="Video resolution to download"),
    keep_video: bool = typer.Option(False, help="Keep downloaded video file after extraction"),
    query: Optional[str] = typer.Option(None, help="Natural language query for VLM-based filtering"),
    threshold: float = typer.Option(0.3, help="Similarity threshold for VLM-based matching (0.0 to 1.0)"),
    model_name: str = typer.Option("openai/clip-vit-base-patch16", help="VLM model to use"),
    save_metadata: bool = typer.Option(True, help="Save metadata along with extracted frames"),
):
    """
    Extract frames by downloading the video and processing it locally.
    This approach provides higher quality frames but requires downloading the full video.
    """
    start_time = time.time()
    
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize extractor
        extractor = DownloadExtractor(output_dir=output_dir)
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Create initial task for downloading
            download_task = progress.add_task(f"Downloading video {video_id}", total=100)
            
            # Create callback for download progress
            def download_callback(percent):
                progress.update(download_task, completed=percent)
            
            # Extract frames
            frames = extractor.extract_frames(
                video_id=video_id,
                frame_rate=frame_rate,
                max_frames=max_frames,
                resolution=resolution,
                keep_video=keep_video,
                download_callback=download_callback
            )
            
            # If using VLM for filtering
            if query and frames:
                vlm_task = progress.add_task(f"Analyzing frames with VLM", total=len(frames))
                
                # Initialize VLM analyzer
                vlm_analyzer = VLMAnalyzer(model_name=model_name)
                
                # Apply VLM filtering
                filtered_frames = []
                for i, frame in enumerate(frames):
                    # Update progress
                    progress.update(vlm_task, completed=i, description=f"Analyzing frame {i+1}/{len(frames)}")
                    
                    # Get image from frame
                    if 'frame' in frame and frame['frame'] is not None:
                        image = frame['frame']
                    elif 'path' in frame and os.path.exists(frame['path']):
                        from PIL import Image
                        image = Image.open(frame['path'])
                    else:
                        continue
                    
                    # Calculate similarity
                    similarity = vlm_analyzer.calculate_similarity(image, query)
                    
                    # Add similarity to frame data
                    frame['similarity'] = float(similarity)
                    frame['query'] = query
                    
                    # If above threshold, keep frame
                    if similarity >= threshold:
                        filtered_frames.append(frame)
                
                # Sort by similarity
                filtered_frames.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                frames = filtered_frames
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Display results
        if frames:
            console.print(f"\n[bold green]Successfully extracted {len(frames)} frames in {elapsed_time:.2f} seconds[/]")
            
            if query:
                console.print(f"Found {len(frames)} frames matching the query: [bold cyan]'{query}'[/]")
            
            # Print information about the first few frames
            console.print("\n[bold]Sample frames:[/]")
            for i, frame in enumerate(frames[:5]):
                frame_info = []
                if 'time' in frame:
                    frame_info.append(f"Time: {frame['time']:.2f}s")
                if 'similarity' in frame:
                    frame_info.append(f"Similarity: {frame['similarity']:.2f}")
                if 'path' in frame:
                    frame_info.append(f"Path: {frame['path']}")
                console.print(f"  Frame {i+1}: {' | '.join(frame_info)}")
            
            if len(frames) > 5:
                console.print(f"  ... and {len(frames) - 5} more frames")
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = os.path.join(output_dir, f"{video_id}_metadata.json")
                metadata = {
                    "video_id": video_id,
                    "extraction_method": "download",
                    "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "threshold": threshold if query else None,
                    "frame_count": len(frames),
                    "frames": [
                        {k: v for k, v in frame.items() if k != 'frame' and not callable(v)}
                        for frame in frames
                    ]
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                console.print(f"\nMetadata saved to: [bold]{metadata_path}[/]")
        else:
            console.print("\n[bold yellow]No frames were extracted[/]")
        
        return frames
        
    except Exception as e:
        console.print(f"\n[bold red]Error extracting frames: {str(e)}[/]")
        logger.error(f"Error in download extraction: {str(e)}", exc_info=True)
        return []

@app.command("batch")
def batch_processing(
    video_ids: List[str] = typer.Argument(..., help="YouTube video IDs to process"),
    method: str = typer.Option("browser", help="Extraction method (browser or download)"),
    output_dir: str = typer.Option("./batch_output", help="Directory to save extracted frames"),
    query: Optional[str] = typer.Option(None, help="Natural language query for VLM-based extraction"),
    interval: float = typer.Option(2.0, help="Interval between frame captures for browser method (seconds)"),
    frame_rate: float = typer.Option(1.0, help="Frames per second for download method"),
    max_frames: int = typer.Option(20, help="Maximum number of frames to extract per video"),
    threshold: float = typer.Option(0.3, help="Similarity threshold for VLM-based matching (0.0 to 1.0)"),
    workers: int = typer.Option(3, help="Maximum number of concurrent workers"),
    report: bool = typer.Option(True, help="Generate batch processing report"),
    report_path: Optional[str] = typer.Option(None, help="Path to save the report (default: {output_dir}/batch_report.md)"),
    video_file: Optional[str] = typer.Option(None, help="Path to file containing YouTube video IDs (one per line)"),
):
    """
    Process multiple YouTube videos in batch.
    Supports parallel processing, progress tracking, and comprehensive reporting.
    """
    # If video_file is provided, load video IDs from file
    if video_file:
        if not os.path.exists(video_file):
            console.print(f"[bold red]Error: File {video_file} not found[/]")
            return
        
        with open(video_file, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            file_video_ids = [line.strip() for line in f.readlines()]
            file_video_ids = [vid for vid in file_video_ids if vid and not vid.startswith('#')]
        
        console.print(f"Loaded {len(file_video_ids)} video IDs from {video_file}")
        
        # Combine with directly provided video IDs
        video_ids.extend(file_video_ids)
    
    # Remove duplicates while preserving order
    video_ids = list(dict.fromkeys(video_ids))
    
    if not video_ids:
        console.print("[bold yellow]No video IDs provided. Exiting.[/]")
        return
    
    start_time = time.time()
    console.print(f"Starting batch processing of [bold cyan]{len(video_ids)}[/] videos")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set report path if not provided
    if report and not report_path:
        report_path = os.path.join(output_dir, "batch_report.md")
    
    # Start processing
    results = {}
    completed = 0
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit jobs for each video
        future_to_video = {}
        
        for video_id in video_ids:
            video_output_dir = os.path.join(output_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            if method.lower() == "browser":
                future = executor.submit(
                    browser_extraction,
                    video_id=video_id,
                    output_dir=video_output_dir,
                    interval=interval,
                    max_frames=max_frames,
                    query=query,
                    threshold=threshold,
                    save_metadata=True
                )
            else:  # download method
                future = executor.submit(
                    download_extraction,
                    video_id=video_id,
                    output_dir=video_output_dir,
                    frame_rate=frame_rate,
                    max_frames=max_frames,
                    query=query,
                    threshold=threshold,
                    save_metadata=True
                )
            
            future_to_video[future] = video_id
        
        # Process results as they complete
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            batch_task = progress.add_task(f"Processing videos", total=len(video_ids))
            
            for future in concurrent.futures.as_completed(future_to_video):
                video_id = future_to_video[future]
                
                try:
                    frames = future.result()
                    success = len(frames) > 0
                    
                    results[video_id] = {
                        "success": success,
                        "frame_count": len(frames),
                        "error": None if success else "No frames extracted",
                        "frames": frames
                    }
                    
                except Exception as e:
                    console.print(f"[bold red]Error processing video {video_id}: {str(e)}[/]")
                    results[video_id] = {
                        "success": False,
                        "frame_count": 0,
                        "error": str(e),
                        "frames": []
                    }
                
                # Update progress
                completed += 1
                progress.update(batch_task, completed=completed, 
                                description=f"Processed {completed}/{len(video_ids)} videos")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Generate report if requested
    if report:
        try:
            with open(report_path, 'w') as f:
                f.write("# Batch Processing Summary Report\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Statistics
                total_videos = len(results)
                successful_videos = sum(1 for v in results.values() if v["success"])
                total_frames = sum(v["frame_count"] for v in results.values())
                
                f.write(f"Total videos processed: {total_videos}\n")
                f.write(f"Successful extractions: {successful_videos}\n")
                f.write(f"Failed extractions: {total_videos - successful_videos}\n")
                f.write(f"Total frames extracted: {total_frames}\n")
                f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")
                
                # Video details
                f.write("## Video Details\n\n")
                for video_id, result in results.items():
                    status = "✅ Success" if result["success"] else "❌ Failed"
                    f.write(f"### Video: {video_id} - {status}\n\n")
                    
                    if result["success"]:
                        f.write(f"Frames extracted: {result['frame_count']}\n")
                        
                        # Show sample frames
                        sample_frames = result["frames"][:3]
                        if sample_frames:
                            f.write("Sample frames:\n")
                            for i, frame in enumerate(sample_frames):
                                frame_info = []
                                if 'time' in frame:
                                    frame_info.append(f"Time: {frame['time']:.2f}s")
                                if 'similarity' in frame:
                                    frame_info.append(f"Similarity: {frame['similarity']:.2f}")
                                if 'path' in frame:
                                    frame_info.append(f"Path: {frame['path']}")
                                f.write(f"- Frame {i+1}: {' | '.join(frame_info)}\n")
                            
                            if result["frame_count"] > 3:
                                f.write(f"- ... and {result['frame_count'] - 3} more frames\n")
                    else:
                        f.write(f"Error: {result['error']}\n")
                    
                    f.write("\n")
            
            console.print(f"\nSummary report generated: [bold]{report_path}[/]")
            
        except Exception as e:
            console.print(f"[bold red]Error generating report: {str(e)}[/]")
    
    # Print summary
    console.print(f"\n[bold green]Batch processing completed in {elapsed_time:.2f} seconds[/]")
    successful = sum(1 for r in results.values() if r["success"])
    console.print(f"Successfully processed {successful}/{len(video_ids)} videos")
    console.print(f"Total frames extracted: {sum(r['frame_count'] for r in results.values())}")
    
    return results

@app.command("vlm")
def vlm_analysis(
    video_id: str = typer.Argument(..., help="YouTube video ID"),
    query: str = typer.Argument(..., help="Natural language description to search for"),
    method: str = typer.Option("browser", help="Extraction method (browser or download)"),
    output_dir: str = typer.Option("./vlm_output", help="Directory to save frames and results"),
    model_name: str = typer.Option("openai/clip-vit-base-patch16", help="VLM model name"),
    threshold: float = typer.Option(0.3, help="Similarity threshold (0.0 to 1.0)"),
    interval: float = typer.Option(1.0, help="Interval between frames for browser method (seconds)"),
    frame_rate: float = typer.Option(0.5, help="Frames per second for download method"),
    max_frames: int = typer.Option(50, help="Maximum number of frames to extract"),
    top_n: int = typer.Option(5, help="Number of top results to display"),
):
    """
    Find frames in YouTube videos that match specific descriptions using Vision Language Models.
    This command combines extraction and VLM analysis in a single step.
    """
    if method.lower() == "browser":
        return browser_extraction(
            video_id=video_id,
            output_dir=output_dir,
            interval=interval,
            max_frames=max_frames,
            query=query,
            threshold=threshold,
            model_name=model_name,
            save_metadata=True
        )
    else:  # download method
        return download_extraction(
            video_id=video_id,
            output_dir=output_dir,
            frame_rate=frame_rate,
            max_frames=max_frames,
            query=query,
            threshold=threshold,
            model_name=model_name,
            save_metadata=True
        )

@app.callback()
def main():
    """
    YouTube Frame Extractor - A tool for extracting and analyzing frames from YouTube videos.
    """
    # Print the package version and banner
    from . import __version__
    console.print(f"[bold cyan]YouTube Frame Extractor v{__version__}[/]")
    console.print("[bold blue]A tool for extracting and analyzing frames from YouTube videos[/]")
    console.print("")

if __name__ == "__main__":
    app()