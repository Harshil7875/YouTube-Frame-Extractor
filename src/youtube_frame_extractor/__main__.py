#!/usr/bin/env python3
"""
YouTube Frame Extractor - Main Entry Point

This module serves as the entry point when the package is run directly.
Example usage:
    python -m youtube_frame_extractor browser --video-id dQw4w9WgXcQ --query "person singing"
    python -m youtube_frame_extractor download --video-id dQw4w9WgXcQ --frame-rate 1.0
    python -m youtube_frame_extractor batch --video-ids dQw4w9WgXcQ 9bZkp7q19f0 --method browser
"""

import sys
from .cli import app

if __name__ == "__main__":
    # Call the Typer app from the CLI module
    app()