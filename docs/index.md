# YouTube Frame Extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-2023.3.4%2B-red.svg)](https://github.com/yt-dlp/yt-dlp)
[![Selenium](https://img.shields.io/badge/Selenium-4.1.0%2B-green.svg)](https://www.selenium.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0%2B-brightgreen.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15.0%2B-EE4C2C.svg)](https://pytorch.org/vision/)
[![Transformers](https://img.shields.io/badge/Transformers-4.27.0%2B-yellow.svg)](https://huggingface.co/transformers/)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-blueviolet.svg)](https://github.com/openai/CLIP)
[![Pillow](https://img.shields.io/badge/Pillow-9.4.0%2B-blue.svg)](https://python-pillow.org/)
[![ffmpeg](https://img.shields.io/badge/ffmpeg--python-0.2.0%2B-orange.svg)](https://github.com/kkroening/ffmpeg-python)
[![Numpy](https://img.shields.io/badge/Numpy-1.24.2%2B-blue.svg)](https://numpy.org/)
[![Tesseract](https://img.shields.io/badge/pytesseract-0.3.10%2B-darkblue.svg)](https://github.com/madmaze/pytesseract)
[![Typer](https://img.shields.io/badge/Typer-0.7.0%2B-green.svg)](https://typer.tiangolo.com/)
[![Rich](https://img.shields.io/badge/Rich-13.3.0%2B-purple.svg)](https://github.com/Textualize/rich)
[![Pydantic](https://img.shields.io/badge/Pydantic-1.10.0%2B-blue.svg)](https://pydantic-docs.helpmanual.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![AWS S3](https://img.shields.io/badge/boto3-1.26.0%2B-yellow.svg)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
[![Google Cloud](https://img.shields.io/badge/GCS-2.7.0%2B-blue.svg)](https://cloud.google.com/storage)
[![pytest](https://img.shields.io/badge/pytest-7.3.0%2B-blue.svg)](https://pytest.org/)
[![MkDocs](https://img.shields.io/badge/MkDocs-1.4.2%2B-blue.svg)](https://www.mkdocs.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-1.0.0%2B-orange.svg)](https://jupyter.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1%2B-blue.svg)](https://matplotlib.org/)

## Overview

The **YouTube Frame Extractor** is a powerful and flexible toolkit for extracting and analyzing frames from YouTube videos. It supports two main methods:

- **Browser-based Extraction:** Captures frames directly from the YouTube player using Selenium.
- **Download-based Extraction:** Downloads videos using yt-dlp and extracts frames with high precision using ffmpeg or OpenCV.

This tool also supports advanced analysis features like AI-powered semantic search (using CLIP), OCR, object detection, and scene change detection. It’s ideal for developers and researchers looking to analyze video content programmatically.

## Features

- **AI-Powered Semantic Search:**  
  Use vision-language models to identify frames that match natural language queries.
- **Browser-based Extraction:**  
  Capture frames without downloading the entire video.
- **Download-based Extraction:**  
  Extract frames accurately from downloaded videos.
- **Batch Processing:**  
  Process multiple videos concurrently with built-in parallelism.
- **Flexible Configuration:**  
  Customize settings via environment variables, configuration files (YAML/JSON), or command-line arguments.
- **Cloud Storage Integration:**  
  Save extracted frames to AWS S3 or Google Cloud Storage.
- **Robust Logging and Error Handling:**  
  Enterprise-grade error handling and detailed logging.

## Installation

### Prerequisites

- **Python:** 3.8 or higher.
- **Web Browser:** Chrome, Firefox, or Edge (for browser-based extraction).
- **ffmpeg:** Ensure you have a recent version installed for download-based extraction.
- **Tesseract:** Required for OCR functionality.
- **CUDA-compatible GPU:** Optional for AI-powered analysis.

### Setup Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Harshil7875/YouTube-Frame-Extractor.git
    cd YouTube-Frame-Extractor
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional) Build the Docker Image:**

    ```bash
    docker build -t youtube-frame-extractor .
    ```

## Usage

### Command Line Interface (CLI)

The CLI is built using Typer and supports several commands:

#### Browser-based Extraction

```bash
python -m youtube_frame_extractor browser --video-id dQw4w9WgXcQ --query "close up of person singing" --interval 2 --threshold 0.3
