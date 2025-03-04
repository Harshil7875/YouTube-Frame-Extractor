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

## What is YouTube Frame Extractor?

YouTube Frame Extractor provides an enterprise-grade solution for extracting specific frames from YouTube videos. Whether you need to capture exact moments, analyze content using AI, or process videos in bulk, this toolkit offers flexible and powerful options for video frame extraction and analysis.

## Key Features

- **Multiple Extraction Methods**
  - Browser-based extraction using Selenium
  - Download-based extraction using yt-dlp and ffmpeg

- **AI-Powered Analysis**
  - Content detection with CLIP vision-language models
  - Object detection for identifying people and objects
  - OCR for extracting text from frames
  - Scene change detection

- **Flexible System Design**
  - Local or cloud storage (AWS S3/Google Cloud)
  - Parallel processing for multiple videos
  - Comprehensive configuration options
  - Robust error handling

## Quick Start

```python
# Example: Extract frames at 1-second intervals
from youtube_frame_extractor.extractors import browser

extractor = browser.BrowserExtractor()
frames = extractor.extract_frames(
    video_id="dQw4w9WgXcQ", 
    interval=1.0,
    max_frames=10
)

# Example: Find frames containing specific content
from youtube_frame_extractor.analysis import clip
from youtube_frame_extractor.extractors import download

analyzer = clip.CLIPAnalyzer()
extractor = download.DownloadExtractor()

matching_frames = extractor.scan_video_for_frames(
    video_id="dQw4w9WgXcQ",
    search_query="person dancing",
    vlm_analyzer=analyzer,
    threshold=0.3
)
```

## Installation

```bash
pip install youtube-frame-extractor
```

For detailed installation instructions, see [Installation Guide](installation.md).

## Documentation Sections

- [Installation](installation.md) - Detailed setup instructions
- [Extractors](api-reference/extractors.md) - Frame extraction methods
- [Analysis](api-reference/analysis.md) - Content analysis tools
- [Storage](api-reference/storage.md) - Local and cloud storage options
- [Configuration](api-reference/config.md) - Customizing behavior
- [Examples](examples/index.md) - Code examples and use cases
- [Contributing](contributing.md) - Guidelines for contributors

## System Requirements

- Python 3.8 or newer
- Browser (Chrome, Firefox, or Edge) for browser-based extraction
- ffmpeg (recent version) for video processing
- Optional: CUDA-compatible GPU for faster AI processing

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
