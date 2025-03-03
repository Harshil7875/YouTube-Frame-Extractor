# YouTube Frame Extractor 🎬✨

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
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0.0%2B-blue.svg)](https://pydantic-docs.helpmanual.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![AWS S3](https://img.shields.io/badge/boto3-1.26.0%2B-yellow.svg)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
[![Google Cloud](https://img.shields.io/badge/GCS-2.7.0%2B-blue.svg)](https://cloud.google.com/storage)
[![pytest](https://img.shields.io/badge/pytest-7.3.0%2B-blue.svg)](https://pytest.org/)
[![MkDocs](https://img.shields.io/badge/MkDocs-1.4.2%2B-blue.svg)](https://www.mkdocs.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-1.0.0%2B-orange.svg)](https://jupyter.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1%2B-blue.svg)](https://matplotlib.org/)

**Extract exactly what you need from YouTube videos with AI-powered precision!** 🔍

YouTube Frame Extractor is a powerful Python toolkit for extracting and analyzing frames from YouTube videos. Whether you need specific content moments or systematically processed frames, this tool provides enterprise-grade functionality with a flexible, modular architecture.

## 📋 Table of Contents

- [Features](#-Features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Extraction Methods

- **🌐 Browser-based extraction**: Capture frames directly from the YouTube player using Selenium
- **📥 Download-based extraction**: Download videos with yt-dlp and extract frames using ffmpeg or OpenCV

### Analysis Capabilities

- **🧠 AI-Powered Content Detection**: Find specific content using CLIP vision-language models
- **🔎 Object Detection**: Identify people, vehicles, and objects using pre-trained models
- **📝 Text Extraction (OCR)**: Extract text from frames with integrated Tesseract
- **🎬 Scene Change Detection**: Identify scene transitions using MSE, SSIM, or histogram comparison

### System Design

- **☁️ Flexible Storage**: Save frames locally or in cloud storage (AWS S3/Google Cloud)
- **🚀 Parallel Processing**: Process multiple videos simultaneously with optimized threading
- **🎭 Multi-Browser Support**: Chrome, Firefox, Edge compatibility
- **🔧 Extensive Configuration**: Command line, environment variables, and YAML/JSON config files
- **🐛 Robust Error Handling**: Comprehensive exception system keeps batch jobs running
- **🐳 Docker Support**: Containerized operation for consistent environments

## 🚀 Installation

### Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.8 or newer |
| Browser | Chrome, Firefox, or Edge (for browser extraction) |
| ffmpeg | Recent version installed and in PATH |
| Tesseract | Optional, for OCR functionality |
| GPU | Optional, CUDA-compatible for faster AI processing |

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/Harshil7875/YouTube-Frame-Extractor.git
cd YouTube-Frame-Extractor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build the Docker image
docker build -t youtube-frame-extractor .

# Run with Docker
docker run -v $(pwd)/output:/app/output youtube-frame-extractor [COMMANDS]
```

## 🏁 Quick Start

### Basic Frame Extraction

```bash
# Browser-based extraction
python -m youtube_frame_extractor browser --video-id dQw4w9WgXcQ --interval 2 --max-frames 20

# Download-based extraction
python -m youtube_frame_extractor download --video-id dQw4w9WgXcQ --frame-rate 1 --max-frames 50
```

### AI-Powered Content Search

```bash
# Find frames matching a description
python -m youtube_frame_extractor vlm \
    --video-id dQw4w9WgXcQ \
    --query "person singing" \
    --threshold 0.3 \
    --method browser
```

### Batch Processing

```bash
# Process multiple videos in parallel
python -m youtube_frame_extractor batch \
    --video-ids dQw4w9WgXcQ 9bZkp7q19f0 kJQP7kiw5Fk \
    --method download \
    --frame-rate 0.5 \
    --worker-count 3
```

## 💻 Usage Examples

### Content-Based Frame Extraction with CLIP

```python
from youtube_frame_extractor.analysis import clip
from youtube_frame_extractor.extractors import browser

# Initialize CLIP analyzer
analyzer = clip.CLIPAnalyzer(
    model_name="openai/clip-vit-large-patch14",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Extract frames matching the description
extractor = browser.BrowserExtractor()
matching_frames = extractor.scan_video_for_frames(
    video_id="dQw4w9WgXcQ",
    search_query="close up of person singing",
    vlm_analyzer=analyzer,
    threshold=0.3
)

# Process matching frames
for frame in matching_frames:
    print(f"Frame at {frame['time']}s matched with score {frame['similarity']:.2f}")
```

### Object Detection in Video Frames

```python
from youtube_frame_extractor.analysis import object_detection
from youtube_frame_extractor.extractors import download

# Extract frames from video
extractor = download.DownloadExtractor()
frames = extractor.extract_frames(
    video_id="dQw4w9WgXcQ", 
    frame_rate=0.5
)

# Initialize object detector
detector = object_detection.ObjectDetector(
    model_name="faster_rcnn_resnet50_fpn",
    score_threshold=0.7
)

# Find frames containing people
for i, frame in enumerate(frames):
    detections = detector.detect_objects(frame["frame"])
    people = [obj for obj in detections if obj["label"] == 1]  # 1 = person in COCO
    
    if people:
        print(f"Frame {i} at {frame['time']}s contains {len(people)} people")
```

### Scene Change Detection

```python
from youtube_frame_extractor.utils import video
from youtube_frame_extractor.extractors import download

# Extract frames at higher rate for scene detection
extractor = download.DownloadExtractor()
frames = extractor.extract_frames(
    video_id="dQw4w9WgXcQ", 
    frame_rate=3.0  # 3 frames per second for better scene detection
)

# Detect scene changes
scene_changes = video.scene_change_detection(
    "path/to/downloaded/video.mp4",
    threshold=25.0,
    method="mse"
)

print(f"Detected {len(scene_changes)} scene changes")
```

### Cloud Storage Integration

```python
from youtube_frame_extractor.storage import cloud
from youtube_frame_extractor.extractors import browser

# Initialize cloud storage (AWS S3)
storage = cloud.CloudStorage(provider="aws")

# Extract frames
extractor = browser.BrowserExtractor()
frames = extractor.extract_frames(video_id="dQw4w9WgXcQ", interval=5)

# Upload frames to cloud
for frame in frames:
    remote_path = f"videos/{frame['video_id']}/{frame['time']:.2f}.jpg"
    storage.store_file(frame["path"], remote_path)
    print(f"Uploaded frame from {frame['time']}s to cloud storage")
```

## ⚙️ Configuration

The YouTube Frame Extractor provides three ways to configure its behavior:

### 1. Environment Variables

All settings can be set via environment variables with the `YFE_` prefix:

```bash
# Example .env file
YFE_BROWSER_HEADLESS=true
YFE_DOWNLOAD_TEMP_DIR=/tmp/yt_frames
YFE_VLM_DEFAULT_MODEL=openai/clip-vit-large-patch14
YFE_STORAGE_OUTPUT_DIR=./output/frames
```

### 2. Configuration Files

Use YAML or JSON configuration files for more detailed settings:

```yaml
# config.yaml
browser:
  headless: true
  browser_type: chrome
  selenium_timeout: 30
  
download:
  keep_video: false
  preferred_resolution: 720
  
vlm:
  model_name: openai/clip-vit-base-patch16
  device: cuda
  default_threshold: 0.3
  
storage:
  output_dir: ./output/frames
  use_cloud_storage: false
```

Use with: `python -m youtube_frame_extractor --config config.yaml ...`

### 3. Command Line Arguments

Override any setting via command line arguments:

```bash
python -m youtube_frame_extractor browser \
    --video-id dQw4w9WgXcQ \
    --headless false \
    --browser-type firefox \
    --output-dir ./custom_output
```

## 🏗️ Architecture

The project follows a modular, layered architecture:

```
┌─────────────────┐      ┌─────────────────┐      ┌────────────────┐
│     CLI Layer   │──────│  Extractor API  │──────│  Analysis API  │
└─────────────────┘      └─────────────────┘      └────────────────┘
        │                      │      │                   │
        │                      │      │                   │
        ▼                      ▼      ▼                   ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐
│  Configuration  │  │Browser Extract│  │Download  │  │ VLM/CLIP    │
└─────────────────┘  └──────────────┘  │ Extract  │  ├─────────────┤
        │                   │          └──────────┘  │Object Detect │
        │                   │               │        ├─────────────┤
        ▼                   │               │        │     OCR     │
┌─────────────────┐         │               │        └─────────────┘
│Exception Handling│        │               │               │
└─────────────────┘         │               │               │
        │                   │               │               │
        ▼                   ▼               ▼               ▼
    ┌───────────────────────────────────────────────────────────┐
    │                     Storage Layer                          │
    │                  (Local/AWS S3/GCS)                        │
    └───────────────────────────────────────────────────────────┘
```

### Key Components

#### Core Framework
- **CLI**: Command-line interface using Typer
- **Config**: Settings management with Pydantic
- **Logger**: Structured logging with Rich integration
- **Exceptions**: Comprehensive hierarchy for error handling

#### Extractors
- **Browser Extractor**: Selenium-based frame capture
- **Download Extractor**: yt-dlp + ffmpeg based extraction

#### Analysis
- **VLM**: Vision-Language Model interface
- **CLIP**: Implementation of OpenAI's CLIP
- **Object Detection**: Faster R-CNN integration
- **OCR**: Tesseract text extraction

#### Storage
- **Local Storage**: File system operations
- **Cloud Storage**: AWS S3 and GCS integration

#### Utilities
- **Browser Utils**: Selenium helpers
- **Concurrency**: Parallel processing tools
- **Image Utils**: Image manipulation
- **Video Utils**: Video operations and scene detection

## 📚 API Reference

For detailed API reference, please see our [API Documentation](docs/api-reference/).

Key modules include:

- `youtube_frame_extractor.extractors`: Frame extraction implementations
- `youtube_frame_extractor.analysis`: Content analysis tools
- `youtube_frame_extractor.storage`: Storage handling
- `youtube_frame_extractor.utils`: Utility functions and helpers

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-addition`
3. Make your changes and commit: `git commit -m 'Add some amazing feature'`
4. Push to your branch: `git push origin feature/amazing-addition`
5. Open a Pull Request with a detailed description

We especially welcome:
- 🧪 Additional test coverage
- 📚 Documentation improvements
- 🔧 Docker configuration refinements
- ⚡ Performance optimizations
- ✨ New analysis capabilities

Please follow the [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Harshil Bhandari**

- GitHub: [Harshil7875](https://github.com/Harshil7875)

## 🙏 Acknowledgments

This project leverages these amazing technologies:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [Selenium](https://www.selenium.dev/) for browser automation
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) for video processing
- [PyTorch](https://pytorch.org/) and [CLIP](https://github.com/openai/CLIP) for AI capabilities
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich) for CLI
- [Pydantic](https://pydantic-docs.helpmanual.io/) for validation
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) and [GCS](https://cloud.google.com/storage) for cloud storage