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

This toolkit gives you superpowers for capturing and analyzing video frames from YouTube through two powerful approaches:

1. **Browser-based extraction** 🌐: Captures frames directly from the YouTube player using Selenium, guided by a Vision Language Model to find exactly the content you're looking for!

2. **Download-based extraction** 📥: Downloads videos using yt-dlp and extracts frames with frame-perfect accuracy using ffmpeg or OpenCV!

## ✨ Awesome Features

🧠 **AI-Powered Content Detection** - Tell the system exactly what to look for ("person dancing," "car driving," "sunset") and it will find those frames using CLIP models!

🔎 **Object Detection** - Find people, cars, animals and more with Faster R-CNN integration

📝 **Text Extraction (OCR)** - Pull text from video frames using integrated Tesseract OCR

🎬 **Scene Change Detection** - Automatically identify scene changes using MSE, SSIM, or histogram comparison

☁️ **Cloud Storage Integration** - Store your frames in AWS S3 or Google Cloud Storage with built-in support

🚀 **Blazing Fast Parallel Processing** - Process multiple videos simultaneously with our ThreadPoolExecutor architecture

🎭 **Multi-Browser Support** - Chrome, Firefox, Edge - take your pick!

🔧 **Ridiculously Configurable** - Tweak everything via command line, environment variables, or config files

🐛 **Enterprise-Grade Error Handling** - Comprehensive exception system keeps your batch jobs running even when some videos fail

🐳 **Docker Ready** - Run in containers for consistent environments everywhere

## 🚀 Quick Installation

```bash
# Clone and get ready for action!
git clone https://github.com/Harshil7875/youtube-scrape-videoframe-analysis.git
cd youtube-scrape-videoframe-analysis

# Set up your environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the magic
pip install -r requirements.txt
```

### 🖥️ System Requirements

| Need | Details |
|------|---------|
| Python | 3.8+ (the newer the better!) |
| Browser | Chrome/Firefox/Edge for browser extraction |
| ffmpeg | Recent version for download extraction |
| Tesseract | For OCR functionality |
| GPU | CUDA-compatible for turbo-charged VLM processing |
| OS | Windows/macOS/Linux - we've got you covered! |

### 📦 Core Dependencies

| Type | Library | Version | Purpose |
|------|---------|---------|---------|
| **Video Processing** | yt-dlp | ≥2023.3.4 | Advanced YouTube downloading |
|  | ffmpeg-python | ≥0.2.0 | Video frame extraction |
|  | moviepy | ≥1.0.3 | Video editing and processing |
| **Web Automation** | selenium | ≥4.1.0 | Browser control for frame capture |
|  | webdriver-manager | ≥3.8.0 | Browser driver management |
| **Image Processing** | opencv-python | ≥4.7.0.72 | Computer vision operations |
|  | Pillow | ≥9.4.0 | Image processing |
|  | numpy | ≥1.24.2 | Numerical operations |
|  | scipy | ≥1.10.1 | Scientific computing |
| **AI & ML** | torch | ≥2.0.0 | Deep learning framework |
|  | torchvision | ≥0.15.0 | Computer vision models |
|  | transformers | ≥4.27.0 | NLP and vision models |
|  | CLIP | git | Vision-language models |
| **OCR** | pytesseract | ≥0.3.10 | Text extraction from images |
| **Web & API** | requests | ≥2.28.0 | HTTP requests |
|  | FastAPI | ≥0.95.0 | API framework |
|  | uvicorn | ≥0.21.0 | ASGI server |
| **CLI & UI** | typer[all] | ≥0.7.0 | Command-line interface |
|  | rich | ≥13.3.0 | Terminal UI and logging |
| **Configuration** | pydantic | ≥1.10.0 | Data validation and settings |
| **Storage** | boto3 | ≥1.26.0 | AWS S3 integration |
|  | google-cloud-storage | ≥2.7.0 | Google Cloud Storage |
| **Async** | aiohttp | ≥3.8.4 | Async HTTP client/server |
|  | asyncio | ≥3.4.3 | Async I/O support |
| **Testing** | pytest | ≥7.3.0 | Testing framework |
|  | pytest-cov | ≥4.1.0 | Test coverage |
| **Development** | black | ≥23.3.0 | Code formatting |
|  | isort | ≥5.12.0 | Import sorting |
|  | flake8 | ≥6.0.0 | Linting |
|  | mypy | ≥1.2.0 | Type checking |
| **Documentation** | mkdocs | ≥1.4.2 | Documentation generator |
|  | mkdocs-material | ≥9.1.0 | Material theme for MkDocs |
|  | mkdocstrings | ≥0.20.0 | API documentation |
| **Notebooks** | jupyter | ≥1.0.0 | Interactive notebooks |
|  | ipykernel | ≥6.22.0 | Jupyter kernel |
|  | matplotlib | ≥3.7.1 | Data visualization |
|  | seaborn | ≥0.12.2 | Statistical visualizations |

## 🎮 Command Line Superpowers

### 🌐 Browser-Based Extraction with VLM

```bash
python -m youtube_frame_extractor browser \
    --video-id dQw4w9WgXcQ \
    --query "close up of person singing" \
    --interval 2 \
    --threshold 0.3
```

### 📥 Download-Based Extraction

```bash
python -m youtube_frame_extractor download \
    --video-id dQw4w9WgXcQ \
    --frame-rate 1.0 \
    --max-frames 100 \
    --preferred-resolution 720
```

### 🚀 Batch Processing for Multiple Videos

```bash
python -m youtube_frame_extractor batch \
    --video-ids dQw4w9WgXcQ 9bZkp7q19f0 \
    --method browser \
    --query "person dancing" \
    --worker-count 4
```

### 🧠 AI-Powered Semantic Search

```bash
python -m youtube_frame_extractor vlm \
    --video-id dQw4w9WgXcQ \
    --query "person dancing" \
    --threshold 0.75 \
    --method download
```

## 🔥 Advanced API Tricks

### 🔍 CLIP-Based Frame Analysis

Search for frames using OpenAI's CLIP model:

```python
from youtube_frame_extractor.analysis import clip
from youtube_frame_extractor.extractors import browser

# Create a CLIP analyzer with custom settings
analyzer = clip.CLIPAnalyzer(
    model_name="openai/clip-vit-large-patch14",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    batch_size=16
)

# Use it to extract matching frames
extractor = browser.BrowserExtractor()
results = extractor.scan_video_for_frames(
    video_id="dQw4w9WgXcQ",
    search_query="person singing with microphone",
    vlm_analyzer=analyzer,
    threshold=0.28
)

# Process your amazing results
for frame in results:
    print(f"Frame at {frame.timestamp}s matched with score {frame.similarity:.2f}")
```

### 🔎 Object Detection

Detect objects in video frames using pre-trained models:

```python
from youtube_frame_extractor.analysis import object_detection
from youtube_frame_extractor.extractors import download

# Extract some frames first
extractor = download.DownloadExtractor()
frames = extractor.extract_frames(video_id="dQw4w9WgXcQ", frame_rate=0.5)

# Create the detector with custom settings
detector = object_detection.ObjectDetector(
    model_name="faster_rcnn_resnet50_fpn",
    confidence_threshold=0.7,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

# Detect objects in all frames
detection_results = detector.detect_objects_in_frames(frames)

# Find all frames with people
people_frames = [f for f in detection_results if any(obj["label"] == "person" for obj in f.objects)]
print(f"Found {len(people_frames)} frames with people!")
```

### 📝 OCR Text Extraction

Extract text from frames:

```python
from youtube_frame_extractor.analysis import ocr
from youtube_frame_extractor.extractors import download

# Get frames with potential text
extractor = download.DownloadExtractor()
frames = extractor.extract_frames(video_id="dQw4w9WgXcQ", frame_rate=0.2)

# Set up OCR analyzer
ocr_analyzer = ocr.OCRAnalyzer(
    language="eng",
    preprocess=True  # Apply image preprocessing for better results
)

# Extract text from frames
for i, frame in enumerate(frames):
    text = ocr_analyzer.extract_text(frame.image_path)
    if text.strip():  # Only show non-empty results
        print(f"Frame {i} at {frame.timestamp}s contains text: {text}")
```

### 🎬 Scene Change Detection

Detect scene changes within a video:

```python
from youtube_frame_extractor.utils import video
from youtube_frame_extractor.extractors import download

# Get frames at a high rate for scene detection
extractor = download.DownloadExtractor()
frames = extractor.extract_frames(
    video_id="dQw4w9WgXcQ", 
    frame_rate=5.0  # 5 frames per second
)

# Detect scene changes using different methods
scene_changes_mse = video.detect_scene_changes(
    frames, 
    method="mse", 
    threshold=25.0
)

scene_changes_ssim = video.detect_scene_changes(
    frames, 
    method="ssim", 
    threshold=0.7
)

scene_changes_hist = video.detect_scene_changes(
    frames, 
    method="histogram", 
    threshold=0.5
)

print(f"MSE detected {len(scene_changes_mse)} scene changes")
print(f"SSIM detected {len(scene_changes_ssim)} scene changes")
print(f"Histogram detected {len(scene_changes_hist)} scene changes")

# Extract key frame from each scene
key_frames = video.extract_scene_keyframes(frames, scene_changes_mse)
print(f"Extracted {len(key_frames)} key frames representing each scene")
```

### ☁️ Cloud Storage Integration

Store your frames in the cloud:

```python
from youtube_frame_extractor.storage import cloud
from youtube_frame_extractor.extractors import browser

# Set up cloud storage (AWS S3)
storage = cloud.CloudStorage(
    provider="s3",
    bucket_name="my-video-frames",
    region="us-west-2"
)

# Extract frames
extractor = browser.BrowserExtractor()
frames = extractor.extract_frames(video_id="dQw4w9WgXcQ", interval=5)

# Upload frames to cloud
for frame in frames:
    cloud_path = f"videos/{frame.video_id}/{frame.timestamp}.jpg"
    url = storage.store_frame(
        frame_path=frame.image_path,
        destination_path=cloud_path,
        metadata={"timestamp": frame.timestamp, "video_id": frame.video_id}
    )
    print(f"Uploaded frame to {url}")

# List all frames for a video
stored_frames = storage.list_frames(prefix=f"videos/dQw4w9WgXcQ/")
print(f"Found {len(stored_frames)} frames in storage")
```

### 🔄 Parallel Processing

Process multiple videos efficiently:

```python
from youtube_frame_extractor.utils import concurrency
from youtube_frame_extractor.extractors import browser

# List of video IDs to process
video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0", "JGwWNGJdvx8", "kJQP7kiw5Fk"]

# Function to process a single video
def process_video(video_id):
    extractor = browser.BrowserExtractor()
    try:
        frames = extractor.scan_video_for_frames(
            video_id=video_id,
            search_query="person dancing",
            threshold=0.3
        )
        return {"video_id": video_id, "frames": frames, "success": True}
    except Exception as e:
        return {"video_id": video_id, "error": str(e), "success": False}

# Process videos in parallel with 4 workers
results = concurrency.map_parallel(
    process_video,
    video_ids,
    max_workers=4,
    timeout=300  # 5 minute timeout per video
)

# Summarize results
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print(f"Successfully processed {len(successful)} videos")
print(f"Failed to process {len(failed)} videos")

# Print extracted frame counts
for result in successful:
    print(f"Video {result['video_id']}: {len(result['frames'])} matching frames")
```

## ⚙️ Super-Flexible Configuration

This toolkit gives you three ways to configure everything:

1. **Environment Variables** - Set with `YFE_` prefix (like `YFE_BROWSER_HEADLESS=true`)
2. **YAML/JSON Files** - Create config files with all your favorite settings
3. **Command Line Args** - Override anything on the fly

Example config file to supercharge your extraction:

```yaml
# Save as config.yaml and use with --config config.yaml

browser:
  headless: true
  browser_type: chrome
  timeout: 60
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  wait_for_video: 5
  scroll_into_view: true
  
download:
  cleanup: true
  preferred_resolution: 720
  format: mp4
  temp_dir: "/tmp/yt_extractor"
  retry_count: 3
  
vlm:
  model_name: openai/clip-vit-base-patch32
  use_cuda: true
  batch_size: 16
  device: "cuda:0"
  preload_model: true
  
object_detection:
  model_name: "faster_rcnn_resnet50_fpn"
  confidence_threshold: 0.7
  device: "cuda:0"
  
ocr:
  language: "eng"
  preprocess: true
  psm: 3
  
storage:
  provider: "local"  # Can be "local", "s3", or "gcs"
  base_path: "./frames"
  # Cloud-specific settings
  bucket_name: "my-video-frames"
  region: "us-west-2"
  
logging:
  level: INFO
  file: "extraction.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 🏗️ Core Architecture

Our toolkit has a clean, modular design that makes it easy to extend:

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

### 🧩 Key Components

#### 1. Core Framework

- **CLI** (`cli.py`): Typer-based commands with rich terminal output
- **Config** (`config.py`): Pydantic models with environment variable support
- **Logging** (`logger.py`): Rich-integrated logging with rotation
- **Exceptions** (`exceptions.py`): Comprehensive hierarchy for all error types

#### 2. Extractors

- **Base** (`extractors/base.py`): Abstract base class defining extraction interface
- **Browser** (`extractors/browser.py`): Selenium-based extraction
- **Download** (`extractors/download.py`): yt-dlp + ffmpeg extraction

#### 3. Analysis

- **VLM** (`analysis/vlm.py`): Vision-Language Model interface
- **CLIP** (`analysis/clip.py`): OpenAI CLIP implementation
- **Object Detection** (`analysis/object_detection.py`): Faster R-CNN detection
- **OCR** (`analysis/ocr.py`): Tesseract-based text extraction

#### 4. Storage

- **Local** (`storage/local.py`): File system operations
- **Cloud** (`storage/cloud.py`): AWS S3 and Google Cloud Storage

#### 5. Utilities

- **Browser** (`utils/browser.py`): Selenium helpers
- **Concurrency** (`utils/concurrency.py`): Parallel processing tools
- **Image** (`utils/image.py`): Image manipulation functions
- **Video** (`utils/video.py`): Video operations and scene detection

## 📂 Project Structure

The project follows a well-organized structure:

```
youtube-frame-extractor/
├── src/youtube_frame_extractor/    # Main package
│   ├── __main__.py                 # Entry point - launches the CLI
│   ├── cli.py                      # Command-line interface with Typer
│   ├── config.py                   # Pydantic-based config system
│   ├── exceptions.py               # Comprehensive exception hierarchy
│   ├── logger.py                   # Rich-integrated logging system
│   ├── extractors/                 # Frame extraction modules
│   │   ├── base.py                 # Abstract base extractor class
│   │   ├── browser.py              # Selenium-based YouTube extraction
│   │   └── download.py             # yt-dlp + ffmpeg based extraction
│   ├── analysis/                   # Frame analysis modules
│   │   ├── vlm.py                  # Vision Language Model interface
│   │   ├── clip.py                 # CLIP model implementation
│   │   ├── object_detection.py     # Object detection with Faster R-CNN
│   │   └── ocr.py                  # Optical character recognition
│   ├── storage/                    # Storage handling
│   │   ├── cloud.py                # AWS S3 & Google Cloud Storage
│   │   └── local.py                # Local filesystem storage
│   └── utils/                      # Utility functions
│       ├── browser.py              # Browser automation helpers
│       ├── concurrency.py          # Threading and parallelism tools
│       ├── image.py                # Image processing utilities
│       └── video.py                # Video handling functions
├── examples/                       # Example scripts and notebooks
│   ├── basic_extraction.py         # Simple extraction demos
│   ├── batch_processing.py         # Multi-video processing example
│   ├── vlm_analysis.py             # VLM-based semantic search demo
│   └── notebooks/                  # Jupyter notebooks
│       ├── quickstart.ipynb        # Getting started tutorial
│       └── advanced_analysis.ipynb # Advanced features walkthrough
├── tests/                          # Comprehensive test suite
│   ├── integration/                # End-to-end testing
│   ├── test_analysis/              # Tests for analysis modules
│   ├── test_extractors/            # Tests for extraction modules
│   └── test_utils/                 # Tests for utility functions
├── docs/                           # Documentation
│   ├── api-reference/             # API documentation (in progress)
│   ├── examples/                  # Usage examples
│   └── installation.md            # Installation guide
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose setup
├── Makefile                        # Development automation
└── requirements.txt                # Package dependencies
```

## 🔄 Development Status

The project is now **feature-complete** with all major components implemented:

✅ **Extractors**: Both browser and download extractors fully functional  
✅ **Analysis**: CLIP, OCR, Object Detection all implemented  
✅ **Storage**: Local and cloud (AWS/GCS) storage ready to use  
✅ **CLI**: Comprehensive command-line interface with all commands  
✅ **Configuration**: Flexible config system with environment variables  
✅ **Examples**: Demonstration scripts and notebooks available  
✅ **Utils**: Browser helpers, concurrency tools, image/video processing  
✅ **Tests**: Unit and integration tests for all major components

🔄 **In Progress**:
- More comprehensive API reference documentation
- Additional test coverage for edge cases
- Docker configuration refinements

## 🤝 Contributing

Join our awesome community of developers! Here's how to contribute:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-addition`
3. Write some amazing code!
4. Commit your changes: `git commit -m 'Add some amazing feature'`
5. Push to your branch: `git push origin feature/amazing-addition`
6. Open a Pull Request and describe your cool changes

We especially welcome:
- 🧪 Additional test implementations
- 📚 Documentation improvements
- 🔧 Docker configuration refinements
- ⚡ Performance optimizations
- ✨ New analysis capabilities

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Harshil Bhandari**

- GitHub: [Harshil7875](https://github.com/Harshil7875)

## 🙏 Acknowledgments

This project leverages these amazing technologies:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Powerful video downloading
- [Selenium](https://www.selenium.dev/) - Browser automation magic
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) - Video processing wizardry
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [Transformers](https://huggingface.co/transformers/) - NLP and vision models
- [CLIP](https://github.com/openai/CLIP) - OpenAI's amazing vision-language model
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Text extraction from images
- [Typer](https://typer.tiangolo.com/) - CLI creation simplified
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) - AWS SDK for Python
- [google-cloud-storage](https://googleapis.dev/python/storage/latest/index.html) - Google Cloud Storage SDK
- [pytest](https://docs.pytest.org/) - Testing framework
- [MkDocs](https://www.mkdocs.org/) - Documentation generator
- [Jupyter](https://jupyter.org/) - Interactive notebook environment
- [matplotlib](https://matplotlib.org/) - Data visualization
- [seaborn](https://seaborn.pydata.org/) - Statistical data visualization