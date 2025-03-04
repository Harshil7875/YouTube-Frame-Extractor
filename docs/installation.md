# Installation Guide

Follow these instructions to install YouTube Frame Extractor and its dependencies, preparing your system for efficient frame extraction and analysis from YouTube videos.

---

## System Requirements

Before installing, ensure your system meets the following requirements:

- **Python**: Version 3.8 or newer
- **ffmpeg**: Required for video processing and frame extraction
- **Browser**: Required for browser-based extraction (Chrome, Firefox, or Edge)
- **CUDA-compatible GPU** (optional): Recommended for faster AI processing

---

## Installation

### Step 1: Install Python

Download and install Python (version 3.8 or newer) from the [official Python website](https://www.python.org/downloads/).

Verify your installation:

```bash
python --version
```

---

### Step 2: Install ffmpeg

**Windows:** Download ffmpeg binaries from [FFmpeg's official site](https://ffmpeg.org/download.html#build-windows) and add it to your system PATH.

**macOS:**

Using Homebrew:

```bash
brew install ffmpeg
```

**Linux:**

Using apt (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify your installation:

```bash
ffmpeg -version
```

---

### Step 3: Install YouTube Frame Extractor

You can easily install YouTube Frame Extractor using pip:

```bash
pip install youtube-frame-extractor
```

If you plan to contribute or access the latest features, install from source:

```bash
git clone https://github.com/your-repo/youtube-frame-extractor.git
cd youtube-frame-extractor
pip install -e .
```

---

### Step 4: Install Browser and Drivers (Optional)

If using browser-based extraction, install one of the following browsers and its corresponding WebDriver:

- **Chrome:** [Download Chrome](https://www.google.com/chrome/) | [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
- **Firefox:** [Download Firefox](https://www.mozilla.org/firefox/new/) | [GeckoDriver](https://github.com/mozilla/geckodriver/releases)
- **Edge:** [Download Edge](https://www.microsoft.com/edge) | [EdgeDriver](https://developer.microsoft.com/microsoft-edge/tools/webdriver/)

For automatic WebDriver management, install `webdriver-manager`:

```bash
pip install webdriver-manager
```

---

### Step 5: Install AI Dependencies (Optional)

For content analysis, object detection, OCR, and VLM analysis, ensure the following dependencies are installed:

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install pytesseract
```

If you're using a GPU, install PyTorch with CUDA support from [PyTorch's official site](https://pytorch.org/get-started/locally/).

---

## Verification

Verify your installation by running a simple extraction command:

```python
from youtube_frame_extractor.extractors import download

extractor = download.DownloadExtractor()
frames = extractor.extract_frames(
    video_id="dQw4w9WgXcQ",
    frame_rate=1.0,
    max_frames=5
)

print(f"Extracted {len(frames)} frames successfully.")
```

---

## Troubleshooting

- **ffmpeg not found**: Ensure ffmpeg is installed and added to your PATH.
- **CUDA issues**: Verify CUDA installation and compatibility with PyTorch.
- **Browser WebDriver errors**: Confirm that WebDrivers are correctly installed or use `webdriver-manager`.

---

Now you're ready to extract and analyze frames from YouTube videos!

