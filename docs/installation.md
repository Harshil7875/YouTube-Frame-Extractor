# Installation Guide

This guide covers all the installation options for YouTube Frame Extractor, from basic setup to advanced configurations.

## Prerequisites

Before installing YouTube Frame Extractor, ensure you have the following prerequisites:

### Required

- Python 3.8 or newer
- pip (Python package installer)
- Internet connection for downloading dependencies

### Browser-based Extraction

If you plan to use browser-based extraction:

- Chrome, Firefox, or Edge browser installed
- Corresponding WebDriver (handled automatically when using `webdriver_manager`)

### Download-based Extraction

If you plan to use download-based extraction:

- ffmpeg installed and available in your system PATH
  - [FFmpeg Download Page](https://ffmpeg.org/download.html)

### Optional

- CUDA-compatible GPU for accelerated AI processing
- Tesseract OCR for text extraction capabilities
  - [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)

## Standard Installation

### From PyPI (Recommended)

```bash
pip install youtube-frame-extractor
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Harshil7875/YouTube-Frame-Extractor.git
cd YouTube-Frame-Extractor

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from local source
pip install -e .
```

## Feature-specific Installation

You can install specific feature sets based on your needs:

### Minimal Installation

```bash
pip install youtube-frame-extractor[minimal]
```

This installs only the core dependencies without AI analysis capabilities.

### Full Installation (All Features)

```bash
pip install youtube-frame-extractor[full]
```

This installs all dependencies, including CLIP, PyTorch, and other analysis tools.

### Custom Feature Sets

```bash
# For browser-based extraction only
pip install youtube-frame-extractor[browser]

# For download-based extraction only
pip install youtube-frame-extractor[download]

# For AI analysis capabilities
pip install youtube-frame-extractor[analysis]

# For cloud storage support
pip install youtube-frame-extractor[cloud]
```

## Docker Installation

We provide a Docker image for containerized operation, ensuring consistent environments:

```bash
# Build the Docker image
docker build -t youtube-frame-extractor .

# Run with Docker, mounting an output directory
docker run -v $(pwd)/output:/app/output youtube-frame-extractor [COMMANDS]
```

### Using Docker Compose

```bash
# Start using docker-compose
docker-compose up
```

## Post-Installation Verification

After installation, verify your setup with:

```bash
# Verify the package is installed
python -c "import youtube_frame_extractor; print(youtube_frame_extractor.__version__)"

# Run a simple extraction test
python -m youtube_frame_extractor.cli verify
```

## Troubleshooting

### Common Issues

#### Browser Driver Issues

If you encounter WebDriver issues:

```bash
# Install webdriver-manager to handle drivers automatically
pip install webdriver-manager
```

#### FFmpeg Not Found

If ffmpeg commands fail:

1. Ensure ffmpeg is installed and in your PATH
2. On Windows, you may need to restart your terminal after installation
3. Verify installation with `ffmpeg -version`

#### CUDA/GPU Issues

If you're having GPU acceleration problems:

1. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Update NVIDIA drivers if needed

### Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/Harshil7875/YouTube-Frame-Extractor/issues) for similar problems
2. Open a new issue with detailed information about your environment and the problem

## Next Steps

Now that you have installed YouTube Frame Extractor, you can:

- Check out the [Quick Start Guide](examples/quickstart.md)
- Explore [Example Scripts](examples/index.md)
- Learn about [Configuration Options](api-reference/config.md)