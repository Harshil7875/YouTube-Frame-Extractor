# Installation Guide

The **YouTube Frame Extractor** is a robust toolkit for extracting and analyzing frames from YouTube videos using both browser-based and download-based methods. This guide will walk you through the steps to set up the project on your local machine or within a Docker container.

---

## Prerequisites

Before you begin, make sure you have the following:

- **Python 3.8+**: The project requires Python version 3.8 or higher.
- **pip**: The package installer for Python.
- **Git**: To clone the repository.
- **Virtual Environment (recommended)**: For isolated dependency management.
- **Web Browser**: Chrome, Firefox, or Edge (for browser-based extraction).
- **ffmpeg**: A recent version installed on your system (for video processing).
- **Tesseract OCR**: Required for OCR functionality.
- **CUDA-compatible GPU (optional)**: For accelerated AI-powered analysis using PyTorch.

---

## Step-by-Step Installation

### 1. Clone the Repository

Clone the repository to your local machine using Git:

```bash
git clone https://github.com/Harshil7875/youtube-scrape-videoframe-analysis.git
cd youtube-scrape-videoframe-analysis
```

### 2. Create and Activate a Virtual Environment

It’s best practice to use a virtual environment. Create and activate one as follows:

```bash
# Create virtual environment
python -m venv venv

# Activate the virtual environment on macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

Install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install key packages such as yt-dlp, Selenium, OpenCV, PyTorch, CLIP, Tesseract OCR, and others.

### 4. (Optional) Install Additional System Dependencies

Depending on your operating system, you might need to install some system-level packages.

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg tesseract-ocr chromium chromium-driver libsm6 libxext6 libxrender1
```

#### On macOS:
- Install [Homebrew](https://brew.sh/) if you haven’t already.
- Then install:
  ```bash
  brew install ffmpeg tesseract
  ```
- For Chromium, install Google Chrome and ensure the corresponding driver is available.

### 5. (Optional) Build the Docker Image

For a consistent, containerized environment, build the Docker image:

```bash
docker build -t youtube-frame-extractor .
```

### 6. Configure Environment Variables

You can customize the extractor using environment variables. All variables should be prefixed with `YFE_`. For example:

```bash
export YFE_BROWSER_HEADLESS=true
export YFE_DOWNLOAD_TEMP_DIR=/path/to/temp
```

Alternatively, create a `.env` file in the project root with your custom settings.

### 7. Set the PYTHONPATH (If Needed)

If you encounter import errors when running example scripts, ensure your `PYTHONPATH` includes the `src` directory. From the project root, run:

```bash
export PYTHONPATH=./src
```

### 8. Test the Installation

Run one of the example scripts to verify that everything is set up correctly. For instance, to test browser-based extraction:

```bash
python3 examples/basic_extraction.py --video-id dQw4w9WgXcQ --method browser --interval 2 --frames 10
```

This command should initialize the browser, navigate to the YouTube video, and extract the specified number of frames.

---

## Troubleshooting

- **Module Not Found Errors:**  
  Ensure you run commands from the project root and that the `PYTHONPATH` includes the `src` directory:
  ```bash
  export PYTHONPATH=./src
  ```

- **Browser Issues:**  
  Verify that a compatible web browser is installed and that its driver (e.g., `chromedriver` for Chrome) is available in your system PATH.

- **System Dependency Errors:**  
  Confirm that `ffmpeg` and `tesseract` are installed and accessible from the command line.

---

## Further Reading

- [Project Documentation (index.md)](index.md)
- [README](../README.md)

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.