# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies:
# - ffmpeg: for video processing
# - tesseract-ocr: for OCR functionality
# - Chromium & Chromium-driver: for Selenium browser-based extraction
# - Additional libraries for image/video support (libsm6, libxext6, libxrender1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    chromium \
    chromium-driver \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force headless mode for browser extraction
ENV YFE_BROWSER_HEADLESS=true
# Specify Chromium binary location for Selenium
ENV CHROME_BIN=/usr/bin/chromium

# Set work directory
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies.
# Also install CLIP directly from GitHub.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copy the source code into the container
COPY src/ /app/youtube_frame_extractor/

# Expose port if you use an API (e.g., FastAPI), uncomment the next line if needed.
# EXPOSE 8000

# Set the default command to run the CLI tool
CMD ["python", "-m", "youtube_frame_extractor"]
