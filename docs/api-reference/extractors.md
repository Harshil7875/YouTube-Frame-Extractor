# Extractors API Reference

Extractors are the components responsible for capturing frames from YouTube videos. Two primary methods are supported:

- **Browser-Based Extraction:** Uses Selenium to capture frames directly from the YouTube player without downloading the full video.
- **Download-Based Extraction:** Downloads the video using yt-dlp (or youtube-dl) and extracts frames from the downloaded file using ffmpeg.

This document details the interface and functionality of the extractors.

---

## BaseExtractor

The `BaseExtractor` is an abstract class that defines the common interface and shared functionality for all extractor implementations.

### Key Methods

- **`extract_frames(video_id: str, **kwargs) -> List[Dict[str, Any]]`**  
  _Abstract method._  
  Must be implemented by subclasses to extract frames from a YouTube video. Returns a list of dictionaries containing frame data and metadata.

- **`save_frames(frames: List[Dict[str, Any]], video_id: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]`**  
  Saves the extracted frames to disk in a structured directory. Updates each frame’s metadata with the file path and returns the updated list.

- **`scan_video_for_frames(video_id: str, search_query: str, vlm_analyzer, **kwargs) -> List[Dict[str, Any]]`**  
  A convenience method that first extracts frames and then uses a Vision Language Model (VLM) analyzer to filter frames based on a natural language query. It calculates similarity scores and returns only those frames that exceed a specified threshold.

- **`get_video_url(video_id: str) -> str`**  
  Constructs and returns the full YouTube URL for the given video ID.

- **`cleanup() -> None`**  
  Provides a hook for cleaning up any resources (such as closing browser drivers). Subclasses can override this if needed.

---

## BrowserExtractor

The `BrowserExtractor` implements frame extraction by automating a web browser using Selenium. It is designed to capture frames directly from the YouTube player.

### Key Features

- **Browser Initialization:**  
  Initializes a Selenium WebDriver for Chrome, Firefox, or Edge based on configuration settings. Supports headless mode and custom browser arguments.

- **Video Navigation and Playback Control:**  
  Navigates to the YouTube video page, waits for the video element to become ready, and controls playback (play, pause, mute, seek) to capture frames at specified intervals.

- **Frame Capture:**  
  Executes JavaScript in the browser to draw the current video frame onto a canvas. The canvas data (a base64-encoded JPEG) is then converted to a PIL Image.

- **VLM Integration:**  
  The `scan_video_for_frames` method uses a VLM analyzer (such as a CLIP-based analyzer) to compute similarity scores between captured frames and a text query, filtering and returning frames that match the desired content.

- **Error Handling:**  
  Raises custom exceptions like `BrowserExtractionError`, `ElementNotFoundError`, and `JavaScriptExecutionError` if issues occur during browser initialization, navigation, or frame capture.

---

## DownloadExtractor

The `DownloadExtractor` provides an alternative extraction method by downloading the full video and then extracting frames using ffmpeg.

### Key Features

- **Video Downloading:**  
  Downloads the video using yt-dlp (with a fallback to youtube-dl if necessary). Supports resolution preferences, progress callbacks, and temporary storage management.

- **Frame Extraction Using ffmpeg:**  
  Uses ffmpeg to extract frames at a specified frame rate or interval from the downloaded video. It calculates the appropriate interval based on the video duration and the desired number of frames, ensuring even distribution.

- **Metadata Handling:**  
  Saves extraction metadata (timestamps, frame count, extraction time) along with the frames. Optionally, the downloaded video file can be retained if needed.

- **VLM Integration:**  
  Similar to the browser extractor, provides a `scan_video_for_frames` method that post-processes extracted frames with a VLM analyzer, filtering based on similarity to a text prompt.

- **Error Handling:**  
  Raises errors such as `DownloadExtractionError`, `FFmpegError`, or `YtDlpError` if issues occur during video download or frame extraction.

---

By offering both browser-based and download-based extractors, YouTube Frame Extractor provides flexible options to suit various use cases—whether you require real-time frame capture without a full video download or higher-quality extraction from downloaded content. This modular design enables seamless integration with analysis tools and storage solutions, making the toolkit a powerful solution for video content analysis.
