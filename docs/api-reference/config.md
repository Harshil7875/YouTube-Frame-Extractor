# Configuration API Reference

The YouTube Frame Extractor leverages Pydantic for configuration management, ensuring that all settings are validated and easily adjustable. Settings can be provided via environment variables (prefixed with `YFE_`), configuration files (YAML/JSON), or command-line arguments, and are centralized in the global settings instance.

---

## General Settings

These settings apply to the application as a whole:

- **`version`**:  
  The current version of the application (e.g., `"1.0.0"`).

- **`default_method`**:  
  The default extraction method, typically `"browser"` or `"download"`.

- **`user_agent`**:  
  The default user agent string used by browser-based extraction.

---

## BrowserSettings

Settings specific to browser-based extraction using Selenium.

- **`headless`**:  
  *(bool)* Run the browser in headless mode (default: `true`).

- **`browser_type`**:  
  *(str)* Type of browser to use (e.g., `"chrome"`, `"firefox"`, `"edge"`).

- **`selenium_timeout`**:  
  *(int)* Maximum wait time (in seconds) for Selenium operations (default: `30`).

- **`wait_time`**:  
  *(float)* Time (in seconds) between Selenium actions (default: `0.5`).

- **`user_agent`**:  
  *(Optional[str])* Custom user agent string; if not set, the default is used.

- **`binary_location`**:  
  *(Optional[str])* Path to the browser binary if a non-standard location is used.

- **`extensions`**:  
  *(List[str])* List of browser extensions to load.

- **`arguments`**:  
  *(List[str])* Extra command-line arguments to pass to the browser (e.g., `"--mute-audio"`, `"--disable-infobars"`).

- **`disable_gpu`**:  
  *(bool)* Whether to disable GPU usage (default: `true`).

- **`default_interval`**:  
  *(float)* Default time interval between frame captures (in seconds, default: `2.0`).

- **`default_max_frames`**:  
  *(int)* Default maximum number of frames to extract (default: `50`).

---

## DownloadSettings

Settings for download-based extraction using yt-dlp and ffmpeg.

- **`use_ytdlp`**:  
  *(bool)* Whether to use yt-dlp (default: `true`).

- **`use_ffmpeg`**:  
  *(bool)* Whether to use ffmpeg for frame extraction (default: `true`).

- **`temp_dir`**:  
  *(str)* Directory for temporary video downloads (default: a user-specific path).

- **`keep_video`**:  
  *(bool)* Whether to keep the downloaded video after extraction (default: `false`).

- **`default_format`**:  
  *(str)* Preferred video format and resolution string (default: typically set to retrieve 720p video).

- **`default_frame_rate`**:  
  *(float)* Default frames per second to extract (default: `1.0`).

- **`default_max_frames`**:  
  *(int)* Default maximum number of frames to extract (default: `100`).

- **`timeout`**:  
  *(int)* Maximum time (in seconds) allowed for download operations (default: `300`).

- **`max_concurrent_downloads`**:  
  *(int)* Maximum number of parallel video downloads (default: `3`).

- **`retry_attempts`** and **`retry_delay`**:  
  *(int)* Control the number of retries and delay between attempts for download failures.

---

## VLMSettings

Settings for Vision-Language Models used for content analysis.

- **`default_model`**:  
  *(VLMModel)* The default model to use (e.g., `"openai/clip-vit-base-patch16"`).

- **`device`**:  
  *(str)* The processing device (`"cuda"` or `"cpu"`). Defaults to CUDA if available.

- **`default_threshold`**:  
  *(float)* The similarity threshold for filtering frames in VLM analysis (default: `0.3`).

- **`cache_dir`**:  
  *(str)* Directory for caching downloaded models (default: a user-specific path).

- **`batch_size`**:  
  *(int)* Batch size for VLM processing (default: `16`).

- **`preload_models`**:  
  *(bool)* Whether to preload models at startup (default: `false`).

- **`timeout`**:  
  *(int)* Timeout (in seconds) for VLM operations (default: `30`).

---

## StorageSettings

Settings that determine where and how extracted frames and metadata are stored.

- **`output_dir`**:  
  *(str)* Base directory for saving extracted frames (default: typically set to `~/youtube_frame_extractor/output`).

- **`use_cloud_storage`**:  
  *(bool)* Whether to use cloud storage for saving files (default: `false`).

- **`cloud_provider`**:  
  *(Optional[str])* Specifies the cloud provider (e.g., `"aws"`, `"gcp"`, or `"azure"`).

- **Provider-specific Settings:**  
  - **AWS:**  
    - **`aws_bucket`**: *(Optional[str])* Name of the AWS S3 bucket.
    - **`aws_region`**: *(Optional[str])* AWS region for the S3 bucket.
  - **GCP:**  
    - **`gcs_bucket`**: *(Optional[str])* Name of the Google Cloud Storage bucket.
  - **Azure:**  
    - *(Optional)* Future settings for Azure or other providers.

---

## LoggingSettings

Settings for controlling application logging behavior.

- **`level`**:  
  *(str)* Logging level (e.g., `"INFO"`, `"DEBUG"`).

- **`format`**:  
  *(str)* Log message format.

- **`file`**:  
  *(Optional[str])* If set, log messages will also be written to this file.

- **`console`**:  
  *(bool)* Whether to log to the console (default: `true`).

- **`rich_formatting`**:  
  *(bool)* Whether to use Rich for enhanced console output (default: `true`).

---

## BatchSettings

Settings that control batch processing of multiple videos.

- **`default_workers`**:  
  *(int)* Default number of concurrent workers for batch processing (default: `3`).

- **`max_workers`**:  
  *(int)* Maximum number of workers allowed (default: `10`).

- **`generate_report`**:  
  *(bool)* Whether to generate a summary report after batch processing (default: `true`).

- **`report_format`**:  
  *(str)* Format of the report (e.g., `"markdown"`, `"html"`, `"json"`).

- **`timeout`**:  
  *(int)* Overall timeout (in seconds) for batch operations (default: `3600`).

---

## Overriding Settings

You can customize configuration settings through several methods:

1. **Environment Variables:**  
   All settings support overrides via environment variables. For example, to disable headless mode in browser extraction:
   ```bash
   export YFE_BROWSER_HEADLESS=false
