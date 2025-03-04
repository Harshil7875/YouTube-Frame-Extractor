# Storage API Reference

This section describes how the YouTube Frame Extractor handles file storage. The system supports two storage options: cloud storage (for remote, scalable storage) and local storage (for file system–based storage).

---

## Cloud Storage

The `CloudStorage` class provides a unified interface to interact with cloud providers such as AWS S3 and Google Cloud Storage.

### Key Methods

- **`__init__(provider: Optional[str] = None)`**  
  Initializes the cloud storage client based on the provided cloud provider (e.g., "aws" or "gcp"). If no provider is specified, it uses the configuration from `settings.storage.cloud_provider`.

- **`store_file(local_path: str, remote_path: str) -> None`**  
  Uploads a local file to the cloud storage bucket.  
  **Usage Example:**
  ```python
  from youtube_frame_extractor.storage.cloud import CloudStorage

  storage = CloudStorage(provider="aws")
  storage.store_file(local_path="output/frame_1.jpg", remote_path="videos/video123/frame_1.jpg")
