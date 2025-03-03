#!/usr/bin/env python3
"""
Custom Exceptions for YouTube Frame Extractor

This module defines custom exceptions used throughout the YouTube Frame Extractor package.
Using specific exception types helps in better error handling and debugging.
"""

class YouTubeFrameExtractorError(Exception):
    """Base exception for all YouTube Frame Extractor errors."""
    
    def __init__(self, message: str, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.message = message
        super().__init__(message, *args)
        
        # Store additional context if provided
        self.context = kwargs.get('context', {})
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.context:
            return f"{self.message} (Context: {self.context})"
        return self.message

# === Extraction Errors ===

class ExtractionError(YouTubeFrameExtractorError):
    """Base class for errors that occur during frame extraction."""
    pass

class BrowserExtractionError(ExtractionError):
    """Error during browser-based extraction."""
    
    def __init__(self, message: str, video_id: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            video_id: YouTube video ID that caused the error
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if video_id:
            context['video_id'] = video_id
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class DownloadExtractionError(ExtractionError):
    """Error during download-based extraction."""
    
    def __init__(self, message: str, video_id: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            video_id: YouTube video ID that caused the error
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if video_id:
            context['video_id'] = video_id
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class FrameExtractionError(ExtractionError):
    """Error while extracting specific frames."""
    
    def __init__(self, message: str, video_id: str = None, timestamp: float = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            video_id: YouTube video ID
            timestamp: Timestamp where the error occurred
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if video_id:
            context['video_id'] = video_id
        if timestamp is not None:
            context['timestamp'] = timestamp
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

# === YouTube Errors ===

class YouTubeError(YouTubeFrameExtractorError):
    """Base class for YouTube-specific errors."""
    pass

class VideoUnavailableError(YouTubeError):
    """Error when a YouTube video is unavailable."""
    
    def __init__(self, video_id: str, reason: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            video_id: YouTube video ID
            reason: Reason why the video is unavailable
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        message = f"YouTube video {video_id} is unavailable"
        if reason:
            message += f": {reason}"
        
        context = kwargs.get('context', {})
        context['video_id'] = video_id
        kwargs['context'] = context
        
        super().__init__(message, *args, **kwargs)

class VideoPrivateError(VideoUnavailableError):
    """Error when a YouTube video is private."""
    
    def __init__(self, video_id: str, *args, **kwargs):
        """Initialize with a specific reason for private videos."""
        super().__init__(video_id, reason="This video is private", *args, **kwargs)

class VideoGeoBlockedError(VideoUnavailableError):
    """Error when a YouTube video is geo-blocked."""
    
    def __init__(self, video_id: str, *args, **kwargs):
        """Initialize with a specific reason for geo-blocked videos."""
        super().__init__(video_id, reason="This video is not available in your country", *args, **kwargs)

class YouTubeRateLimitError(YouTubeError):
    """Error when YouTube rate limits are exceeded."""
    
    def __init__(self, message: str = "YouTube rate limit exceeded", *args, **kwargs):
        """Initialize with a default message for rate limiting."""
        super().__init__(message, *args, **kwargs)

# === Browser Errors ===

class BrowserError(YouTubeFrameExtractorError):
    """Base class for browser-related errors."""
    pass

class BrowserInitializationError(BrowserError):
    """Error initializing browser."""
    
    def __init__(self, message: str = "Failed to initialize browser", browser_type: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            browser_type: Type of browser being initialized
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if browser_type:
            context['browser_type'] = browser_type
            if message == "Failed to initialize browser":
                message = f"Failed to initialize {browser_type} browser"
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class ElementNotFoundError(BrowserError):
    """Error when a browser element is not found."""
    
    def __init__(self, element_descriptor: str, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            element_descriptor: Description of the element that wasn't found
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        message = f"Element not found: {element_descriptor}"
        context = kwargs.get('context', {})
        context['element'] = element_descriptor
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class JavaScriptExecutionError(BrowserError):
    """Error executing JavaScript in browser."""
    
    def __init__(self, script_info: str = None, error_message: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            script_info: Description of the script that failed
            error_message: Original JavaScript error message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        message = "Error executing JavaScript"
        if script_info:
            message += f" in {script_info}"
        if error_message:
            message += f": {error_message}"
        
        context = kwargs.get('context', {})
        if script_info:
            context['script_info'] = script_info
        if error_message:
            context['js_error'] = error_message
        kwargs['context'] = context
        
        super().__init__(message, *args, **kwargs)

# === Analysis Errors ===

class AnalysisError(YouTubeFrameExtractorError):
    """Base class for errors that occur during frame analysis."""
    pass

class VLMError(AnalysisError):
    """Error during Vision Language Model analysis."""
    
    def __init__(self, message: str, model_name: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            model_name: Name of the VLM model
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class ModelLoadError(VLMError):
    """Error loading a machine learning model."""
    
    def __init__(self, model_name: str, reason: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            model_name: Name of the model that failed to load
            reason: Reason for the failure
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, model_name, *args, **kwargs)

class OCRError(AnalysisError):
    """Error during Optical Character Recognition."""
    
    def __init__(self, message: str, *args, **kwargs):
        """Initialize the OCR error."""
        super().__init__(message, *args, **kwargs)

class ObjectDetectionError(AnalysisError):
    """Error during object detection."""
    
    def __init__(self, message: str, model_info: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            model_info: Information about the detection model
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if model_info:
            context['model_info'] = model_info
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

# === Storage Errors ===

class StorageError(YouTubeFrameExtractorError):
    """Base class for storage-related errors."""
    pass

class FileWriteError(StorageError):
    """Error writing to a file."""
    
    def __init__(self, file_path: str, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            file_path: Path to the file that couldn't be written
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        message = f"Error writing to file: {file_path}"
        context = kwargs.get('context', {})
        context['file_path'] = file_path
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class CloudStorageError(StorageError):
    """Error interacting with cloud storage."""
    
    def __init__(self, message: str, provider: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            provider: Cloud provider name (aws, gcp, azure)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if provider:
            context['provider'] = provider
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

# === Batch Processing Errors ===

class BatchProcessingError(YouTubeFrameExtractorError):
    """Base class for batch processing errors."""
    pass

class WorkerError(BatchProcessingError):
    """Error in a batch processing worker."""
    
    def __init__(self, message: str, worker_id: str = None, video_id: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            worker_id: Identifier for the worker
            video_id: YouTube video ID being processed
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if worker_id:
            context['worker_id'] = worker_id
        if video_id:
            context['video_id'] = video_id
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class BatchTimeoutError(BatchProcessingError):
    """Error when batch processing exceeds timeout."""
    
    def __init__(self, timeout_seconds: int, completed: int = None, total: int = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            timeout_seconds: Timeout in seconds
            completed: Number of completed tasks
            total: Total number of tasks
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        message = f"Batch processing timed out after {timeout_seconds} seconds"
        if completed is not None and total is not None:
            message += f" ({completed}/{total} tasks completed)"
        
        context = kwargs.get('context', {})
        context['timeout_seconds'] = timeout_seconds
        if completed is not None:
            context['completed'] = completed
        if total is not None:
            context['total'] = total
        kwargs['context'] = context
        
        super().__init__(message, *args, **kwargs)

# === Configuration Errors ===

class ConfigurationError(YouTubeFrameExtractorError):
    """Error in configuration settings."""
    
    def __init__(self, message: str, setting_name: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            setting_name: Name of the problematic setting
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if setting_name:
            context['setting'] = setting_name
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class ValidationError(ConfigurationError):
    """Error validating input parameters."""
    
    def __init__(self, message: str, parameter: str = None, value: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            parameter: Name of the invalid parameter
            value: Invalid value
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if parameter:
            context['parameter'] = parameter
        if value:
            context['value'] = value
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

# === Utility Function Errors ===

class UtilityError(YouTubeFrameExtractorError):
    """Base class for utility function errors."""
    pass

class FFmpegError(UtilityError):
    """Error running FFmpeg."""
    
    def __init__(self, message: str, command: str = None, exit_code: int = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            command: FFmpeg command that failed
            exit_code: FFmpeg exit code
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if command:
            context['command'] = command
        if exit_code is not None:
            context['exit_code'] = exit_code
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class YtDlpError(UtilityError):
    """Error running yt-dlp."""
    
    def __init__(self, message: str, video_id: str = None, error_output: str = None, *args, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            video_id: YouTube video ID
            error_output: Error output from yt-dlp
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        context = kwargs.get('context', {})
        if video_id:
            context['video_id'] = video_id
        if error_output:
            context['error_output'] = error_output
        kwargs['context'] = context
        super().__init__(message, *args, **kwargs)

class ImageProcessingError(UtilityError):
    """Error processing an image."""
    
    def __init__(self, message: str, *args, **kwargs):
        """Initialize the image processing error."""
        super().__init__(message, *args, **kwargs)

class ConcurrencyError(UtilityError):
    """Error in concurrent processing."""
    
    def __init__(self, message: str, *args, **kwargs):
        """Initialize the concurrency error."""
        super().__init__(message, *args, **kwargs)