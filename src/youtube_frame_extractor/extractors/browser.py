#!/usr/bin/env python3
"""
Browser-based YouTube Frame Extractor

This module implements a frame extractor that captures frames directly
from the YouTube player using Selenium browser automation.
"""

import os
import sys
import time
import base64
from io import BytesIO
import json
import traceback
from typing import List, Dict, Any, Optional, Callable, Union
import tempfile

from PIL import Image
import numpy as np

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        TimeoutException, NoSuchElementException, 
        WebDriverException, JavascriptException
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from .base import BaseExtractor
from ..exceptions import (
    BrowserExtractionError, BrowserInitializationError,
    ElementNotFoundError, JavaScriptExecutionError,
    VideoUnavailableError
)
from ..logger import get_logger
from ..config import get_settings

# Get logger and settings
logger = get_logger(__name__)
settings = get_settings()

class BrowserExtractor(BaseExtractor):
    """
    Browser-based frame extraction from YouTube videos.
    
    This extractor uses Selenium to control a web browser and capture
    frames directly from the YouTube player, without downloading the
    full video.
    """
    
    def __init__(
        self, 
        output_dir: str = "output", 
        headless: bool = True,
        browser_type: str = "chrome",
        selenium_timeout: int = 30
    ):
        """
        Initialize the browser-based extractor.
        
        Args:
            output_dir: Directory to save extracted frames
            headless: Whether to run browser in headless mode
            browser_type: Type of browser to use (chrome, firefox, edge)
            selenium_timeout: Timeout for selenium operations (seconds)
        """
        super().__init__(output_dir)
        
        if not SELENIUM_AVAILABLE:
            raise ImportError(
                "Selenium is required for browser-based extraction. "
                "Install it with: pip install selenium"
            )
        
        self.headless = headless
        self.browser_type = browser_type.lower()
        self.selenium_timeout = selenium_timeout
        self._driver = None
        
        # Store additional settings from config
        self.browser_settings = settings.browser
        
        logger.info(f"Initialized browser-based extractor (headless: {headless}, browser: {browser_type})")
    
    def _initialize_driver(self):
        """
        Initialize the Selenium WebDriver if not already initialized.
        
        This method is called automatically when needed, but can also
        be called manually to pre-initialize the browser.
        """
        if self._driver is not None:
            return
        
        logger.info(f"Initializing {self.browser_type} browser")
        
        try:
            if self.browser_type == "chrome":
                self._initialize_chrome_driver()
            elif self.browser_type == "firefox":
                self._initialize_firefox_driver()
            elif self.browser_type == "edge":
                self._initialize_edge_driver()
            else:
                raise BrowserInitializationError(
                    f"Unsupported browser type: {self.browser_type}",
                    browser_type=self.browser_type
                )
            
            # Set page load timeout
            self._driver.set_page_load_timeout(self.selenium_timeout)
            
            logger.info(f"Successfully initialized {self.browser_type} browser")
            
        except Exception as e:
            logger.error(f"Error initializing browser: {str(e)}")
            raise BrowserInitializationError(
                f"Failed to initialize {self.browser_type} browser: {str(e)}",
                browser_type=self.browser_type
            )
    
    def _initialize_chrome_driver(self):
        """Initialize Chrome WebDriver."""
        from selenium.webdriver.chrome.service import Service
        
        try:
            # Try to use webdriver_manager for automatic chromedriver management
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        except ImportError:
            # Fall back to system path
            service = Service()
        
        options = ChromeOptions()
        
        # Add browser settings
        if self.headless:
            options.add_argument("--headless=new")
        
        if self.browser_settings.disable_gpu:
            options.add_argument("--disable-gpu")
        
        # Add browser arguments from settings
        for arg in self.browser_settings.arguments:
            options.add_argument(arg)
        
        # Set custom user agent if specified
        if self.browser_settings.user_agent:
            options.add_argument(f"--user-agent={self.browser_settings.user_agent}")
        
        # Set binary location if specified
        if self.browser_settings.binary_location:
            options.binary_location = self.browser_settings.binary_location
        
        # Load extensions if specified
        for extension in self.browser_settings.extensions:
            options.add_extension(extension)
        
        self._driver = webdriver.Chrome(service=service, options=options)
    
    def _initialize_firefox_driver(self):
        """Initialize Firefox WebDriver."""
        from selenium.webdriver.firefox.service import Service
        
        try:
            # Try to use webdriver_manager for automatic geckodriver management
            from webdriver_manager.firefox import GeckoDriverManager
            service = Service(GeckoDriverManager().install())
        except ImportError:
            # Fall back to system path
            service = Service()
        
        options = FirefoxOptions()
        
        # Add browser settings
        if self.headless:
            options.add_argument("--headless")
        
        # Set custom user agent if specified
        if self.browser_settings.user_agent:
            options.set_preference("general.useragent.override", self.browser_settings.user_agent)
        
        self._driver = webdriver.Firefox(service=service, options=options)
    
    def _initialize_edge_driver(self):
        """Initialize Edge WebDriver."""
        from selenium.webdriver.edge.service import Service
        
        try:
            # Try to use webdriver_manager for automatic edgedriver management
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
            service = Service(EdgeChromiumDriverManager().install())
        except ImportError:
            # Fall back to system path
            service = Service()
        
        options = EdgeOptions()
        
        # Add browser settings
        if self.headless:
            options.add_argument("--headless")
        
        if self.browser_settings.disable_gpu:
            options.add_argument("--disable-gpu")
        
        # Add browser arguments from settings
        for arg in self.browser_settings.arguments:
            options.add_argument(arg)
        
        # Set custom user agent if specified
        if self.browser_settings.user_agent:
            options.add_argument(f"--user-agent={self.browser_settings.user_agent}")
        
        self._driver = webdriver.Edge(service=service, options=options)
    
    def _close_driver(self):
        """Close the Selenium WebDriver if it's open."""
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception as e:
                logger.warning(f"Error closing browser: {str(e)}")
            finally:
                self._driver = None
                logger.info("Closed browser")
    
    def cleanup(self):
        """Clean up resources (close browser)."""
        self._close_driver()
    
    def _capture_frame(self) -> Image.Image:
        """
        Capture the current frame from the YouTube player.
        
        Returns:
            PIL Image object of the captured frame
        """
        if self._driver is None:
            raise BrowserExtractionError("Browser not initialized")
        
        # Execute JavaScript to capture the video element as a data URL
        script = """
        (function() {
            const video = document.querySelector('video');
            if (!video) {
                return { error: 'Video element not found' };
            }
            
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                return { error: 'Video has zero dimensions' };
            }
            
            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                return { 
                    dataUrl: canvas.toDataURL('image/jpeg', 0.95),
                    width: canvas.width,
                    height: canvas.height,
                    currentTime: video.currentTime
                };
            } catch (e) {
                return { error: e.toString() };
            }
        })();
        """
        
        try:
            result = self._driver.execute_script(script)
            
            if not result:
                raise JavaScriptExecutionError("Script returned null result")
            
            if 'error' in result:
                raise JavaScriptExecutionError(
                    f"Error capturing frame: {result['error']}",
                    error_message=result['error']
                )
            
            # Extract data URL and convert to PIL Image
            data_url = result['dataUrl']
            if not data_url.startswith('data:image/'):
                raise JavaScriptExecutionError("Invalid data URL format")
            
            # Extract base64 data from data URL
            image_data = base64.b64decode(data_url.split(',')[1])
            
            # Create PIL Image
            image = Image.open(BytesIO(image_data))
            
            # Add current time as metadata
            metadata = {'currentTime': result.get('currentTime', 0)}
            
            return image, metadata
            
        except JavaScriptExecutionError:
            # Re-raise JavaScript-specific errors
            raise
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            raise BrowserExtractionError(f"Failed to capture frame: {str(e)}")
    
    def _get_current_timestamp(self) -> float:
        """
        Get the current timestamp from the YouTube player.
        
        Returns:
            Current playback time in seconds
        """
        if self._driver is None:
            raise BrowserExtractionError("Browser not initialized")
        
        script = "return document.querySelector('video').currentTime;"
        
        try:
            timestamp = self._driver.execute_script(script)
            return float(timestamp)
        except Exception as e:
            logger.error(f"Error getting timestamp: {str(e)}")
            raise JavaScriptExecutionError(
                "Failed to get current timestamp",
                script_info="get_current_timestamp",
                error_message=str(e)
            )
    
    def _wait_for_video_element(self, timeout: int = None) -> bool:
        """
        Wait for the video element to be present and ready.
        
        Args:
            timeout: Wait timeout in seconds (None for default timeout)
            
        Returns:
            True if video element is ready
        """
        if timeout is None:
            timeout = self.selenium_timeout
        
        try:
            # Wait for video element to be present
            WebDriverWait(self._driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            
            # Check if video is ready using JavaScript
            script = """
            const video = document.querySelector('video');
            return {
                ready: video && video.readyState >= 2,
                paused: video ? video.paused : true,
                videoWidth: video ? video.videoWidth : 0,
                videoHeight: video ? video.videoHeight : 0,
                currentTime: video ? video.currentTime : 0,
                duration: video ? video.duration : 0
            };
            """
            
            max_attempts = 5
            for attempt in range(max_attempts):
                result = self._driver.execute_script(script)
                
                if result.get('ready', False) and result.get('videoWidth', 0) > 0:
                    # Video is ready and has valid dimensions
                    logger.info(f"Video element ready: {result}")
                    return True
                
                # If not ready, wait a bit
                time.sleep(1)
            
            # If we reach here, video element exists but isn't fully ready
            logger.warning(f"Video element found but not ready after {max_attempts} attempts")
            return False
            
        except TimeoutException:
            logger.error(f"Timeout waiting for video element after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error waiting for video element: {str(e)}")
            return False
    
    def _handle_video_availability(self):
        """
        Check if the video is available or if there are error messages.
        Raises appropriate exceptions for unavailable videos.
        """
        try:
            # Check for common YouTube error messages
            error_messages = [
                (By.XPATH, "//div[contains(text(), 'This video is unavailable')]"),
                (By.XPATH, "//div[contains(text(), 'Video unavailable')]"),
                (By.XPATH, "//div[contains(text(), 'This video is no longer available')]"),
                (By.XPATH, "//div[contains(text(), 'This video is private')]"),
                (By.XPATH, "//div[contains(text(), 'This video has been removed')]")
            ]
            
            for locator in error_messages:
                try:
                    # Use a short timeout for checking error messages
                    element = WebDriverWait(self._driver, 2).until(
                        EC.presence_of_element_located(locator)
                    )
                    
                    error_text = element.text
                    logger.error(f"Video unavailable: {error_text}")
                    
                    if "private" in error_text.lower():
                        raise VideoUnavailableError(
                            "This video is private", 
                            reason="Private video"
                        )
                    elif "removed" in error_text.lower():
                        raise VideoUnavailableError(
                            "This video has been removed", 
                            reason="Video removed"
                        )
                    else:
                        raise VideoUnavailableError(
                            "Video is unavailable", 
                            reason=error_text
                        )
                except TimeoutException:
                    # Error message not found, continue checking others
                    continue
        
        except (TimeoutException, NoSuchElementException):
            # No error messages found, video might be available
            pass
        except VideoUnavailableError:
            # Re-raise video availability exceptions
            raise
        except Exception as e:
            # Log other exceptions but don't raise, as this is just a check
            logger.warning(f"Error checking video availability: {str(e)}")
    
    def _control_playback(self, action: str) -> bool:
        """
        Control video playback (play, pause, mute, etc.).
        
        Args:
            action: Playback action ('play', 'pause', 'mute', 'unmute')
            
        Returns:
            True if action was successful
        """
        actions = {
            'play': "if(video.paused) video.play();",
            'pause': "if(!video.paused) video.pause();",
            'mute': "video.muted = true;",
            'unmute': "video.muted = false;",
            'reset': "video.currentTime = 0;"
        }
        
        if action not in actions:
            logger.error(f"Invalid playback action: {action}")
            return False
        
        script = f"const video = document.querySelector('video'); {actions[action]} return true;"
        
        try:
            result = self._driver.execute_script(script)
            return bool(result)
        except Exception as e:
            logger.error(f"Error controlling playback ({action}): {str(e)}")
            return False
    
    def _seek_to_time(self, time_seconds: float) -> bool:
        """
        Seek to a specific time in the video.
        
        Args:
            time_seconds: Time to seek to in seconds
            
        Returns:
            True if seek was successful
        """
        script = f"document.querySelector('video').currentTime = {time_seconds}; return true;"
        
        try:
            result = self._driver.execute_script(script)
            # Wait a short time for seek to complete
            time.sleep(0.2)
            return bool(result)
        except Exception as e:
            logger.error(f"Error seeking to time {time_seconds}: {str(e)}")
            return False
    
    def _get_video_duration(self) -> float:
        """
        Get the total duration of the current video.
        
        Returns:
            Video duration in seconds
        """
        script = "return document.querySelector('video').duration;"
        
        try:
            duration = self._driver.execute_script(script)
            return float(duration)
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            raise JavaScriptExecutionError(
                "Failed to get video duration",
                script_info="get_video_duration",
                error_message=str(e)
            )
    
    def extract_frames(
        self,
        video_id: str,
        interval: float = 2.0,
        max_frames: int = 50,
        duration: Optional[float] = None,
        start_time: float = 0.0,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from a YouTube video at regular intervals.
        
        Args:
            video_id: YouTube video ID
            interval: Time interval between frames in seconds
            max_frames: Maximum number of frames to extract
            duration: Duration of video to process (None for full video)
            start_time: Time to start extraction from (seconds)
            progress_callback: Optional callback function that receives the frame number
            
        Returns:
            List of dictionaries containing frame data and metadata
        """
        logger.info(f"Extracting frames from video {video_id} using browser method")
        
        # Initialize the driver if not already
        self._initialize_driver()
        
        video_url = self.get_video_url(video_id)
        frames = []
        
        try:
            # Navigate to the video
            logger.info(f"Navigating to video: {video_url}")
            self._driver.get(video_url)
            
            # Wait for video element to be ready
            if not self._wait_for_video_element():
                logger.error("Video element not ready after waiting")
                raise ElementNotFoundError("Video element not ready")
            
            # Check if video is available
            self._handle_video_availability()
            
            # Get video duration
            video_duration = self._get_video_duration()
            logger.info(f"Video duration: {video_duration:.2f} seconds")
            
            # Set duration limit if specified
            if duration is not None and duration < video_duration:
                end_time = start_time + duration
            else:
                end_time = video_duration
            
            # Seek to start time
            if start_time > 0:
                logger.info(f"Seeking to start time: {start_time:.2f} seconds")
                self._seek_to_time(start_time)
            
            # Start playback
            self._control_playback('play')
            self._control_playback('mute')  # Mute audio
            
            # Calculate number of frames to extract
            if interval <= 0:
                raise ValueError("Interval must be greater than 0")
            
            total_duration = end_time - start_time
            potential_frames = int(total_duration / interval)
            num_frames = min(potential_frames, max_frames) if max_frames > 0 else potential_frames
            
            logger.info(f"Will extract up to {num_frames} frames at {interval}s intervals")
            
            # Extract frames at specified intervals
            current_time = start_time
            frame_count = 0
            
            while current_time <= end_time and (max_frames <= 0 or frame_count < max_frames):
                # Seek to the exact time we want to capture
                self._seek_to_time(current_time)
                
                # Capture frame
                image, metadata = self._capture_frame()
                actual_time = metadata.get('currentTime', current_time)
                
                # Create frame data
                frame_data = {
                    'frame': image,
                    'time': actual_time,
                    'index': frame_count
                }
                
                frames.append(frame_data)
                frame_count += 1
                
                # Call progress callback if provided
                if progress_callback and frame_count % 5 == 0:  # Update every 5 frames
                    progress_callback(frame_count)
                
                # Move to next interval
                current_time += interval
            
            # Pause playback when done
            self._control_playback('pause')
            
            logger.info(f"Successfully extracted {len(frames)} frames from video {video_id}")
            
            # Save frames to output directory
            video_output_dir = os.path.join(self.output_dir, video_id)
            self.save_frames(frames, video_id, video_output_dir)
            
            return frames
            
        except VideoUnavailableError:
            # Re-raise video availability errors
            raise
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            # Include traceback for debugging
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            raise BrowserExtractionError(
                f"Error extracting frames from video {video_id}: {str(e)}",
                video_id=video_id
            )
    
    def scan_video_for_frames(
        self,
        video_id: str,
        search_query: str,
        vlm_analyzer,  # Type hint omitted to avoid circular import
        interval: float = 2.0,
        threshold: float = 0.3,
        max_frames: int = 50,
        duration: Optional[float] = None,
        start_time: float = 0.0,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find frames in a video that match a specific description.
        
        This method is more efficient than the base class implementation
        as it analyzes frames during extraction.
        
        Args:
            video_id: YouTube video ID
            search_query: Natural language description to search for
            vlm_analyzer: Vision Language Model analyzer instance
            interval: Time interval between frames in seconds
            threshold: Similarity threshold for matching (0.0 to 1.0)
            max_frames: Maximum number of frames to extract
            duration: Duration of video to process (None for full video)
            start_time: Time to start extraction from (seconds)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of frames that match the search query
        """
        logger.info(f"Scanning video {video_id} for frames matching: '{search_query}'")
        
        # Initialize the driver if not already
        self._initialize_driver()
        
        video_url = self.get_video_url(video_id)
        matched_frames = []
        analyzed_count = 0
        
        try:
            # Navigate to the video
            logger.info(f"Navigating to video: {video_url}")
            self._driver.get(video_url)
            
            # Wait for video element to be ready
            if not self._wait_for_video_element():
                logger.error("Video element not ready after waiting")
                raise ElementNotFoundError("Video element not ready")
            
            # Check if video is available
            self._handle_video_availability()
            
            # Get video duration
            video_duration = self._get_video_duration()
            logger.info(f"Video duration: {video_duration:.2f} seconds")
            
            # Set duration limit if specified
            if duration is not None and duration < video_duration:
                end_time = start_time + duration
            else:
                end_time = video_duration
            
            # Seek to start time
            if start_time > 0:
                logger.info(f"Seeking to start time: {start_time:.2f} seconds")
                self._seek_to_time(start_time)
            
            # Start playback
            self._control_playback('play')
            self._control_playback('mute')  # Mute audio
            
            # Calculate number of frames to analyze
            if interval <= 0:
                raise ValueError("Interval must be greater than 0")
            
            total_duration = end_time - start_time
            potential_frames = int(total_duration / interval)
            total_frames = min(potential_frames, max_frames) if max_frames > 0 else potential_frames
            
            logger.info(f"Will analyze up to {total_frames} frames at {interval}s intervals")
            
            # Analyze frames at specified intervals
            current_time = start_time
            
            while current_time <= end_time and (max_frames <= 0 or analyzed_count < max_frames):
                # Seek to the exact time we want to capture
                self._seek_to_time(current_time)
                
                # Capture frame
                image, metadata = self._capture_frame()
                actual_time = metadata.get('currentTime', current_time)
                
                # Update analysis count
                analyzed_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(analyzed_count)
                
                # Analyze the frame with VLM
                similarity = vlm_analyzer.calculate_similarity(image, search_query)
                
                logger.debug(f"Frame at {actual_time:.2f}s: similarity = {similarity:.4f}")
                
                # If similarity is above threshold, save the frame
                if similarity >= threshold:
                    # Create frame data
                    frame_data = {
                        'frame': image,
                        'time': actual_time,
                        'similarity': float(similarity),
                        'query': search_query,
                        'threshold': threshold,
                        'index': analyzed_count - 1
                    }
                    
                    matched_frames.append(frame_data)
                    logger.info(f"Found matching frame at {actual_time:.2f}s (similarity: {similarity:.4f})")
                
                # Move to next interval
                current_time += interval
            
            # Pause playback when done
            self._control_playback('pause')
            
            logger.info(f"Analyzed {analyzed_count} frames, found {len(matched_frames)} matches above threshold {threshold}")
            
            # Sort frames by similarity score (highest first)
            matched_frames.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Save matched frames to output directory
            video_output_dir = os.path.join(self.output_dir, video_id)
            self.save_frames(matched_frames, video_id, video_output_dir)
            
            # Save VLM metadata
            metadata_path = os.path.join(video_output_dir, f"{video_id}_vlm_metadata.json")
            
            metadata = {
                "video_id": video_id,
                "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "search_query": search_query,
                "interval": interval,
                "threshold": threshold,
                "total_frames_analyzed": analyzed_count,
                "matching_frames": len(matched_frames),
                "frames": [
                    {
                        "time": frame.get("time", 0),
                        "similarity": frame.get("similarity", 0),
                        "index": frame.get("index", 0),
                        "path": frame.get("path", "")
                    }
                    for frame in matched_frames
                ]
            }
            
            try:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Error saving VLM metadata: {str(e)}")
            
            return matched_frames
            
        except VideoUnavailableError:
            # Re-raise video availability errors
            raise
        except Exception as e:
            logger.error(f"Error scanning video for frames: {str(e)}")
            # Include traceback for debugging
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            raise BrowserExtractionError(
                f"Error scanning video {video_id} for frames: {str(e)}",
                video_id=video_id
            )