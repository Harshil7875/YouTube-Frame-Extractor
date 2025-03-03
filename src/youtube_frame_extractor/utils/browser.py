#!/usr/bin/env python3
"""
Browser Utility Module for YouTube Frame Extractor

This module provides helper functions for managing Selenium WebDriver instances
and performing common browser-related tasks.
"""

import time
import base64
from typing import Optional

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        TimeoutException, WebDriverException, JavascriptException
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    # webdriver_manager is optional but helps automate driver installation
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
except ImportError:
    ChromeDriverManager = None
    GeckoDriverManager = None
    EdgeChromiumDriverManager = None

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import BrowserError, BrowserInitializationError, JavaScriptExecutionError

logger = get_logger(__name__)
settings = get_settings()


def check_selenium_available() -> None:
    """
    Check if Selenium is installed and available.
    Raises an ImportError if not.
    """
    if not SELENIUM_AVAILABLE:
        raise ImportError(
            "Selenium is required for browser utilities. "
            "Install it with: pip install selenium"
        )


def create_driver(
    browser_type: str = "chrome",
    headless: bool = True,
    extra_args: Optional[list] = None,
    timeout: int = 30
):
    """
    Create and return a Selenium WebDriver instance for the specified browser.

    Args:
        browser_type: "chrome", "firefox", or "edge"
        headless: Run browser in headless mode if True
        extra_args: Additional browser arguments to pass
        timeout: Page load timeout in seconds

    Returns:
        A Selenium WebDriver instance.

    Raises:
        BrowserInitializationError: If the driver cannot be initialized.
    """
    check_selenium_available()

    browser_type = browser_type.lower()
    extra_args = extra_args or []

    logger.info(f"Creating Selenium driver for '{browser_type}' (headless={headless})")

    try:
        if browser_type == "chrome":
            return _create_chrome_driver(headless, extra_args, timeout)
        elif browser_type == "firefox":
            return _create_firefox_driver(headless, extra_args, timeout)
        elif browser_type == "edge":
            return _create_edge_driver(headless, extra_args, timeout)
        else:
            raise BrowserInitializationError(
                f"Unsupported browser type: {browser_type}",
                browser_type=browser_type
            )
    except Exception as e:
        logger.error(f"Error creating driver for {browser_type}: {str(e)}")
        raise BrowserInitializationError(
            f"Failed to initialize {browser_type} driver: {str(e)}",
            browser_type=browser_type
        )


def quit_driver(driver) -> None:
    """
    Cleanly quit the Selenium WebDriver instance.

    Args:
        driver: The WebDriver to quit.
    """
    if driver:
        try:
            driver.quit()
            logger.info("Successfully quit the Selenium WebDriver.")
        except Exception as e:
            logger.warning(f"Error quitting browser driver: {str(e)}")


def wait_for_element(
    driver, by_locator: tuple, timeout: int = 30, poll_frequency: float = 0.5
):
    """
    Wait for a web element to be present in the DOM and visible.

    Args:
        driver: The Selenium WebDriver.
        by_locator: A tuple of (By, locator), e.g. (By.ID, "video-element")
        timeout: Maximum time to wait (in seconds)
        poll_frequency: How often to poll for the element (in seconds)

    Returns:
        The found web element.

    Raises:
        BrowserError: If the element isn't found within the timeout.
    """
    try:
        logger.debug(f"Waiting for element: {by_locator}")
        wait = WebDriverWait(driver, timeout, poll_frequency)
        element = wait.until(EC.visibility_of_element_located(by_locator))
        return element
    except TimeoutException:
        msg = f"Timed out after {timeout}s waiting for element: {by_locator}"
        logger.error(msg)
        raise BrowserError(msg)


def execute_javascript(driver, script: str):
    """
    Safely execute JavaScript in the browser context and return the result.

    Args:
        driver: The Selenium WebDriver.
        script: JavaScript code as a string.

    Returns:
        Result of the JavaScript execution.

    Raises:
        JavaScriptExecutionError: If there's a JS-related error.
    """
    try:
        return driver.execute_script(script)
    except JavascriptException as e:
        logger.error(f"JavaScript execution error: {str(e)}")
        raise JavaScriptExecutionError(
            f"Error executing JavaScript: {str(e)}",
            script_info=script,
            error_message=str(e)
        )


def capture_screenshot_as_base64(driver, element=None) -> str:
    """
    Capture a screenshot of the specified element (or full page if None) and return it as a base64 string.

    Args:
        driver: The Selenium WebDriver.
        element: A WebElement to screenshot (optional). If None, captures the full screen.

    Returns:
        A base64-encoded screenshot string.

    Raises:
        BrowserError: If the screenshot operation fails.
    """
    try:
        if element:
            return element.screenshot_as_base64
        else:
            return driver.get_screenshot_as_base64()
    except WebDriverException as e:
        msg = f"Failed to capture screenshot: {str(e)}"
        logger.error(msg)
        raise BrowserError(msg)


def highlight_element(driver, element, duration: float = 1.0, color: str = "yellow", border: int = 2):
    """
    Temporarily highlight a web element by changing its style.

    Args:
        driver: The Selenium WebDriver.
        element: A WebElement to highlight.
        duration: How long (in seconds) to keep it highlighted.
        color: Highlight color (CSS color string).
        border: Border thickness in pixels.
    """
    try:
        original_style = element.get_attribute("style")
        highlight_style = f"border: {border}px solid {color};"
        driver.execute_script("arguments[0].setAttribute('style', arguments[1]);", element, highlight_style)
        time.sleep(duration)
        driver.execute_script("arguments[0].setAttribute('style', arguments[1]);", element, original_style or "")
    except Exception as e:
        logger.debug(f"Highlight element failed: {str(e)}")


# === INTERNAL HELPERS ===

def _create_chrome_driver(headless: bool, extra_args: list, timeout: int):
    from selenium.webdriver.chrome.service import Service as ChromeService

    if ChromeDriverManager:
        service = ChromeService(ChromeDriverManager().install())
    else:
        service = ChromeService()  # fallback to PATH

    options = ChromeOptions()
    if headless:
        # The newer Chrome uses "--headless=new" for better performance
        options.add_argument("--headless=new")
    for arg in extra_args:
        options.add_argument(arg)

    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(timeout)
    return driver


def _create_firefox_driver(headless: bool, extra_args: list, timeout: int):
    from selenium.webdriver.firefox.service import Service as FirefoxService

    if GeckoDriverManager:
        service = FirefoxService(GeckoDriverManager().install())
    else:
        service = FirefoxService()

    options = FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    for arg in extra_args:
        options.add_argument(arg)

    driver = webdriver.Firefox(service=service, options=options)
    driver.set_page_load_timeout(timeout)
    return driver


def _create_edge_driver(headless: bool, extra_args: list, timeout: int):
    from selenium.webdriver.edge.service import Service as EdgeService

    if EdgeChromiumDriverManager:
        service = EdgeService(EdgeChromiumDriverManager().install())
    else:
        service = EdgeService()

    options = EdgeOptions()
    if headless:
        options.add_argument("--headless")
    for arg in extra_args:
        options.add_argument(arg)

    driver = webdriver.Edge(service=service, options=options)
    driver.set_page_load_timeout(timeout)
    return driver
