import pytest
from youtube_frame_extractor.utils import browser

# Assume that utils/browser.py defines a helper function, e.g.:
# def scroll_into_view(driver, element):
#     driver["scrolled"] = True
#     return True

def test_scroll_into_view():
    # Create dummy driver and dummy element
    dummy_driver = {}
    dummy_element = {"id": "dummy"}
    
    # Call the utility function.
    result = browser.scroll_into_view(dummy_driver, dummy_element)
    
    # Verify that the driver now indicates the element has been scrolled into view.
    assert result is True
    assert dummy_driver.get("scrolled") is True
