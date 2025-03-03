import os
import pytest
from pathlib import Path
from youtube_frame_extractor.config import load_settings, get_settings

# Fixture to load the global settings for tests.
# If a test-specific configuration file is provided via the TEST_CONFIG_FILE env var,
# it will be used; otherwise, default settings are loaded.
@pytest.fixture(scope="session")
def settings():
    config_file = os.getenv("TEST_CONFIG_FILE", None)
    if config_file and Path(config_file).exists():
        return load_settings(config_file)
    else:
        return get_settings()

# Fixture to create a temporary output directory for tests.
@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    # Create a temporary directory for test outputs (shared for the session)
    output_dir = tmp_path_factory.mktemp("test_output")
    return output_dir

# Automatically set common environment variables for tests.
# This fixture is applied to all tests.
@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch):
    # Ensure browsers run in headless mode during tests
    monkeypatch.setenv("YFE_BROWSER_HEADLESS", "true")
    # Set logging to DEBUG to capture detailed logs during testing
    monkeypatch.setenv("YFE_LOGGING_LEVEL", "DEBUG")
    # Set any other test-specific environment variables here
    # For example, you can set temporary paths or mock API keys
    # monkeypatch.setenv("YFE_STORAGE_OUTPUT_DIR", str(Path.cwd() / "test_output"))
