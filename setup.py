import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read the long description from the README file.
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="youtube-frame-extractor",
    version="1.0.0",
    author="Harshil Bhandari",
    author_email="your-email@example.com",
    description="Extract and analyze frames from YouTube videos using browser automation, AI-powered analysis, and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Harshil7875/youtube-scrape-videoframe-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/Harshil7875/youtube-scrape-videoframe-analysis/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="YouTube frame extraction computer vision AI CLIP OCR Selenium",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "yt-dlp>=2023.3.4",
        "selenium>=4.1.0",
        "webdriver-manager>=3.8.0",
        "opencv-python>=4.7.0.72",
        "Pillow>=9.4.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.27.0",
        "pytesseract>=0.3.10",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.2",
        "scipy>=1.10.1",
        "moviepy>=1.0.3",
        "typer[all]>=0.7.0",
        "rich>=13.3.0",
        "pydantic>=1.10.0",
        "boto3>=1.26.0",
        "google-cloud-storage>=2.7.0",
        "requests>=2.28.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "aiohttp>=3.8.4",
        "asyncio>=3.4.3"
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "mkdocs>=1.4.2",
            "mkdocs-material>=9.1.0",
            "mkdocstrings>=0.20.0",
            "mkdocstrings-python>=0.8.3",
            "jupyter>=1.0.0",
            "ipykernel>=6.22.0",
            "matplotlib>=3.7.1",
            "seaborn>=0.12.2"
        ],
    },
    include_package_data=True,
)
